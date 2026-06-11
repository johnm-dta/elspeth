"""DataFlowRepository: token/row lifecycle, graph structure, and error recording.

Atomic transactions in fork/coalesce/expand preserved via direct
LandscapeDB.connection() usage.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from sqlalchemy import and_, select
from sqlalchemy.engine import Row as SQLAlchemyRow

from elspeth.contracts import (
    ContractAuditRecord,
    Determinism,
    Edge,
    Node,
    NodeType,
    NonCanonicalMetadata,
    RoutingMode,
    Row,
    Token,
    TokenOutcome,
    TransformErrorReason,
    TransformErrorRecord,
    ValidationErrorRecord,
    ValidationErrorWithContract,
)
from elspeth.contracts.audit import _TERMINAL_PAIR_FIELD_CONSTRAINTS, DISCARD_SINK_NAME, TokenRef
from elspeth.contracts.enums import BatchStatus, NodeStateStatus, TerminalOutcome, TerminalPath
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.freeze import deep_thaw
from elspeth.contracts.hashing import repr_hash
from elspeth.contracts.schema_contract import SchemaContract
from elspeth.core.canonical import canonical_json, stable_hash
from elspeth.core.checkpoint.serialization import checkpoint_dumps
from elspeth.core.landscape._database_ops import DatabaseOps, LandscapeConnectionProvider
from elspeth.core.landscape._helpers import generate_id, now
from elspeth.core.landscape.model_loaders import (
    EdgeLoader,
    NodeLoader,
    TokenOutcomeLoader,
    TransformErrorLoader,
    ValidationErrorLoader,
)
from elspeth.core.landscape.schema import (
    artifacts_table,
    batches_table,
    edges_table,
    node_states_table,
    nodes_table,
    rows_table,
    token_outcomes_table,
    token_parents_table,
    tokens_table,
    transform_errors_table,
    validation_errors_table,
)

if TYPE_CHECKING:
    from elspeth.contracts.errors import ContractViolation
    from elspeth.contracts.payload_store import PayloadStore
    from elspeth.contracts.schema import SchemaConfig
    from elspeth.contracts.schema_contract import PipelineRow


class DataFlowRepository:
    """Records data flow: tokens, rows, graph structure, and errors.

    Atomic transactions in fork/coalesce/expand preserved via direct
    LandscapeDB.connection() usage.

    NOTE: nodes table has composite PK (node_id, run_id). Always filter
    by both columns when querying individual nodes.
    """

    def __init__(
        self,
        db: LandscapeConnectionProvider,
        ops: DatabaseOps,
        *,
        token_outcome_loader: TokenOutcomeLoader,
        node_loader: NodeLoader,
        edge_loader: EdgeLoader,
        validation_error_loader: ValidationErrorLoader,
        transform_error_loader: TransformErrorLoader,
        payload_store: PayloadStore | None = None,
    ) -> None:
        self._db = db
        self._ops = ops
        self._token_outcome_loader = token_outcome_loader
        self._node_loader = node_loader
        self._edge_loader = edge_loader
        self._validation_error_loader = validation_error_loader
        self._transform_error_loader = transform_error_loader
        self._payload_store = payload_store

    # ── Token recording: private helpers ─────────────────────────────────

    def _sanitize_node_config_for_audit(self, config: Mapping[str, object]) -> Mapping[str, object]:
        """Return an audit-safe node config with secrets fingerprinted."""
        import os

        from elspeth.core.config import _fingerprint_secrets

        thawed = deep_thaw(config)
        if type(thawed) is not dict:
            raise TypeError(f"Node config must thaw to dict[str, object], got {type(thawed).__name__}: {thawed!r}")

        allow_raw = False
        if "ELSPETH_ALLOW_RAW_SECRETS" in os.environ:
            allow_raw = os.environ["ELSPETH_ALLOW_RAW_SECRETS"].lower() == "true"
        return _fingerprint_secrets(thawed, fail_if_no_key=not allow_raw)

    # ── Tier-3 external-data audit serialization (coerce-and-record) ──────

    def _canonical_or_recorded_hash(self, data: Any) -> str:
        """Hash external row data, recording a non-canonical fallback on failure.

        Tier-3 boundary. ``data`` is external-origin (a source/quarantined row or
        transform-result payload) that may legitimately contain NaN/Infinity or
        otherwise non-canonical values which ``stable_hash`` rejects. This is the
        sanctioned coerce-and-record boundary: we return the canonical hash when
        possible, and otherwise return an explicit ``repr_hash`` fallback. The
        absence of a canonical hash is recorded as a distinct (repr-based) value,
        never fabricated and never silently swallowed.
        """
        try:
            return stable_hash(data)
        except (ValueError, TypeError):
            # Non-canonical external data: return the explicit repr-based fallback
            # so the audit row still records what we received.
            return repr_hash(data)

    def _canonical_or_recorded_json(self, data: Any) -> str:
        """Serialize external row data, recording a non-canonical fallback on failure.

        Tier-3 boundary companion to :meth:`_canonical_or_recorded_hash`. Returns
        canonical JSON when ``data`` is canonicalizable, otherwise returns an
        explicit :class:`NonCanonicalMetadata` envelope (repr + type + error)
        serialized as JSON. The failure is recorded as structured metadata in the
        audit trail, not discarded.
        """
        try:
            return canonical_json(data)
        except (ValueError, TypeError) as exc:
            # Non-canonical external data: return an explicit structured envelope
            # capturing what we saw (repr, type, serialization error).
            return json.dumps(NonCanonicalMetadata.from_error(data, exc).to_dict(), allow_nan=False)

    def _canonical_or_recorded_error_details_json(self, error_details: Any) -> str:
        """Serialize transform error_details, recording a non-canonical fallback.

        Tier-3 boundary. ``error_details`` originates from transform results and
        may carry arbitrary row-derived data (NaN/Infinity, non-serializable
        objects from exception context). Returns canonical JSON when possible,
        otherwise an explicit ``__non_canonical__`` envelope (repr + error) — the
        failure is recorded as structured audit data, not discarded.
        """
        try:
            return canonical_json(error_details)
        except (ValueError, TypeError) as exc:
            return json.dumps(
                {
                    "__non_canonical__": True,
                    "repr": repr(error_details)[:500],
                    "serialization_error": str(exc),
                },
                allow_nan=False,
            )

    def _canonical_or_recorded_repr_payload(self, data: Any) -> str:
        """Serialize a quarantined payload, recording a repr sentinel on failure.

        Tier-3 boundary for the payload store. ``data`` is quarantined external
        data that may contain non-canonical values. Returns canonical JSON when
        possible, otherwise an explicit ``{"_repr": repr(data)}`` sentinel that
        the query repository recognizes on read-back — the absence of a canonical
        payload is recorded, never fabricated or swallowed.
        """
        try:
            return canonical_json(data)
        except (ValueError, TypeError):
            return json.dumps({"_repr": repr(data)}, allow_nan=False)

    def _resolve_run_id_for_row(self, row_id: str) -> str:
        """Resolve the run_id that owns a given row_id.

        This is Tier 1 (our data). If the row doesn't exist, it's a bug
        in our code or database corruption -- crash immediately.

        Args:
            row_id: Row ID to look up

        Returns:
            run_id that owns the row

        Raises:
            AuditIntegrityError: If row_id not found (Tier 1 corruption)
        """
        query = select(rows_table.c.run_id).where(rows_table.c.row_id == row_id)
        result = self._ops.execute_fetchone(query)
        if result is None:
            raise AuditIntegrityError(
                f"Token references row_id={row_id!r} which does not exist in the rows table. "
                f"This is Tier 1 data corruption -- the row should have been created before any token."
            )
        run_id: str = result.run_id
        return run_id

    def resolve_row_ingest_sequence(self, row_id: str) -> int:
        """Resolve a row's global ingest ordering for scheduler fairness.

        This is Tier 1 audit data. A token continuation can only exist for a
        persisted row, so a missing row is corruption or an orchestration bug.
        """
        query = select(rows_table.c.ingest_sequence).where(rows_table.c.row_id == row_id)
        result = self._ops.execute_fetchone(query)
        if result is None:
            raise AuditIntegrityError(
                f"Cannot schedule work for row_id={row_id!r}: row does not exist. "
                "This is Tier 1 data corruption -- scheduler work must reference persisted rows."
            )
        ingest_sequence: int = result.ingest_sequence
        return ingest_sequence

    def _resolve_token_ownership(self, token_id: str) -> tuple[str, str]:
        """Resolve the (row_id, run_id) that owns a given token_id.

        Looks up token -> row_id, then row -> run_id. This is Tier 1 (our data).
        If the token or its row doesn't exist, it's a bug or database corruption.

        Args:
            token_id: Token ID to look up

        Returns:
            Tuple of (row_id, run_id) that own the token

        Raises:
            AuditIntegrityError: If token or its row not found (Tier 1 corruption)
        """
        query = select(tokens_table.c.row_id, tokens_table.c.run_id).where(tokens_table.c.token_id == token_id)
        result = self._ops.execute_fetchone(query)
        if result is None:
            raise AuditIntegrityError(
                f"Token {token_id!r} does not exist in the tokens table. "
                f"This is Tier 1 data corruption -- the token should have been created before recording outcomes."
            )
        return result.row_id, result.run_id

    def _prepare_source_row_record(
        self,
        run_id: str,
        source_node_id: str,
        row_index: int,
        data: Mapping[str, object],
        *,
        source_row_index: int | None,
        ingest_sequence: int | None,
        row_id: str | None,
        quarantined: bool,
    ) -> Row:
        row_id = row_id or generate_id()
        missing_identity_fields = []
        if source_row_index is None:
            missing_identity_fields.append("source_row_index")
        if ingest_sequence is None:
            missing_identity_fields.append("ingest_sequence")
        if missing_identity_fields:
            raise AuditIntegrityError(
                f"create_row requires explicit source-scoped identity for run_id={run_id!r} row_id={row_id!r} "
                f"source_node_id={source_node_id!r}; missing {', '.join(missing_identity_fields)}. "
                "Do not fabricate source_row_index or ingest_sequence from row_index."
            )
        assert source_row_index is not None
        assert ingest_sequence is not None

        # Quarantined rows are Tier-3 external data that may contain non-canonical
        # values (NaN, Infinity). The coerce-and-record helper returns the canonical
        # hash or an explicit repr-based fallback per canonical.py docs. Non-quarantined
        # rows are trusted to be canonical — let stable_hash crash on any anomaly.
        if quarantined:
            data_hash = self._canonical_or_recorded_hash(data)
        else:
            data_hash = stable_hash(data)

        timestamp = now()

        # Landscape owns payload persistence - serialize and store if configured
        final_payload_ref: str | None = None
        if self._payload_store is not None:
            # Canonical JSON handles pandas/numpy/Decimal/datetime types.
            # For quarantined data, fall back to json.dumps(repr()) if
            # canonical serialization fails on non-canonical values.
            if quarantined:
                payload_bytes = self._canonical_or_recorded_repr_payload(data).encode("utf-8")
            else:
                payload_bytes = canonical_json(data).encode("utf-8")
            final_payload_ref = self._payload_store.store(payload_bytes)

        return Row(
            row_id=row_id,
            run_id=run_id,
            source_node_id=source_node_id,
            row_index=row_index,
            source_row_index=source_row_index,
            ingest_sequence=ingest_sequence,
            source_data_hash=data_hash,
            source_data_ref=final_payload_ref,
            created_at=timestamp,
        )

    @staticmethod
    def _row_insert_values(row: Row) -> dict[str, object]:
        assert row.source_row_index is not None
        assert row.ingest_sequence is not None
        return {
            "row_id": row.row_id,
            "run_id": row.run_id,
            "source_node_id": row.source_node_id,
            "row_index": row.row_index,
            "source_row_index": row.source_row_index,
            "ingest_sequence": row.ingest_sequence,
            "source_data_hash": row.source_data_hash,
            "source_data_ref": row.source_data_ref,
            "created_at": row.created_at,
        }

    def _validate_token_run_ownership(self, ref: TokenRef) -> None:
        """Validate that a token belongs to the specified run.

        Per Tier 1 trust model: cross-run contamination of audit records is
        evidence tampering. Crash immediately if the invariant is violated.

        Args:
            ref: TokenRef to validate — token_id must belong to run_id

        Raises:
            AuditIntegrityError: If token does not belong to the specified run
        """
        _row_id, actual_run_id = self._resolve_token_ownership(ref.token_id)
        if actual_run_id != ref.run_id:
            raise AuditIntegrityError(
                f"Cross-run contamination prevented: token {ref.token_id!r} belongs to "
                f"run {actual_run_id!r}, but caller supplied run_id={ref.run_id!r}. "
                f"This would corrupt the audit trail by attributing records to the wrong run."
            )

    def _validate_token_row_ownership(self, token_id: str, row_id: str) -> None:
        """Validate that a token belongs to the specified row.

        Per Tier 1 trust model: cross-row lineage corruption makes the audit
        trail unreliable. Crash immediately if the invariant is violated.

        Args:
            token_id: Token to validate
            row_id: Expected row ID

        Raises:
            AuditIntegrityError: If token does not belong to the specified row
        """
        actual_row_id, _run_id = self._resolve_token_ownership(token_id)
        if actual_row_id != row_id:
            raise AuditIntegrityError(
                f"Cross-row lineage corruption prevented: token {token_id!r} belongs to "
                f"row {actual_row_id!r}, but caller supplied row_id={row_id!r}. "
                f"This would create invalid parent-child lineage across different rows."
            )

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

    # ── Token recording: public methods ──────────────────────────────────

    def create_row(
        self,
        run_id: str,
        source_node_id: str,
        row_index: int,
        data: Mapping[str, object],
        *,
        source_row_index: int | None = None,
        ingest_sequence: int | None = None,
        row_id: str | None = None,
        quarantined: bool = False,
    ) -> Row:
        """Create a source row record.

        Args:
            run_id: Run this row belongs to
            source_node_id: Source node that loaded this row
            row_index: Legacy/display row position
            data: Row data for hashing and optional storage
            source_row_index: Position within the source (0-indexed)
            ingest_sequence: Monotonic run-wide ingest order
            row_id: Optional row ID (generated if not provided)
            quarantined: If True, data is Tier-3 external data that may contain
                non-canonical values (NaN, Infinity). Uses repr_hash fallback.

        Returns:
            Row model

        Note:
            Payload persistence is handled by PayloadStore, not callers.
            If self._payload_store is configured, the method will:
            1. Serialize data using canonical_json (handles pandas/numpy/datetime/Decimal)
            2. Store in payload store
            3. Record reference in audit trail

            This ensures Landscape owns its audit format end-to-end.
        """
        row = self._prepare_source_row_record(
            run_id=run_id,
            source_node_id=source_node_id,
            row_index=row_index,
            data=data,
            source_row_index=source_row_index,
            ingest_sequence=ingest_sequence,
            row_id=row_id,
            quarantined=quarantined,
        )

        self._ops.execute_insert(rows_table.insert().values(**self._row_insert_values(row)))

        return row

    def create_row_with_token(
        self,
        run_id: str,
        source_node_id: str,
        row_index: int,
        data: Mapping[str, object],
        *,
        source_row_index: int | None = None,
        ingest_sequence: int | None = None,
        row_id: str | None = None,
        token_id: str | None = None,
        quarantined: bool = False,
    ) -> tuple[Row, Token]:
        """Create a source row and its initial token in one audit transaction."""
        row = self._prepare_source_row_record(
            run_id=run_id,
            source_node_id=source_node_id,
            row_index=row_index,
            data=data,
            source_row_index=source_row_index,
            ingest_sequence=ingest_sequence,
            row_id=row_id,
            quarantined=quarantined,
        )
        token = Token(
            token_id=token_id or generate_id(),
            row_id=row.row_id,
            run_id=run_id,
            created_at=row.created_at,
        )

        with self._db.write_connection() as conn:
            result = conn.execute(rows_table.insert().values(**self._row_insert_values(row)))
            if result.rowcount == 0:
                raise AuditIntegrityError(f"create_row_with_token: row INSERT affected zero rows (row_id={row.row_id})")
            result = conn.execute(
                tokens_table.insert().values(
                    token_id=token.token_id,
                    row_id=token.row_id,
                    run_id=token.run_id,
                    fork_group_id=token.fork_group_id,
                    join_group_id=token.join_group_id,
                    branch_name=token.branch_name,
                    created_at=token.created_at,
                )
            )
            if result.rowcount == 0:
                raise AuditIntegrityError(f"create_row_with_token: token INSERT affected zero rows (token_id={token.token_id})")

        return row, token

    def create_token(
        self,
        row_id: str,
        *,
        token_id: str | None = None,
        branch_name: str | None = None,
        fork_group_id: str | None = None,
        join_group_id: str | None = None,
    ) -> Token:
        """Create a token (row instance in DAG path).

        Derives run_id from the row record to guarantee run ownership
        consistency. The tokens table stores run_id to enable composite
        FK enforcement on downstream tables.

        Args:
            row_id: Source row this token represents
            token_id: Optional token ID (generated if not provided)
            branch_name: Optional branch name (for forked tokens)
            fork_group_id: Optional fork group (links siblings)
            join_group_id: Optional join group (links merged tokens)

        Returns:
            Token model

        Raises:
            AuditIntegrityError: If row_id does not exist (Tier 1 corruption)
        """
        token_id = token_id or generate_id()
        timestamp = now()

        # Derive run_id from the row record (Tier 1 -- our data, must exist)
        run_id = self._resolve_run_id_for_row(row_id)

        # Validate lineage metadata invariants (Tier 1 write-side enforcement)
        # The read side (explain) assumes these are mutually exclusive.
        group_ids = [gid for gid in (fork_group_id, join_group_id) if gid is not None]
        if len(group_ids) > 1:
            raise AuditIntegrityError(
                f"create_token: conflicting lineage metadata — at most one of "
                f"fork_group_id, join_group_id may be set. "
                f"Got fork_group_id={fork_group_id!r}, join_group_id={join_group_id!r}"
            )

        # branch_name requires fork_group_id (it names which fork branch this token is on)
        if branch_name is not None and fork_group_id is None:
            raise AuditIntegrityError(f"create_token: branch_name={branch_name!r} requires fork_group_id to be set")

        # Reject empty-string group IDs (should be None, not "")
        for name, value in [("fork_group_id", fork_group_id), ("join_group_id", join_group_id)]:
            if value is not None and not value.strip():
                raise AuditIntegrityError(f"create_token: {name} must be None or non-empty, got {value!r}")

        token = Token(
            token_id=token_id,
            row_id=row_id,
            fork_group_id=fork_group_id,
            join_group_id=join_group_id,
            branch_name=branch_name,
            created_at=timestamp,
            run_id=run_id,
        )

        self._ops.execute_insert(
            tokens_table.insert().values(
                token_id=token.token_id,
                row_id=token.row_id,
                run_id=run_id,
                fork_group_id=token.fork_group_id,
                join_group_id=token.join_group_id,
                branch_name=token.branch_name,
                created_at=token.created_at,
            )
        )

        return token

    def fork_token(
        self,
        parent_ref: TokenRef,
        row_id: str,
        branches: list[str],
        *,
        step_in_pipeline: int | None = None,
    ) -> tuple[list[Token], str]:
        """Fork a token to multiple branches.

        ATOMIC: Creates children AND records parent FORKED outcome in single transaction.
        Stores branch contract for recovery validation.

        Validates that parent token belongs to the specified row_id and run_id
        before any writes. Cross-run/cross-row contamination crashes immediately
        per Tier 1 trust model.

        Args:
            parent_ref: TokenRef bundling parent token_id and run_id
            row_id: Row ID (same for all children)
            branches: List of branch names (must have at least one)
            step_in_pipeline: Step in the DAG where the fork occurs

        Returns:
            Tuple of (child Token models, fork_group_id)

        Raises:
            ValueError: If branches is empty (defense-in-depth for audit integrity)
            AuditIntegrityError: If parent token does not belong to specified run/row
        """
        # Defense-in-depth: validate even though RoutingAction.fork_to_paths()
        # already validates. Per CLAUDE.md "no silent drops" - empty forks
        # would cause tokens to disappear without audit trail.
        if not branches:
            raise ValueError("fork_token requires at least one branch")

        # Validate parent token ownership before any writes (Tier 1 invariant)
        self._validate_token_run_ownership(parent_ref)
        self._validate_token_row_ownership(parent_ref.token_id, row_id)

        fork_group_id = generate_id()
        children = []

        with self._db.write_connection() as conn:
            # 1. Create child tokens
            for ordinal, branch_name in enumerate(branches):
                child_id = generate_id()
                timestamp = now()

                # Create child token (run_id derived from parent -- already validated)
                result = conn.execute(
                    tokens_table.insert().values(
                        token_id=child_id,
                        row_id=row_id,
                        run_id=parent_ref.run_id,
                        fork_group_id=fork_group_id,
                        branch_name=branch_name,
                        step_in_pipeline=step_in_pipeline,
                        created_at=timestamp,
                    )
                )
                if result.rowcount == 0:
                    raise AuditIntegrityError(
                        f"fork_token: child token INSERT affected zero rows (token_id={child_id}, branch={branch_name!r})"
                    )

                # Record parent relationship
                result = conn.execute(
                    token_parents_table.insert().values(
                        token_id=child_id,
                        parent_token_id=parent_ref.token_id,
                        ordinal=ordinal,
                    )
                )
                if result.rowcount == 0:
                    raise AuditIntegrityError(
                        f"fork_token: token_parent INSERT affected zero rows (child={child_id}, parent={parent_ref.token_id})"
                    )

                children.append(
                    Token(
                        token_id=child_id,
                        row_id=row_id,
                        fork_group_id=fork_group_id,
                        branch_name=branch_name,
                        step_in_pipeline=step_in_pipeline,
                        created_at=timestamp,
                        run_id=parent_ref.run_id,
                    )
                )

            # 2. Record parent FORKED outcome in SAME transaction (atomic)
            outcome_id = f"out_{generate_id()[:12]}"
            result = conn.execute(
                token_outcomes_table.insert().values(
                    outcome_id=outcome_id,
                    run_id=parent_ref.run_id,
                    token_id=parent_ref.token_id,
                    outcome=TerminalOutcome.TRANSIENT.value,
                    path=TerminalPath.FORK_PARENT.value,
                    completed=1,
                    recorded_at=now(),
                    fork_group_id=fork_group_id,
                    expected_branches_json=json.dumps(branches, allow_nan=False),
                )
            )
            if result.rowcount == 0:
                raise AuditIntegrityError(
                    f"fork_token: FORKED outcome INSERT affected zero rows (parent={parent_ref.token_id}, outcome_id={outcome_id})"
                )

        return children, fork_group_id

    def coalesce_tokens(
        self,
        parent_refs: list[TokenRef],
        row_id: str,
        merged_payload: Mapping[str, object],
        *,
        merged_contract: SchemaContract,
        step_in_pipeline: int | None = None,
    ) -> Token:
        """Coalesce multiple tokens into one (join operation).

        Creates a new token representing the merged result.
        Records all parent relationships.
        Persists a {data, contract} envelope to the payload store so the merged token
        is reconstructable on resume without re-executing the merge strategy and without
        any nodes-table lookup (ADDENDUM 3 — nodes.output_contract_json is NULL in prod
        for non-source nodes; the envelope is self-contained).

        Validates that all parent tokens belong to the specified row_id and
        that they all share the same run_id. Cross-run/cross-row contamination
        crashes immediately per Tier 1 trust model.

        Args:
            parent_refs: TokenRefs for tokens being merged (bundled token_id + run_id)
            row_id: Row ID for the merged token
            merged_payload: The merged row data dict to persist (Tier-1 audit write).
            merged_contract: The SchemaContract under which the merged token was produced.
                Serialised into the envelope via to_checkpoint_format() so recovery can
                restore a faithful PipelineRow without any nodes-table lookup.
                Uses checkpoint_dumps (type-faithful: preserves datetime via type-tagged
                envelopes) — NOT canonical_json, which would stringify datetime.
            step_in_pipeline: Step in the DAG where the coalesce occurs

        Returns:
            Merged Token model

        Raises:
            AuditIntegrityError: If parent tokens do not belong to specified row,
                if parent tokens span multiple runs, or if no payload store is configured
        """
        if self._payload_store is None:
            raise AuditIntegrityError(
                "coalesce_tokens requires a configured payload store — the merged token's "
                "payload must be persisted for resume correctness (epoch 11 invariant). "
                "Pass payload_store= to DataFlowRepository or RecorderFactory."
            )
        if not parent_refs:
            raise AuditIntegrityError(
                "coalesce_tokens requires at least one parent token — a coalesce with zero parents creates an unexplainable audit state"
            )

        # Validate all parent tokens belong to the same row and run (Tier 1 invariant)
        run_id: str | None = None
        for ref in parent_refs:
            self._validate_token_row_ownership(ref.token_id, row_id)
            self._validate_token_run_ownership(ref)
            if run_id is None:
                run_id = ref.run_id
            elif ref.run_id != run_id:
                raise AuditIntegrityError(
                    f"Cross-run contamination prevented in coalesce: parent token {ref.token_id!r} "
                    f"belongs to run {ref.run_id!r}, but other parents belong to run {run_id!r}. "
                    f"All parent tokens in a coalesce must belong to the same run."
                )

        # Derive run_id from row if no parents (edge case: shouldn't happen in practice)
        if run_id is None:
            run_id = self._resolve_run_id_for_row(row_id)

        join_group_id = generate_id()
        token_id = generate_id()
        timestamp = now()

        # Persist a self-contained {data, contract} envelope before the DB write so
        # the token_data_ref is available atomically at INSERT time.
        # The envelope carries both the row data and its SchemaContract so recovery can
        # reconstruct a faithful PipelineRow without any nodes-table lookup (ADDENDUM 3:
        # nodes.output_contract_json is NULL for non-source nodes in production).
        # checkpoint_dumps is type-faithful (datetime survives as datetime, not a
        # string) — canonical_json would stringify datetime and destroy Tier-1 fidelity.
        # Crash on store failure: a merged token with no persisted payload is
        # unreconstructable on resume (Tier-1 audit invariant).
        envelope = {"data": dict(merged_payload), "contract": merged_contract.to_checkpoint_format()}
        token_data_ref = self._payload_store.store(checkpoint_dumps(envelope).encode("utf-8"))

        with self._db.write_connection() as conn:
            # Create merged token
            result = conn.execute(
                tokens_table.insert().values(
                    token_id=token_id,
                    row_id=row_id,
                    run_id=run_id,
                    join_group_id=join_group_id,
                    step_in_pipeline=step_in_pipeline,
                    created_at=timestamp,
                    token_data_ref=token_data_ref,
                )
            )
            if result.rowcount == 0:
                raise AuditIntegrityError(f"coalesce_tokens: merged token INSERT affected zero rows (token_id={token_id})")

            # Record all parent relationships
            for ordinal, ref in enumerate(parent_refs):
                result = conn.execute(
                    token_parents_table.insert().values(
                        token_id=token_id,
                        parent_token_id=ref.token_id,
                        ordinal=ordinal,
                    )
                )
                if result.rowcount == 0:
                    raise AuditIntegrityError(
                        f"coalesce_tokens: token_parent INSERT affected zero rows (child={token_id}, parent={ref.token_id})"
                    )

        return Token(
            token_id=token_id,
            row_id=row_id,
            join_group_id=join_group_id,
            step_in_pipeline=step_in_pipeline,
            created_at=timestamp,
            run_id=run_id,
            token_data_ref=token_data_ref,
        )

    def expand_token(
        self,
        parent_ref: TokenRef,
        row_id: str,
        child_payloads: Sequence[Mapping[str, object]],
        *,
        output_contract: SchemaContract,
        step_in_pipeline: int | None = None,
        record_parent_outcome: bool = True,
    ) -> tuple[list[Token], str]:
        """Expand a token into multiple child tokens (deaggregation).

        ATOMIC: Creates children AND optionally records parent EXPANDED outcome
        in single transaction.

        Validates that parent token belongs to the specified row_id and run_id
        before any writes. Cross-run/cross-row contamination crashes immediately
        per Tier 1 trust model.

        Creates N child tokens from a single parent for 1->N expansion.
        All children share the same row_id (same source row) and are
        linked to the parent via token_parents table.

        Unlike fork_token (parallel DAG paths with branch names), expand_token
        creates sequential children for deaggregation transforms.

        Each child's {data, contract} envelope is persisted to the payload store before
        the DB write so token_data_ref is written atomically at INSERT time. This
        makes each expanded child self-contained and reconstructable on resume without
        re-executing the deaggregation transform and without any nodes-table lookup
        (ADDENDUM 3 — nodes.output_contract_json is NULL for non-source nodes in prod).

        Args:
            parent_ref: TokenRef bundling parent token_id and run_id
            row_id: Row ID (same for all children)
            child_payloads: Per-child row data dicts (one per expanded child).
                Must have at least 1 element. Uses checkpoint_dumps (type-faithful:
                preserves datetime via type-tagged envelopes) — NOT canonical_json.
            output_contract: The SchemaContract shared by all expanded children (from
                TransformResult.contract, locked before expansion). Serialised into each
                child's envelope so recovery can restore a faithful PipelineRow.
            step_in_pipeline: Step where expansion occurs (optional)
            record_parent_outcome: If True (default), record EXPANDED outcome for parent.
                Set to False for batch aggregation where parent gets CONSUMED_IN_BATCH.

        Returns:
            Tuple of (child Token list, expand_group_id)

        Raises:
            ValueError: If child_payloads is empty
            AuditIntegrityError: If parent token does not belong to specified run/row,
                or if no payload store is configured
        """
        count = len(child_payloads)
        if count < 1:
            raise ValueError("expand_token requires at least 1 child payload")

        if self._payload_store is None:
            raise AuditIntegrityError(
                "expand_token requires a configured payload store — each expanded child's "
                "payload must be persisted for resume correctness (epoch 11 invariant). "
                "Pass payload_store= to DataFlowRepository or RecorderFactory."
            )

        # Validate parent token ownership before any writes (Tier 1 invariant)
        self._validate_token_run_ownership(parent_ref)
        self._validate_token_row_ownership(parent_ref.token_id, row_id)

        expand_group_id = generate_id()
        children = []

        # Persist each child's {data, contract} envelope BEFORE the DB transaction so
        # the token_data_ref values are ready to write atomically at INSERT time.
        # The envelope is self-contained: recovery can restore a faithful PipelineRow
        # without any nodes-table lookup (ADDENDUM 3 — nodes.output_contract_json is
        # NULL for non-source nodes in production).
        # checkpoint_dumps is type-faithful (datetime preserved as datetime, not
        # stringified) — canonical_json would destroy Tier-1 fidelity.
        # Crash on store failure: a child token with no persisted payload is
        # unreconstructable on resume (epoch 11 invariant).
        contract_fmt = output_contract.to_checkpoint_format()
        child_data_refs = [
            self._payload_store.store(checkpoint_dumps({"data": dict(payload), "contract": contract_fmt}).encode("utf-8"))
            for payload in child_payloads
        ]

        with self._db.write_connection() as conn:
            for ordinal, payload_ref in enumerate(child_data_refs):
                child_id = generate_id()
                timestamp = now()

                # Create child token with expand_group_id (run_id from parent -- already validated)
                result = conn.execute(
                    tokens_table.insert().values(
                        token_id=child_id,
                        row_id=row_id,
                        run_id=parent_ref.run_id,
                        expand_group_id=expand_group_id,
                        step_in_pipeline=step_in_pipeline,
                        created_at=timestamp,
                        token_data_ref=payload_ref,
                    )
                )
                if result.rowcount == 0:
                    raise AuditIntegrityError(
                        f"expand_token: child token INSERT affected zero rows (token_id={child_id}, ordinal={ordinal})"
                    )

                # Record parent relationship
                result = conn.execute(
                    token_parents_table.insert().values(
                        token_id=child_id,
                        parent_token_id=parent_ref.token_id,
                        ordinal=ordinal,
                    )
                )
                if result.rowcount == 0:
                    raise AuditIntegrityError(
                        f"expand_token: token_parent INSERT affected zero rows (child={child_id}, parent={parent_ref.token_id})"
                    )

                children.append(
                    Token(
                        token_id=child_id,
                        row_id=row_id,
                        expand_group_id=expand_group_id,
                        step_in_pipeline=step_in_pipeline,
                        created_at=timestamp,
                        run_id=parent_ref.run_id,
                        token_data_ref=payload_ref,
                    )
                )

            # Optionally record parent EXPANDED outcome in SAME transaction (atomic)
            # This eliminates the crash window where children exist but parent
            # outcome is not yet recorded.
            #
            # Set record_parent_outcome=False for batch aggregation where the
            # parent token gets CONSUMED_IN_BATCH instead of EXPANDED.
            if record_parent_outcome:
                outcome_id = f"out_{generate_id()[:12]}"
                result = conn.execute(
                    token_outcomes_table.insert().values(
                        outcome_id=outcome_id,
                        run_id=parent_ref.run_id,
                        token_id=parent_ref.token_id,
                        outcome=TerminalOutcome.TRANSIENT.value,
                        path=TerminalPath.EXPAND_PARENT.value,
                        completed=1,
                        recorded_at=now(),
                        expand_group_id=expand_group_id,
                        # Store expected count for recovery validation
                        expected_branches_json=json.dumps({"count": count}, allow_nan=False),
                    )
                )
                if result.rowcount == 0:
                    raise AuditIntegrityError(
                        f"expand_token: EXPANDED outcome INSERT affected zero rows (parent={parent_ref.token_id}, outcome_id={outcome_id})"
                    )

        return children, expand_group_id

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
        self._validate_token_run_ownership(ref)
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

    # ── Graph recording: public methods ──────────────────────────────────

    def register_node(
        self,
        run_id: str,
        plugin_name: str,
        node_type: NodeType,
        plugin_version: str,
        config: Mapping[str, object],
        *,
        node_id: str | None = None,
        sequence: int | None = None,
        schema_hash: str | None = None,
        determinism: Determinism = Determinism.DETERMINISTIC,
        schema_config: SchemaConfig,
        source_file_hash: str | None = None,
        input_contract: SchemaContract | None = None,
        output_contract: SchemaContract | None = None,
    ) -> Node:
        """Register a node in the execution graph.

        Args:
            run_id: Run this node belongs to
            plugin_name: Name of the plugin (None for gates and coalesces, which are config-driven)
            node_type: NodeType enum (SOURCE, TRANSFORM, GATE, AGGREGATION, COALESCE, SINK)
            plugin_version: Version of the plugin (None for non-plugin nodes)
            config: Node configuration
            node_id: Optional node ID (generated if not provided)
            sequence: Position in pipeline
            schema_hash: Optional input/output schema hash
            determinism: Determinism enum (defaults to DETERMINISTIC)
            schema_config: Schema configuration for audit trail (WP-11.99)
            source_file_hash: Optional truncated SHA-256 hash of the plugin source file
            input_contract: Optional input schema contract (what node requires)
            output_contract: Optional output schema contract (what node guarantees)

        Returns:
            Node model
        """
        node_id = node_id or generate_id()
        audit_safe_config = self._sanitize_node_config_for_audit(config)
        config_json = canonical_json(audit_safe_config)
        config_hash = stable_hash(audit_safe_config)
        timestamp = now()

        # Extract schema info for audit (WP-11.99)
        schema_fields_json: str | None = None
        schema_fields_list: list[dict[str, object]] | None = None

        # Extract schema mode directly - no translation needed
        schema_mode = schema_config.mode
        if not schema_config.is_observed and schema_config.fields:
            # FieldDefinition.to_dict() returns dict[str, str | bool]
            # Cast each dict to wider type for storage
            field_dicts = [f.to_dict() for f in schema_config.fields]
            schema_fields_list = [dict(d) for d in field_dicts]
            schema_fields_json = canonical_json(field_dicts)

        # Convert schema contracts to audit records if provided
        input_contract_json: str | None = None
        output_contract_json: str | None = None
        if input_contract is not None:
            input_contract_json = ContractAuditRecord.from_contract(input_contract).to_json()
        if output_contract is not None:
            output_contract_json = ContractAuditRecord.from_contract(output_contract).to_json()

        node = Node(
            node_id=node_id,
            run_id=run_id,
            plugin_name=plugin_name,
            node_type=node_type,
            plugin_version=plugin_version,
            determinism=determinism,
            config_hash=config_hash,
            config_json=config_json,
            source_file_hash=source_file_hash,
            schema_hash=schema_hash,
            sequence_in_pipeline=sequence,
            registered_at=timestamp,
            schema_mode=schema_mode,
            schema_fields=schema_fields_list,
        )

        self._ops.execute_insert(
            nodes_table.insert().values(
                node_id=node.node_id,
                run_id=node.run_id,
                plugin_name=node.plugin_name,
                node_type=node.node_type,
                plugin_version=node.plugin_version,
                determinism=node.determinism,
                config_hash=node.config_hash,
                config_json=node.config_json,
                source_file_hash=node.source_file_hash,
                schema_hash=node.schema_hash,
                sequence_in_pipeline=node.sequence_in_pipeline,
                registered_at=node.registered_at,
                schema_mode=node.schema_mode,
                schema_fields_json=schema_fields_json,
                input_contract_json=input_contract_json,
                output_contract_json=output_contract_json,
            )
        )

        return node

    def register_edge(
        self,
        run_id: str,
        from_node_id: str,
        to_node_id: str,
        label: str,
        mode: RoutingMode,
        *,
        edge_id: str | None = None,
    ) -> Edge:
        """Register an edge in the execution graph.

        Args:
            run_id: Run this edge belongs to
            from_node_id: Source node
            to_node_id: Destination node
            label: Edge label ("continue", route name, etc.)
            mode: RoutingMode enum (MOVE or COPY)
            edge_id: Optional edge ID (generated if not provided)

        Returns:
            Edge model
        """
        edge_id = edge_id or generate_id()
        timestamp = now()

        edge = Edge(
            edge_id=edge_id,
            run_id=run_id,
            from_node_id=from_node_id,
            to_node_id=to_node_id,
            label=label,
            default_mode=mode,
            created_at=timestamp,
        )

        self._ops.execute_insert(
            edges_table.insert().values(
                edge_id=edge.edge_id,
                run_id=edge.run_id,
                from_node_id=edge.from_node_id,
                to_node_id=edge.to_node_id,
                label=edge.label,
                default_mode=edge.default_mode,
                created_at=edge.created_at,
            )
        )

        return edge

    def get_node(self, node_id: str, run_id: str) -> Node | None:
        """Get a node by its composite primary key (node_id, run_id).

        NOTE: The nodes table has a composite PK (node_id, run_id). The same
        node_id can exist in multiple runs, so run_id is required to identify
        the specific node.

        Args:
            node_id: Node ID to retrieve
            run_id: Run ID the node belongs to

        Returns:
            Node model or None if not found
        """
        query = select(nodes_table).where((nodes_table.c.node_id == node_id) & (nodes_table.c.run_id == run_id))
        row = self._ops.execute_fetchone(query)
        if row is None:
            return None
        return self._node_loader.load(row)

    def get_nodes(self, run_id: str) -> list[Node]:
        """Get all nodes for a run.

        Args:
            run_id: Run ID

        Returns:
            List of Node models, ordered by sequence (NULL sequences last)
        """
        query = (
            select(nodes_table)
            .where(nodes_table.c.run_id == run_id)
            # Use nullslast() for consistent NULL handling across databases
            # Nodes without sequence (e.g., dynamically added) sort last
            # Tiebreakers (registered_at, node_id) ensure deterministic ordering
            # for export signing when sequence_in_pipeline is NULL
            .order_by(
                nodes_table.c.sequence_in_pipeline.nullslast(),
                nodes_table.c.registered_at,
                nodes_table.c.node_id,
            )
        )
        rows = self._ops.execute_fetchall(query)
        return [self._node_loader.load(row) for row in rows]

    def get_node_contracts(
        self, run_id: str, node_id: str, *, allow_missing: bool = False
    ) -> tuple[SchemaContract | None, SchemaContract | None]:
        """Get input and output contracts for a node.

        Retrieves stored schema contracts and verifies integrity via hash.

        Args:
            run_id: Run ID the node belongs to
            node_id: Node ID to query
            allow_missing: If False (default), crash when node not found
                (Tier 1 invariant — our audit data must be present).
                Set to True only for external query paths (MCP, analysis).

        Returns:
            Tuple of (input_contract, output_contract), either may be None
            if the node exists but has no contracts recorded.

        Raises:
            AuditIntegrityError: If node not found and allow_missing is False
            ValueError: If stored contract fails integrity verification
        """
        query = select(
            nodes_table.c.input_contract_json,
            nodes_table.c.output_contract_json,
        ).where((nodes_table.c.node_id == node_id) & (nodes_table.c.run_id == run_id))
        row = self._ops.execute_fetchone(query)

        if row is None:
            if allow_missing:
                return None, None
            raise AuditIntegrityError(
                f"Node not found in audit trail: node_id={node_id!r}, run_id={run_id!r}. Expected node to exist (Tier 1 data)."
            )

        input_contract: SchemaContract | None = None
        output_contract: SchemaContract | None = None

        if row.input_contract_json is not None:
            audit_record = ContractAuditRecord.from_json(row.input_contract_json)
            input_contract = audit_record.to_schema_contract()

        if row.output_contract_json is not None:
            audit_record = ContractAuditRecord.from_json(row.output_contract_json)
            output_contract = audit_record.to_schema_contract()

        return input_contract, output_contract

    def get_edges(self, run_id: str) -> list[Edge]:
        """Get all edges for a run.

        Args:
            run_id: Run ID

        Returns:
            List of Edge models for this run, ordered by created_at then edge_id
            for deterministic export signatures.
        """
        query = select(edges_table).where(edges_table.c.run_id == run_id).order_by(edges_table.c.created_at, edges_table.c.edge_id)
        rows = self._ops.execute_fetchall(query)
        return [self._edge_loader.load(row) for row in rows]

    def get_edge(self, edge_id: str) -> Edge:
        """Get a single edge by ID.

        Tier 1: crash on missing — an edge_id from our own routing_events
        table MUST resolve. Missing means audit DB corruption.

        Args:
            edge_id: Edge ID to look up

        Returns:
            Edge model

        Raises:
            AuditIntegrityError: If edge not found (audit integrity violation)
        """
        query = select(edges_table).where(edges_table.c.edge_id == edge_id)
        row = self._ops.execute_fetchone(query)
        if row is None:
            raise AuditIntegrityError(
                f"Audit integrity violation: edge '{edge_id}' not found. "
                f"A routing_event references a non-existent edge. "
                f"This indicates database corruption."
            )
        return self._edge_loader.load(row)

    def get_edge_map(self, run_id: str) -> dict[tuple[str, str], str]:
        """Get edge mapping for a run (from_node_id, label) -> edge_id.

        Args:
            run_id: Run to query

        Returns:
            Dictionary mapping (from_node_id, label) to edge_id

        Raises:
            AuditIntegrityError: If run has no edges registered (data corruption).
                DAG compilation always registers edges, so an empty map
                indicates the run was never properly initialized.

        Note:
            This encapsulates Landscape schema access for Orchestrator resume.
            Edge IDs are required for FK integrity when recording routing events.
        """
        query = select(edges_table).where(edges_table.c.run_id == run_id)
        edges = self._ops.execute_fetchall(query)

        edge_map: dict[tuple[str, str], str] = {}
        for edge in edges:
            edge_map[(edge.from_node_id, edge.label)] = edge.edge_id

        if not edge_map:
            raise AuditIntegrityError(
                f"Run {run_id!r} has no edges registered — cannot build edge map. "
                f"DAG compilation always registers edges; an empty map indicates "
                f"the run was never properly initialized or database corruption."
            )

        return edge_map

    def update_node_output_contract(
        self,
        run_id: str,
        node_id: str,
        contract: SchemaContract,
    ) -> None:
        """Update a node's output_contract after first-row inference or schema evolution.

        Called in two scenarios:
        1. Source infers schema from first valid row during OBSERVED mode
        2. Transform adds fields during execution (schema evolution)

        Args:
            run_id: Run containing the node
            node_id: Node to update (source or transform node)
            contract: SchemaContract with inferred/evolved fields

        Note:
            This is the complement to ``update_run_source_contract()`` for
            node-level contracts (the per-source ``run_sources`` writer that
            superseded the deleted run-level singleton ``update_run_contract``).
            Used for dynamic schema discovery and transform schema evolution.
        """
        audit_record = ContractAuditRecord.from_contract(contract)
        output_contract_json = audit_record.to_json()

        self._ops.execute_update(
            nodes_table.update()
            .where((nodes_table.c.run_id == run_id) & (nodes_table.c.node_id == node_id))
            .values(output_contract_json=output_contract_json)
        )

    # ── Error recording: public methods ──────────────────────────────────

    def record_validation_error(
        self,
        run_id: str,
        node_id: str | None,
        row_data: Any,
        error: str,
        schema_mode: str,
        destination: str,
        *,
        row_id: str | None = None,
        contract_violation: ContractViolation | None = None,
    ) -> str:
        """Record a validation error in the audit trail.

        Called when a source row fails schema validation. The row is
        quarantined (not processed further) but we record what we saw
        for complete audit coverage.

        Args:
            run_id: Current run ID
            node_id: Node where validation failed
            row_data: The row that failed validation (may be non-dict or contain non-finite values)
            error: Error description
            schema_mode: Schema mode that caught the error ("fixed", "flexible", "observed")
            destination: Where row was routed ("discard" or sink name)
            contract_violation: Optional contract violation details for structured auditing

        Returns:
            error_id for tracking
        """
        error_id = f"verr_{generate_id()[:12]}"

        # Tier-3 (external data) trust boundary: row_data may be non-canonical.
        # The coerce-and-record helpers return the canonical representation or an
        # explicit non-canonical fallback recorded in the audit trail.
        row_hash = self._canonical_or_recorded_hash(row_data)
        row_data_json = self._canonical_or_recorded_json(row_data)

        # Extract contract violation details if provided
        violation_type: str | None = None
        normalized_field_name: str | None = None
        original_field_name: str | None = None
        expected_type: str | None = None
        actual_type: str | None = None

        if contract_violation is not None:
            violation_record = ValidationErrorWithContract.from_violation(contract_violation)
            violation_type = violation_record.violation_type
            normalized_field_name = violation_record.normalized_field_name
            original_field_name = violation_record.original_field_name
            expected_type = violation_record.expected_type
            actual_type = violation_record.actual_type

        self._ops.execute_insert(
            validation_errors_table.insert().values(
                error_id=error_id,
                run_id=run_id,
                node_id=node_id,
                row_id=row_id,
                row_hash=row_hash,
                row_data_json=row_data_json,
                error=error,
                schema_mode=schema_mode,
                destination=destination,
                created_at=now(),
                violation_type=violation_type,
                normalized_field_name=normalized_field_name,
                original_field_name=original_field_name,
                expected_type=expected_type,
                actual_type=actual_type,
            )
        )

        return error_id

    def link_validation_error_to_row(
        self,
        *,
        run_id: str,
        error_id: str,
        row_id: str,
    ) -> None:
        """Attach a persisted quarantine row to an existing validation error."""
        actual_run_id = self._resolve_run_id_for_row(row_id)
        if actual_run_id != run_id:
            raise AuditIntegrityError(
                f"Validation error linkage prevented cross-run contamination: row {row_id!r} belongs to "
                f"run {actual_run_id!r}, but caller supplied run_id={run_id!r}."
            )

        error_row = self._ops.execute_fetchone(
            select(
                validation_errors_table.c.run_id,
                validation_errors_table.c.row_id,
            ).where(validation_errors_table.c.error_id == error_id)
        )
        if error_row is None:
            raise AuditIntegrityError(f"Validation error {error_id!r} does not exist in validation_errors. This is Tier 1 data corruption.")
        if error_row.run_id != run_id:
            raise AuditIntegrityError(
                f"Validation error linkage prevented cross-run contamination: error {error_id!r} belongs to "
                f"run {error_row.run_id!r}, but caller supplied run_id={run_id!r}."
            )
        if error_row.row_id is not None:
            if error_row.row_id != row_id:
                raise AuditIntegrityError(
                    f"Validation error {error_id!r} is already linked to row {error_row.row_id!r}; refusing to relink it to {row_id!r}."
                )
            return

        self._ops.execute_update(
            validation_errors_table.update()
            .where(
                validation_errors_table.c.error_id == error_id,
                validation_errors_table.c.run_id == run_id,
            )
            .values(row_id=row_id),
            context="validation_errors.row_id linkage",
        )

    def record_transform_error(
        self,
        ref: TokenRef,
        transform_id: str,
        row_data: Mapping[str, object] | PipelineRow,
        error_details: TransformErrorReason,
        destination: str,
    ) -> str:
        """Record a transform processing error in the audit trail.

        Called when a transform returns TransformResult.error().
        This is for legitimate errors, NOT transform bugs.

        Validates that the token belongs to the specified run_id before recording.
        Cross-run contamination crashes immediately per Tier 1 trust model.

        Args:
            ref: TokenRef bundling token_id and run_id
            transform_id: Transform that returned the error
            row_data: The row that could not be processed
            error_details: Error details from TransformResult (TransformErrorReason TypedDict)
            destination: Where row was routed ("discard" or sink name)

        Returns:
            error_id for tracking

        Raises:
            AuditIntegrityError: If token does not belong to the specified run
        """
        # Validate token belongs to the specified run (Tier 1 invariant)
        self._validate_token_run_ownership(ref)

        # Validate reason is a known TransformErrorCategory (Tier 1 write guard).
        # TypedDict has zero runtime enforcement — the Literal annotation only
        # helps at compile time. Invalid reasons must crash before persisting.
        from typing import get_args

        from elspeth.contracts.errors import TransformErrorCategory

        reason = error_details["reason"]
        valid_reasons = get_args(TransformErrorCategory)
        if reason not in valid_reasons:
            raise AuditIntegrityError(
                f"Invalid TransformErrorCategory '{reason}' at Tier 1 write boundary. "
                f"This is a plugin bug — transforms must use a valid error category. "
                f"Valid categories: {sorted(valid_reasons)}"
            )

        error_id = f"terr_{generate_id()[:12]}"

        # error_details may contain NaN/Infinity or non-serializable values
        # (e.g. from exception context in row operations). Tier-3 boundary:
        # error_details originates from transform results which may carry
        # arbitrary row-derived data. The helper returns canonical JSON or an
        # explicit __non_canonical__ envelope recorded in the audit trail.
        error_details_json = self._canonical_or_recorded_error_details_json(error_details)

        # row_data may contain NaN/Infinity (valid floats that passed source
        # validation). The coerce-and-record helpers return the canonical
        # representation or an explicit non-canonical fallback — losing the
        # error record is worse than recording a repr-based hash.
        row_hash = self._canonical_or_recorded_hash(row_data)
        row_data_json = self._canonical_or_recorded_json(row_data)

        self._ops.execute_insert(
            transform_errors_table.insert().values(
                error_id=error_id,
                run_id=ref.run_id,
                token_id=ref.token_id,
                transform_id=transform_id,
                row_hash=row_hash,
                row_data_json=row_data_json,
                error_details_json=error_details_json,
                destination=destination,
                created_at=now(),
            )
        )

        return error_id

    def get_validation_errors_for_row(
        self,
        run_id: str,
        row_hash: str | None = None,
        *,
        row_id: str | None = None,
    ) -> list[ValidationErrorRecord]:
        """Get validation errors for a row by stable row linkage or legacy hash.

        Args:
            run_id: Run ID to query
            row_hash: Legacy hash of the row data (used for historical/fallback lookup)
            row_id: Persisted row identifier for quarantined rows when available

        Returns:
            List of ValidationErrorRecord models
        """
        if row_id is not None:
            row_query = select(validation_errors_table).where(
                validation_errors_table.c.run_id == run_id,
                validation_errors_table.c.row_id == row_id,
            )
            row_rows = self._ops.execute_fetchall(row_query)
            if row_rows or row_hash is None:
                return [self._validation_error_loader.load(r) for r in row_rows]

        if row_hash is None:
            raise ValueError("get_validation_errors_for_row requires row_id or row_hash")

        hash_query = select(validation_errors_table).where(
            validation_errors_table.c.run_id == run_id,
            validation_errors_table.c.row_hash == row_hash,
        )
        hash_rows = self._ops.execute_fetchall(hash_query)
        return [self._validation_error_loader.load(r) for r in hash_rows]

    def get_validation_errors_for_run(self, run_id: str) -> list[ValidationErrorRecord]:
        """Get all validation errors for a run.

        Args:
            run_id: Run ID to query

        Returns:
            List of ValidationErrorRecord models, ordered by created_at
        """
        query = (
            select(validation_errors_table).where(validation_errors_table.c.run_id == run_id).order_by(validation_errors_table.c.created_at)
        )
        rows = self._ops.execute_fetchall(query)
        return [self._validation_error_loader.load(r) for r in rows]

    def get_transform_errors_for_token(self, token_id: str) -> list[TransformErrorRecord]:
        """Get transform errors for a specific token.

        Args:
            token_id: Token ID to query

        Returns:
            List of TransformErrorRecord models
        """
        query = select(transform_errors_table).where(
            transform_errors_table.c.token_id == token_id,
        )
        rows = self._ops.execute_fetchall(query)
        return [self._transform_error_loader.load(r) for r in rows]

    def get_transform_errors_for_run(self, run_id: str) -> list[TransformErrorRecord]:
        """Get all transform errors for a run.

        Args:
            run_id: Run ID to query

        Returns:
            List of TransformErrorRecord models, ordered by created_at
        """
        query = (
            select(transform_errors_table).where(transform_errors_table.c.run_id == run_id).order_by(transform_errors_table.c.created_at)
        )
        rows = self._ops.execute_fetchall(query)
        return [self._transform_error_loader.load(r) for r in rows]
