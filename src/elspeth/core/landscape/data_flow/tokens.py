"""Row and token lifecycle persistence (split from ``DataFlowRepository``).

Owns the ``rows``, ``tokens`` and ``token_parents`` audit aggregates: source
row creation, token creation, and the atomic fork/coalesce/expand lineage
writes. Fork and expand record the parent's TRANSIENT outcome inside the same
transaction as the child inserts — that atomicity is why those
``token_outcomes`` writes live here rather than in the outcome component.

Atomic transactions in fork/coalesce/expand preserved via direct
LandscapeDB.connection() usage.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from sqlalchemy.engine import Connection

from elspeth.contracts import Row, Token
from elspeth.contracts.audit import TokenRef
from elspeth.contracts.coordination import DEFAULT_RUN_LIVENESS_WINDOW_SECONDS, CoordinationToken
from elspeth.contracts.enums import TerminalOutcome, TerminalPath
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.schema_contract import SchemaContract
from elspeth.core.canonical import canonical_json, stable_hash
from elspeth.core.checkpoint.serialization import checkpoint_dumps
from elspeth.core.ids import generate_id
from elspeth.core.landscape._database_ops import DatabaseOps
from elspeth.core.landscape._helpers import now
from elspeth.core.landscape.data_flow.ownership import RowTokenOwnership
from elspeth.core.landscape.data_flow.serialization import (
    canonical_or_recorded_hash,
    canonical_or_recorded_repr_payload,
)
from elspeth.core.landscape.ports import LandscapeConnectionProvider
from elspeth.core.landscape.run_coordination_repository import fenced_leader_transaction
from elspeth.core.landscape.schema import (
    rows_table,
    token_outcomes_table,
    token_parents_table,
    tokens_table,
)

if TYPE_CHECKING:
    from elspeth.contracts.payload_store import PayloadStore

__all__ = ["RowTokenRepository"]


class RowTokenRepository:
    """Source row and token lifecycle writes: create, fork, coalesce, expand.

    Atomic transactions in fork/coalesce/expand preserved via direct
    LandscapeDB.connection() usage.
    """

    def __init__(
        self,
        db: LandscapeConnectionProvider,
        ops: DatabaseOps,
        *,
        ownership: RowTokenOwnership,
        payload_store: PayloadStore | None = None,
    ) -> None:
        self._db = db
        self._ops = ops
        self._ownership = ownership
        self._payload_store = payload_store

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
            data_hash = canonical_or_recorded_hash(data)
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
                payload_bytes = canonical_or_recorded_repr_payload(data).encode("utf-8")
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
        coordination_token: CoordinationToken | None = None,
    ) -> tuple[Row, Token]:
        """Create a source row and its initial token in one audit transaction.

        ``coordination_token`` (ADR-030 §C.4 row 9): an ingest-adjacent
        durable ``rows`` write at sequence N — when supplied, the
        verify-and-extend epoch fence is the first statement of the
        transaction (the boundary-failure and quarantine ingest arms ride
        this; the happy path composes through the scheduler's fenced
        ``ingest_row_with_initial_claim`` instead). ``None`` preserves the
        unfenced legacy arm for direct repository-level callers.
        """
        if coordination_token is None:
            with self._db.write_connection() as conn:
                return self.insert_row_with_token_on(
                    conn,
                    run_id=run_id,
                    source_node_id=source_node_id,
                    row_index=row_index,
                    data=data,
                    source_row_index=source_row_index,
                    ingest_sequence=ingest_sequence,
                    row_id=row_id,
                    token_id=token_id,
                    quarantined=quarantined,
                )
        with fenced_leader_transaction(
            self._db.engine,
            token=coordination_token,
            now=now(),
            window_seconds=DEFAULT_RUN_LIVENESS_WINDOW_SECONDS,
            verb="create_row_with_token",
        ) as conn:
            return self.insert_row_with_token_on(
                conn,
                run_id=run_id,
                source_node_id=source_node_id,
                row_index=row_index,
                data=data,
                source_row_index=source_row_index,
                ingest_sequence=ingest_sequence,
                row_id=row_id,
                token_id=token_id,
                quarantined=quarantined,
            )

    def insert_row_with_token_on(
        self,
        conn: Connection,
        *,
        run_id: str,
        source_node_id: str,
        row_index: int,
        data: Mapping[str, object],
        source_row_index: int | None = None,
        ingest_sequence: int | None = None,
        row_id: str | None = None,
        token_id: str | None = None,
        quarantined: bool = False,
    ) -> tuple[Row, Token]:
        """Connection-accepting rows+tokens insert: composes into the caller's transaction.

        Extracted from :meth:`create_row_with_token` so the fenced leader
        ingest (``TokenSchedulerRepository.ingest_row_with_initial_claim``,
        ADR-030 §C.4 row 9) can compose the rows insert, the tokens insert
        and the initial enqueue on ONE connection.
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
        token = Token(
            token_id=token_id or generate_id(),
            row_id=row.row_id,
            run_id=run_id,
            created_at=row.created_at,
        )

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
        run_id = self._ownership.resolve_run_id_for_row(row_id)

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
        self._ownership.validate_token_run_ownership(parent_ref)
        self._ownership.validate_token_row_ownership(parent_ref.token_id, row_id)

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
            self._ownership.validate_token_row_ownership(ref.token_id, row_id)
            self._ownership.validate_token_run_ownership(ref)
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
            run_id = self._ownership.resolve_run_id_for_row(row_id)

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
        self._ownership.validate_token_run_ownership(parent_ref)
        self._ownership.validate_token_row_ownership(parent_ref.token_id, row_id)

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
