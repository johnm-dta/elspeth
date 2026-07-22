"""Row and token lifecycle persistence (split from ``DataFlowRepository``).

Owns the ``rows``, ``tokens`` and ``token_parents`` audit aggregates: source
row creation, token creation, and the atomic fork/coalesce/expand lineage
writes. Fork and expand record the parent's TRANSIENT outcome inside the same
transaction as the child inserts — that atomicity is why those
``token_outcomes`` writes live here rather than in the outcome component.

Atomic transactions in fork/coalesce/expand use the connection provider's
explicit write-transaction boundary.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from hashlib import sha256
from typing import TYPE_CHECKING

from sqlalchemy import and_, select
from sqlalchemy.engine import Connection, RowMapping

from elspeth.contracts import CoalesceParentCompletion, Row, Token
from elspeth.contracts.audit import TokenRef
from elspeth.contracts.coordination import DEFAULT_RUN_LIVENESS_WINDOW_SECONDS, CoordinationToken
from elspeth.contracts.enums import NodeStateStatus, TerminalOutcome, TerminalPath
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
    batch_members_table,
    batches_table,
    coalesce_effect_members_table,
    coalesce_effects_table,
    node_states_table,
    rows_table,
    token_outcomes_table,
    token_parents_table,
    tokens_table,
)

if TYPE_CHECKING:
    from elspeth.contracts.payload_store import PayloadStore
    from elspeth.core.landscape.data_flow.outcomes import TokenOutcomeRepository
    from elspeth.core.landscape.execution.node_states import NodeStateRepository

__all__ = ["RowTokenRepository"]


class RowTokenRepository:
    """Source row and token lifecycle writes: create, fork, coalesce, expand.

    Atomic transactions in fork/coalesce/expand use the connection provider's
    explicit write-transaction boundary.
    """

    def __init__(
        self,
        db: LandscapeConnectionProvider,
        ops: DatabaseOps,
        *,
        ownership: RowTokenOwnership,
        payload_store: PayloadStore | None = None,
        outcomes: TokenOutcomeRepository | None = None,
        node_states: NodeStateRepository | None = None,
    ) -> None:
        self._db = db
        self._ops = ops
        self._ownership = ownership
        self._payload_store = payload_store
        self._outcomes = outcomes
        self._node_states = node_states

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

        if self._outcomes is None:
            raise RuntimeError("fork_token requires the token-outcome repository capability")

        with self._db.write_connection() as conn:
            self._outcomes.lock_token_outcome_dependencies((parent_ref,), conn=conn)
            existing_terminal = (
                conn.execute(
                    select(
                        token_outcomes_table.c.outcome,
                        token_outcomes_table.c.path,
                        token_outcomes_table.c.fork_group_id,
                        token_outcomes_table.c.expected_branches_json,
                    )
                    .where(token_outcomes_table.c.token_id == parent_ref.token_id)
                    .where(token_outcomes_table.c.run_id == parent_ref.run_id)
                    .where(token_outcomes_table.c.completed == 1)
                )
                .mappings()
                .one_or_none()
            )
            if existing_terminal is not None:
                return self._reconcile_fork_replay(
                    conn,
                    parent_ref=parent_ref,
                    row_id=row_id,
                    branches=branches,
                    step_in_pipeline=step_in_pipeline,
                    outcome=existing_terminal,
                )

            fork_group_id = generate_id()
            children = []
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
                        run_id=parent_ref.run_id,
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

    def _reconcile_fork_replay(
        self,
        conn: Connection,
        *,
        parent_ref: TokenRef,
        row_id: str,
        branches: Sequence[str],
        step_in_pipeline: int | None,
        outcome: RowMapping,
    ) -> tuple[list[Token], str]:
        """Return a previously committed exact fork or refuse divergent replay."""
        fork_group_id = outcome["fork_group_id"]
        try:
            recorded_branches = json.loads(outcome["expected_branches_json"])
        except (TypeError, ValueError):
            recorded_branches = None
        children = self._load_children_for_parent(conn, parent_ref=parent_ref)
        exact = (
            outcome["outcome"] == TerminalOutcome.TRANSIENT.value
            and outcome["path"] == TerminalPath.FORK_PARENT.value
            and isinstance(fork_group_id, str)
            and bool(fork_group_id)
            and recorded_branches == list(branches)
            and len(children) == len(branches)
            and all(
                child.row_id == row_id
                and child.run_id == parent_ref.run_id
                and child.fork_group_id == fork_group_id
                and child.join_group_id is None
                and child.expand_group_id is None
                and child.branch_name == branch
                and child.step_in_pipeline == step_in_pipeline
                and ordinal == expected_ordinal
                for expected_ordinal, ((child, ordinal), branch) in enumerate(zip(children, branches, strict=True))
            )
        )
        if not exact:
            raise AuditIntegrityError(
                f"fork_token: divergent fork replay for parent token {parent_ref.token_id!r}; "
                "the requested branches or lineage metadata do not match the committed fork"
            )
        return [child for child, _ordinal in children], fork_group_id

    @staticmethod
    def _load_children_for_parent(conn: Connection, *, parent_ref: TokenRef) -> list[tuple[Token, int]]:
        rows = (
            conn.execute(
                select(
                    tokens_table.c.token_id,
                    tokens_table.c.row_id,
                    tokens_table.c.run_id,
                    tokens_table.c.fork_group_id,
                    tokens_table.c.join_group_id,
                    tokens_table.c.expand_group_id,
                    tokens_table.c.branch_name,
                    tokens_table.c.step_in_pipeline,
                    tokens_table.c.token_data_ref,
                    tokens_table.c.created_at,
                    token_parents_table.c.ordinal,
                )
                .select_from(
                    token_parents_table.join(
                        tokens_table,
                        and_(
                            tokens_table.c.token_id == token_parents_table.c.token_id,
                            tokens_table.c.run_id == token_parents_table.c.run_id,
                        ),
                    )
                )
                .where(token_parents_table.c.parent_token_id == parent_ref.token_id)
                .where(token_parents_table.c.run_id == parent_ref.run_id)
                .order_by(token_parents_table.c.ordinal)
            )
            .mappings()
            .all()
        )
        return [
            (
                Token(
                    token_id=str(row["token_id"]),
                    row_id=str(row["row_id"]),
                    run_id=str(row["run_id"]),
                    fork_group_id=row["fork_group_id"],
                    join_group_id=row["join_group_id"],
                    expand_group_id=row["expand_group_id"],
                    branch_name=row["branch_name"],
                    step_in_pipeline=row["step_in_pipeline"],
                    token_data_ref=row["token_data_ref"],
                    created_at=row["created_at"],
                ),
                int(row["ordinal"]),
            )
            for row in rows
        ]

    def coalesce_tokens(
        self,
        parent_refs: list[TokenRef],
        row_id: str,
        merged_payload: Mapping[str, object],
        *,
        coalesce_node_id: str | None = None,
        parent_state_ids: Sequence[str] | None = None,
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
        ordered_parent_ids = tuple(ref.token_id for ref in parent_refs)
        duplicate_parent_ids = sorted(token_id for token_id, count in Counter(ordered_parent_ids).items() if count > 1)
        if duplicate_parent_ids:
            raise AuditIntegrityError(
                "coalesce_tokens received duplicate parent tokens; normalized effect membership requires distinct parents: "
                f"{duplicate_parent_ids!r}"
            )
        normalized_state_ids = None if parent_state_ids is None else tuple(parent_state_ids)
        if normalized_state_ids is not None:
            if len(normalized_state_ids) != len(parent_refs):
                raise AuditIntegrityError("coalesce_tokens parent_state_ids must align one-for-one with parent_refs")
            if any(not state_id for state_id in normalized_state_ids) or len(set(normalized_state_ids)) != len(normalized_state_ids):
                raise AuditIntegrityError("coalesce_tokens parent_state_ids must be distinct non-empty identities")
        resolved_node_id = coalesce_node_id or f"legacy-step:{step_in_pipeline if step_in_pipeline is not None else 'unknown'}"
        if not resolved_node_id:
            raise AuditIntegrityError("coalesce_tokens requires a non-empty coalesce_node_id")

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
        envelope_bytes = checkpoint_dumps(envelope).encode("utf-8")
        expected_token_data_ref = sha256(envelope_bytes).hexdigest()
        token_data_ref = self._payload_store.store(envelope_bytes)
        if token_data_ref != expected_token_data_ref:
            raise AuditIntegrityError(
                "coalesce_tokens payload store violated its content-addressed contract: "
                f"expected {expected_token_data_ref}, got {token_data_ref}"
            )

        canonical_parent_ids = tuple(sorted(ordered_parent_ids))
        parent_set_hash = stable_hash(canonical_parent_ids)
        effect_hash = stable_hash(
            {
                "ordered_parent_ids": ordered_parent_ids,
                "step_in_pipeline": step_in_pipeline,
                "token_data_ref": token_data_ref,
            }
        )
        scope = (
            coalesce_effects_table.c.run_id == run_id,
            coalesce_effects_table.c.coalesce_node_id == resolved_node_id,
            coalesce_effects_table.c.row_id == row_id,
            coalesce_effects_table.c.parent_set_hash == parent_set_hash,
        )

        with self._db.write_connection() as conn:
            # Global PostgreSQL order: parent tokens, parent node states, then
            # the effect scope row. Every writer for the same parent set must
            # overlap on these token locks, so the loser re-reads the winner's
            # committed effect below. A natural-key IntegrityError here is not
            # an idempotency signal; it indicates a violated lock invariant or
            # an unrelated structural constraint and must surface unchanged.
            self._lock_coalesce_dependencies(
                conn,
                parent_refs=parent_refs,
                parent_state_ids=normalized_state_ids,
                coalesce_node_id=resolved_node_id,
            )
            existing = conn.execute(select(coalesce_effects_table).where(*scope).with_for_update()).mappings().one_or_none()
            if existing is not None:
                return self._validate_and_load_coalesce_effect(
                    conn,
                    existing,
                    ordered_parent_ids=ordered_parent_ids,
                    parent_state_ids=normalized_state_ids,
                    step_in_pipeline=step_in_pipeline,
                    effect_hash=effect_hash,
                    expected_token_data_ref=expected_token_data_ref,
                )

            join_group_id = generate_id()
            token_id = generate_id()
            effect_id = generate_id()
            timestamp = now()
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
                        run_id=run_id,
                        ordinal=ordinal,
                    )
                )
                if result.rowcount == 0:
                    raise AuditIntegrityError(
                        f"coalesce_tokens: token_parent INSERT affected zero rows (child={token_id}, parent={ref.token_id})"
                    )

            result = conn.execute(
                coalesce_effects_table.insert().values(
                    effect_id=effect_id,
                    run_id=run_id,
                    coalesce_node_id=resolved_node_id,
                    row_id=row_id,
                    parent_set_hash=parent_set_hash,
                    effect_hash=effect_hash,
                    expected_token_data_ref=expected_token_data_ref,
                    step_in_pipeline=step_in_pipeline,
                    status="materialized",
                    result_token_id=token_id,
                    result_join_group_id=join_group_id,
                    created_at=timestamp,
                    completed_at=None,
                )
            )
            if result.rowcount == 0:
                raise AuditIntegrityError(f"coalesce_tokens: effect INSERT affected zero rows (effect_id={effect_id})")
            for ordinal, ref in enumerate(parent_refs):
                result = conn.execute(
                    coalesce_effect_members_table.insert().values(
                        effect_id=effect_id,
                        run_id=run_id,
                        ordinal=ordinal,
                        parent_token_id=ref.token_id,
                        parent_state_id=None if normalized_state_ids is None else normalized_state_ids[ordinal],
                    )
                )
                if result.rowcount == 0:
                    raise AuditIntegrityError(
                        f"coalesce_tokens: normalized effect member INSERT affected zero rows (effect_id={effect_id}, ordinal={ordinal})"
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

    def finalize_coalesce_effect(
        self,
        *,
        merged: Token,
        parent_completions: Sequence[CoalesceParentCompletion],
    ) -> None:
        """Atomically terminalize every parent and CAS the materialized effect."""
        if self._outcomes is None or self._node_states is None:
            raise RuntimeError("coalesce finalization is unavailable on this repository capability")
        if merged.join_group_id is None:
            raise AuditIntegrityError("coalesce finalization requires a merged token with join_group_id")
        completions = tuple(parent_completions)
        if not completions:
            raise AuditIntegrityError("coalesce finalization requires at least one parent completion")
        refs = tuple(item.parent_ref for item in completions)
        state_ids = tuple(item.state_id for item in completions)
        if len({ref.token_id for ref in refs}) != len(refs) or len(set(state_ids)) != len(state_ids):
            raise AuditIntegrityError("coalesce finalization requires distinct parent token and state identities")

        try:
            with self._db.write_connection() as conn:
                self._lock_coalesce_dependencies(
                    conn,
                    parent_refs=refs,
                    parent_state_ids=state_ids,
                    coalesce_node_id=None,
                )
                effect = (
                    conn.execute(
                        select(coalesce_effects_table)
                        .where(coalesce_effects_table.c.result_token_id == merged.token_id)
                        .where(coalesce_effects_table.c.run_id == merged.run_id)
                        .where(coalesce_effects_table.c.result_join_group_id == merged.join_group_id)
                        .with_for_update()
                    )
                    .mappings()
                    .one_or_none()
                )
                if effect is None:
                    raise AuditIntegrityError("coalesce finalization could not find the materialized effect receipt")
                members = self._load_coalesce_members(conn, str(effect["effect_id"]))
                observed = tuple((str(member["parent_token_id"]), member["parent_state_id"]) for member in members)
                expected = tuple((item.parent_ref.token_id, item.state_id) for item in completions)
                if tuple(token_id for token_id, _state_id in observed) != tuple(token_id for token_id, _state_id in expected):
                    raise AuditIntegrityError("coalesce finalization parent/state membership has a divergent ordered parent sequence")
                for member, (_token_id, expected_state_id) in zip(members, expected, strict=True):
                    observed_state_id = member["parent_state_id"]
                    if observed_state_id is None:
                        result = conn.execute(
                            coalesce_effect_members_table.update()
                            .where(coalesce_effect_members_table.c.effect_id == effect["effect_id"])
                            .where(coalesce_effect_members_table.c.ordinal == member["ordinal"])
                            .where(coalesce_effect_members_table.c.parent_state_id.is_(None))
                            .values(parent_state_id=expected_state_id)
                        )
                        if result.rowcount != 1:
                            raise AuditIntegrityError("coalesce finalization parent/state membership CAS lost")
                    elif str(observed_state_id) != expected_state_id:
                        raise AuditIntegrityError("coalesce finalization parent/state membership is divergent")

                if effect["status"] == "completed":
                    self._verify_completed_coalesce_effect(
                        conn,
                        merged=merged,
                        parent_completions=completions,
                    )
                    return
                if effect["status"] != "materialized":
                    raise AuditIntegrityError(f"coalesce effect has unknown status {effect['status']!r}")

                for item in completions:
                    self._node_states.complete_node_state(
                        state_id=item.state_id,
                        status=NodeStateStatus.COMPLETED,
                        output_data={"merged_into": merged.token_id},
                        duration_ms=item.duration_ms,
                        context_after=item.context_after,
                        conn=conn,
                    )
                    self._outcomes.record_token_outcome(
                        ref=item.parent_ref,
                        outcome=TerminalOutcome.SUCCESS,
                        path=TerminalPath.COALESCED,
                        join_group_id=merged.join_group_id,
                        conn=conn,
                        dependencies_prelocked=True,
                    )

                completed_at = now()
                result = conn.execute(
                    coalesce_effects_table.update()
                    .where(coalesce_effects_table.c.effect_id == effect["effect_id"])
                    .where(coalesce_effects_table.c.status == "materialized")
                    .where(coalesce_effects_table.c.completed_at.is_(None))
                    .values(status="completed", completed_at=completed_at)
                )
                if result.rowcount != 1:
                    raise AuditIntegrityError("coalesce effect completion CAS lost")
        except AuditIntegrityError:
            raise
        except Exception as exc:
            raise AuditIntegrityError(f"coalesce effect finalization failure: {type(exc).__name__}: {exc}") from exc

    def _lock_coalesce_dependencies(
        self,
        conn: Connection,
        *,
        parent_refs: Sequence[TokenRef],
        parent_state_ids: Sequence[str] | None,
        coalesce_node_id: str | None,
    ) -> None:
        if self._outcomes is None:
            raise RuntimeError("coalesce writes require the token-outcome repository capability")
        self._outcomes.lock_token_outcome_dependencies(parent_refs, conn=conn)
        if parent_state_ids is None:
            return
        state_rows = (
            conn.execute(
                select(
                    node_states_table.c.state_id,
                    node_states_table.c.token_id,
                    node_states_table.c.run_id,
                    node_states_table.c.node_id,
                )
                .where(node_states_table.c.state_id.in_(sorted(parent_state_ids)))
                .order_by(node_states_table.c.state_id)
                .with_for_update(of=node_states_table)
            )
            .mappings()
            .all()
        )
        if len(state_rows) != len(parent_state_ids):
            raise AuditIntegrityError("coalesce parent/state membership references a missing node state")
        expected_by_state = dict(zip(parent_state_ids, parent_refs, strict=True))
        for row in state_rows:
            expected_ref = expected_by_state[str(row["state_id"])]
            if row["token_id"] != expected_ref.token_id or row["run_id"] != expected_ref.run_id:
                raise AuditIntegrityError("coalesce parent/state membership crosses token or run identity")
            if coalesce_node_id is not None and row["node_id"] != coalesce_node_id:
                raise AuditIntegrityError("coalesce parent/state membership belongs to a different coalesce node")

    @staticmethod
    def _load_coalesce_members(conn: Connection, effect_id: str) -> list[RowMapping]:
        return list(
            conn.execute(
                select(coalesce_effect_members_table)
                .where(coalesce_effect_members_table.c.effect_id == effect_id)
                .order_by(coalesce_effect_members_table.c.ordinal)
            )
            .mappings()
            .all()
        )

    def _validate_and_load_coalesce_effect(
        self,
        conn: Connection,
        effect: RowMapping,
        *,
        ordered_parent_ids: Sequence[str],
        parent_state_ids: Sequence[str] | None,
        step_in_pipeline: int | None,
        effect_hash: str,
        expected_token_data_ref: str,
    ) -> Token:
        members = self._load_coalesce_members(conn, str(effect["effect_id"]))
        observed_parent_ids = tuple(str(member["parent_token_id"]) for member in members)
        if observed_parent_ids != tuple(ordered_parent_ids):
            raise AuditIntegrityError(
                "coalesce_tokens divergent collision: ordered parent sequence differs; "
                f"recorded={observed_parent_ids!r}, requested={tuple(ordered_parent_ids)!r}"
            )
        if parent_state_ids is not None:
            observed_state_ids = tuple(member["parent_state_id"] for member in members)
            if any(value is not None for value in observed_state_ids) and observed_state_ids != tuple(parent_state_ids):
                raise AuditIntegrityError("coalesce_tokens divergent collision: parent/state membership differs")
        if effect["step_in_pipeline"] != step_in_pipeline:
            raise AuditIntegrityError("coalesce_tokens divergent collision: step_in_pipeline differs")
        if effect["effect_hash"] != effect_hash or effect["expected_token_data_ref"] != expected_token_data_ref:
            raise AuditIntegrityError("coalesce_tokens divergent collision: merged payload/contract effect differs")

        token_row = (
            conn.execute(
                select(tokens_table)
                .where(tokens_table.c.token_id == effect["result_token_id"])
                .where(tokens_table.c.run_id == effect["run_id"])
                .where(tokens_table.c.join_group_id == effect["result_join_group_id"])
            )
            .mappings()
            .one_or_none()
        )
        if token_row is None:
            raise AuditIntegrityError("coalesce effect result tuple no longer resolves to its merged token")
        parent_links = tuple(
            conn.execute(
                select(token_parents_table.c.parent_token_id)
                .where(token_parents_table.c.token_id == token_row["token_id"])
                .where(token_parents_table.c.run_id == token_row["run_id"])
                .order_by(token_parents_table.c.ordinal)
            )
            .scalars()
            .all()
        )
        if parent_links != tuple(ordered_parent_ids):
            raise AuditIntegrityError("coalesce effect result token has divergent ordered parent links")
        return Token(
            token_id=str(token_row["token_id"]),
            row_id=str(token_row["row_id"]),
            run_id=str(token_row["run_id"]),
            fork_group_id=token_row["fork_group_id"],
            join_group_id=token_row["join_group_id"],
            expand_group_id=token_row["expand_group_id"],
            branch_name=token_row["branch_name"],
            step_in_pipeline=token_row["step_in_pipeline"],
            token_data_ref=token_row["token_data_ref"],
            created_at=token_row["created_at"],
        )

    @staticmethod
    def _verify_completed_coalesce_effect(
        conn: Connection,
        *,
        merged: Token,
        parent_completions: Sequence[CoalesceParentCompletion],
    ) -> None:
        state_ids = [item.state_id for item in parent_completions]
        states = conn.execute(
            select(node_states_table.c.state_id, node_states_table.c.status).where(node_states_table.c.state_id.in_(state_ids))
        ).all()
        if len(states) != len(state_ids) or any(row.status != NodeStateStatus.COMPLETED.value for row in states):
            raise AuditIntegrityError("completed coalesce effect has incomplete parent node-state evidence")
        token_ids = [item.parent_ref.token_id for item in parent_completions]
        outcomes = conn.execute(
            select(
                token_outcomes_table.c.token_id,
                token_outcomes_table.c.outcome,
                token_outcomes_table.c.path,
                token_outcomes_table.c.join_group_id,
            )
            .where(token_outcomes_table.c.token_id.in_(token_ids))
            .where(token_outcomes_table.c.completed == 1)
        ).all()
        expected = {(token_id, TerminalOutcome.SUCCESS.value, TerminalPath.COALESCED.value, merged.join_group_id) for token_id in token_ids}
        observed = {(row.token_id, row.outcome, row.path, row.join_group_id) for row in outcomes}
        if observed != expected:
            raise AuditIntegrityError("completed coalesce effect has divergent parent outcome evidence")

    def expand_token(
        self,
        parent_ref: TokenRef,
        row_id: str,
        child_payloads: Sequence[Mapping[str, object]],
        *,
        output_contract: SchemaContract,
        step_in_pipeline: int | None = None,
        parent_path: TerminalPath = TerminalPath.EXPAND_PARENT,
        parent_batch_id: str | None = None,
    ) -> tuple[list[Token], str]:
        """Expand a token into multiple child tokens (deaggregation).

        ATOMIC: Creates children and records the parent's explicit terminal
        disposition in one transaction.

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
            parent_path: Parent terminal path. Normal deaggregation uses
                EXPAND_PARENT; batch aggregation uses BATCH_CONSUMED.
            parent_batch_id: Required when parent_path is BATCH_CONSUMED and
                forbidden for EXPAND_PARENT.

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

        if parent_path == TerminalPath.EXPAND_PARENT:
            if parent_batch_id is not None:
                raise ValueError("expand_token EXPAND_PARENT forbids parent_batch_id")
        elif parent_path == TerminalPath.BATCH_CONSUMED:
            if parent_batch_id is None:
                raise ValueError("expand_token BATCH_CONSUMED requires parent_batch_id")
        else:
            raise ValueError(f"expand_token parent_path must be EXPAND_PARENT or BATCH_CONSUMED, got {parent_path.value!r}")

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

        # Prepare each child's self-contained {data, contract} envelope and its
        # protocol-defined content address before taking the expansion claim.
        # Exact or divergent replay performs no payload-store writes. A new claim
        # stores the envelopes after the terminal-outcome lock/check and verifies
        # that the store returned the required SHA-256 identities before any DB
        # child insert.
        # checkpoint_dumps is type-faithful (datetime preserved as datetime, not
        # stringified) — canonical_json would destroy Tier-1 fidelity.
        # Crash on store failure: a child token with no persisted payload is
        # unreconstructable on resume (epoch 11 invariant).
        contract_fmt = output_contract.to_checkpoint_format()
        child_payload_bytes = [
            checkpoint_dumps({"data": dict(payload), "contract": contract_fmt}).encode("utf-8") for payload in child_payloads
        ]
        child_data_refs = [sha256(payload_bytes).hexdigest() for payload_bytes in child_payload_bytes]

        if self._outcomes is None:
            raise RuntimeError("expand_token requires the token-outcome repository capability")

        with self._db.write_connection() as conn:
            # Serialize expansion contenders on PostgreSQL before child inserts;
            # SQLite's write_connection() already holds BEGIN IMMEDIATE.  The
            # completed-outcome check is the durable claim shared by normal and
            # batch expansion paths.
            self._outcomes.lock_token_outcome_dependencies((parent_ref,), conn=conn)
            existing_terminal = (
                conn.execute(
                    select(
                        token_outcomes_table.c.outcome,
                        token_outcomes_table.c.path,
                        token_outcomes_table.c.batch_id,
                        token_outcomes_table.c.expand_group_id,
                        token_outcomes_table.c.expected_branches_json,
                    )
                    .where(token_outcomes_table.c.token_id == parent_ref.token_id)
                    .where(token_outcomes_table.c.run_id == parent_ref.run_id)
                    .where(token_outcomes_table.c.completed == 1)
                )
                .mappings()
                .one_or_none()
            )
            if existing_terminal is not None:
                if parent_path == TerminalPath.EXPAND_PARENT and existing_terminal["path"] == TerminalPath.EXPAND_PARENT.value:
                    return self._reconcile_expansion_replay(
                        conn,
                        parent_ref=parent_ref,
                        row_id=row_id,
                        child_data_refs=child_data_refs,
                        step_in_pipeline=step_in_pipeline,
                        outcome=existing_terminal,
                        expand_group_id=existing_terminal["expand_group_id"],
                        expected_path=TerminalPath.EXPAND_PARENT,
                        require_recorded_count=True,
                    )
                if parent_path == TerminalPath.BATCH_CONSUMED and existing_terminal["path"] == TerminalPath.BATCH_CONSUMED.value:
                    batch_claim = conn.execute(
                        select(batches_table.c.run_id, batches_table.c.expansion_group_id).where(
                            batches_table.c.batch_id == parent_batch_id
                        )
                    ).one_or_none()
                    if existing_terminal["batch_id"] != parent_batch_id or batch_claim is None or batch_claim.run_id != parent_ref.run_id:
                        raise AuditIntegrityError(
                            f"expand_token: divergent expansion replay for parent token {parent_ref.token_id!r}; "
                            "the committed batch claim does not match the requested batch"
                        )
                    return self._reconcile_expansion_replay(
                        conn,
                        parent_ref=parent_ref,
                        row_id=row_id,
                        child_data_refs=child_data_refs,
                        step_in_pipeline=step_in_pipeline,
                        outcome=existing_terminal,
                        expand_group_id=batch_claim.expansion_group_id,
                        expected_path=TerminalPath.BATCH_CONSUMED,
                        require_recorded_count=False,
                    )
                raise AuditIntegrityError(
                    f"expand_token: parent token {parent_ref.token_id!r} already has a terminal outcome ({existing_terminal.path!r})"
                )

            stored_refs = [self._payload_store.store(payload_bytes) for payload_bytes in child_payload_bytes]
            if stored_refs != child_data_refs:
                raise AuditIntegrityError(
                    "expand_token: payload store returned an identity other than the required SHA-256 content address"
                )

            if parent_path == TerminalPath.BATCH_CONSUMED:
                assert parent_batch_id is not None  # validated above
                membership = conn.execute(
                    select(batch_members_table.c.token_id)
                    .where(batch_members_table.c.batch_id == parent_batch_id)
                    .where(batch_members_table.c.run_id == parent_ref.run_id)
                    .where(batch_members_table.c.token_id == parent_ref.token_id)
                ).one_or_none()
                if membership is None:
                    raise AuditIntegrityError(
                        f"expand_token: parent token {parent_ref.token_id!r} is not a member of batch {parent_batch_id!r} "
                        f"in run {parent_ref.run_id!r}"
                    )

                claim = conn.execute(
                    batches_table.update()
                    .where(batches_table.c.batch_id == parent_batch_id)
                    .where(batches_table.c.run_id == parent_ref.run_id)
                    .where(batches_table.c.expansion_group_id.is_(None))
                    .values(expansion_group_id=expand_group_id)
                )
                if claim.rowcount != 1:
                    existing_claim = conn.execute(
                        select(batches_table.c.run_id, batches_table.c.expansion_group_id).where(
                            batches_table.c.batch_id == parent_batch_id
                        )
                    ).one_or_none()
                    if existing_claim is None:
                        raise AuditIntegrityError(f"expand_token: batch {parent_batch_id!r} does not exist")
                    if existing_claim.run_id != parent_ref.run_id:
                        raise AuditIntegrityError(
                            f"expand_token: batch {parent_batch_id!r} belongs to run {existing_claim.run_id!r}, not {parent_ref.run_id!r}"
                        )
                    raise AuditIntegrityError(
                        f"expand_token: batch {parent_batch_id!r} already claimed an expansion ({existing_claim.expansion_group_id!r})"
                    )

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
                        run_id=parent_ref.run_id,
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

            # Record the explicit parent disposition in the SAME transaction.
            # No successful expansion path may leave a reprocessable parent.
            if parent_path == TerminalPath.EXPAND_PARENT:
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
            else:
                self._outcomes.record_token_outcome(
                    ref=parent_ref,
                    outcome=TerminalOutcome.TRANSIENT,
                    path=TerminalPath.BATCH_CONSUMED,
                    batch_id=parent_batch_id,
                    conn=conn,
                    dependencies_prelocked=True,
                )

        return children, expand_group_id

    def _reconcile_expansion_replay(
        self,
        conn: Connection,
        *,
        parent_ref: TokenRef,
        row_id: str,
        child_data_refs: Sequence[str],
        step_in_pipeline: int | None,
        outcome: RowMapping,
        expand_group_id: object,
        expected_path: TerminalPath,
        require_recorded_count: bool,
    ) -> tuple[list[Token], str]:
        """Return a previously committed exact expansion or refuse divergence."""
        try:
            recorded_contract = json.loads(outcome["expected_branches_json"])
        except (TypeError, ValueError):
            recorded_contract = None
        children = self._load_children_for_parent(conn, parent_ref=parent_ref)
        exact = (
            outcome["outcome"] == TerminalOutcome.TRANSIENT.value
            and outcome["path"] == expected_path.value
            and isinstance(expand_group_id, str)
            and bool(expand_group_id)
            and (not require_recorded_count or recorded_contract == {"count": len(child_data_refs)})
            and len(children) == len(child_data_refs)
            and all(
                child.row_id == row_id
                and child.run_id == parent_ref.run_id
                and child.fork_group_id is None
                and child.join_group_id is None
                and child.expand_group_id == expand_group_id
                and child.branch_name is None
                and child.step_in_pipeline == step_in_pipeline
                and child.token_data_ref == expected_payload_ref
                and ordinal == expected_ordinal
                for expected_ordinal, ((child, ordinal), expected_payload_ref) in enumerate(zip(children, child_data_refs, strict=True))
            )
        )
        if not exact:
            raise AuditIntegrityError(
                f"expand_token: divergent expansion replay for parent token {parent_ref.token_id!r}; "
                "the requested payloads or lineage metadata do not match the committed expansion"
            )
        assert isinstance(expand_group_id, str)  # narrowed by the exact-replay predicate above
        return [child for child, _ordinal in children], expand_group_id
