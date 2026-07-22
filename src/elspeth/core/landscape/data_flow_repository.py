"""Data-flow recording repository — compatibility facade.

The behaviour lives in the cohesive components under
:mod:`elspeth.core.landscape.data_flow` (filigree elspeth-b194136580):
row/token lifecycle with the atomic fork/coalesce/expand lineage writes
(:class:`RowTokenRepository`), token (outcome, path) terminals with the
ADR-019 policy (:class:`TokenOutcomeRepository`), execution-graph
nodes/edges (:class:`GraphAuditRepository`), validation/transform errors
(:class:`ErrorAuditRepository`), and the shared Tier-1 row/token ownership
guards (:class:`RowTokenOwnership`). :class:`DataFlowRepository` composes
them behind the historical surface so call sites can migrate
incrementally — new code should prefer the component attributes
(``.tokens``, ``.outcomes``, ``.graph``, ``.errors``, ``.ownership``) over
the flat delegators.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any

from sqlalchemy.engine import Connection
from sqlalchemy.engine import Row as SQLAlchemyRow

from elspeth.contracts import (
    Determinism,
    Edge,
    Node,
    NodeType,
    RoutingMode,
    Row,
    Token,
    TokenOutcome,
    TransformErrorReason,
    TransformErrorRecord,
    ValidationErrorRecord,
)
from elspeth.contracts.audit import TokenRef
from elspeth.contracts.coordination import CoordinationToken
from elspeth.contracts.engine import CoalesceParentCompletion
from elspeth.contracts.enums import TerminalOutcome, TerminalPath
from elspeth.contracts.schema_contract import SchemaContract
from elspeth.core.landscape._database_ops import DatabaseOps
from elspeth.core.landscape.data_flow import (
    ErrorAuditRepository,
    GraphAuditRepository,
    RowTokenOwnership,
    RowTokenRepository,
    TokenOutcomeRepository,
)
from elspeth.core.landscape.data_flow.serialization import (
    canonical_or_recorded_error_details_json,
    canonical_or_recorded_hash,
    canonical_or_recorded_json,
    canonical_or_recorded_repr_payload,
)
from elspeth.core.landscape.model_loaders import (
    EdgeLoader,
    NodeLoader,
    TokenOutcomeLoader,
    TransformErrorLoader,
    ValidationErrorLoader,
)
from elspeth.core.landscape.ports import LandscapeConnectionProvider

if TYPE_CHECKING:
    from elspeth.contracts.errors import ContractViolation
    from elspeth.contracts.payload_store import PayloadStore
    from elspeth.contracts.schema import SchemaConfig
    from elspeth.contracts.schema_contract import PipelineRow
    from elspeth.core.landscape.execution.node_states import NodeStateRepository

__all__ = ["DataFlowRepository"]


class DataFlowRepository:
    """Records data flow: tokens, rows, graph structure, and errors.

    Compatibility facade over the data-flow components: every historical
    verb delegates to exactly one component. The components share the same
    connection provider and :class:`DatabaseOps` instances, so test seams
    that patch ``repo._db`` / ``repo._ops`` attributes remain effective.

    Atomic transactions in fork/coalesce/expand use the connection provider's
    explicit write-transaction boundary.

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
        node_state_repository: NodeStateRepository | None = None,
    ) -> None:
        self._db = db
        self._ops = ops
        self._payload_store = payload_store
        self.ownership = RowTokenOwnership(ops)
        self.outcomes = TokenOutcomeRepository(db, ops, token_outcome_loader=token_outcome_loader, ownership=self.ownership)
        self.tokens = RowTokenRepository(
            db,
            ops,
            ownership=self.ownership,
            payload_store=payload_store,
            outcomes=self.outcomes,
            node_states=node_state_repository,
        )
        self.graph = GraphAuditRepository(ops, node_loader=node_loader, edge_loader=edge_loader)
        self.errors = ErrorAuditRepository(
            ops,
            validation_error_loader=validation_error_loader,
            transform_error_loader=transform_error_loader,
            ownership=self.ownership,
        )

    def write_connection(self) -> AbstractContextManager[Connection]:
        """Open a caller-owned audit write transaction for composed repository verbs."""
        return self._db.write_connection()

    # ── Tier-3 audit serialization (module functions in data_flow.serialization) ──

    def _canonical_or_recorded_hash(self, data: Any) -> str:
        """Hash external row data, recording a non-canonical fallback on failure."""
        return canonical_or_recorded_hash(data)

    def _canonical_or_recorded_json(self, data: Any) -> str:
        """Serialize external row data, recording a non-canonical fallback on failure."""
        return canonical_or_recorded_json(data)

    def _canonical_or_recorded_error_details_json(self, error_details: Any) -> str:
        """Serialize transform error_details, recording a non-canonical fallback."""
        return canonical_or_recorded_error_details_json(error_details)

    def _canonical_or_recorded_repr_payload(self, data: Any) -> str:
        """Serialize a quarantined payload, recording a repr sentinel on failure."""
        return canonical_or_recorded_repr_payload(data)

    # ── Row/token ownership (RowTokenOwnership) ────────────────────────────

    def _resolve_run_id_for_row(self, row_id: str) -> str:
        """Resolve the run_id that owns a given row_id."""
        return self.ownership.resolve_run_id_for_row(row_id)

    def resolve_row_ingest_sequence(self, row_id: str) -> int:
        """Resolve a row's global ingest ordering for scheduler fairness."""
        return self.ownership.resolve_row_ingest_sequence(row_id)

    def _resolve_token_ownership(self, token_id: str) -> tuple[str, str]:
        """Resolve the (row_id, run_id) that owns a given token_id."""
        return self.ownership.resolve_token_ownership(token_id)

    def _validate_token_run_ownership(self, ref: TokenRef) -> None:
        """Validate that a token belongs to the specified run."""
        self.ownership.validate_token_run_ownership(ref)

    def _validate_token_row_ownership(self, token_id: str, row_id: str) -> None:
        """Validate that a token belongs to the specified row."""
        self.ownership.validate_token_row_ownership(token_id, row_id)

    # ── Token recording (RowTokenRepository) ───────────────────────────────

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
        return self.tokens._prepare_source_row_record(
            run_id=run_id,
            source_node_id=source_node_id,
            row_index=row_index,
            data=data,
            source_row_index=source_row_index,
            ingest_sequence=ingest_sequence,
            row_id=row_id,
            quarantined=quarantined,
        )

    @staticmethod
    def _row_insert_values(row: Row) -> dict[str, object]:
        return RowTokenRepository._row_insert_values(row)

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
        """Create a source row record."""
        return self.tokens.create_row(
            run_id,
            source_node_id,
            row_index,
            data,
            source_row_index=source_row_index,
            ingest_sequence=ingest_sequence,
            row_id=row_id,
            quarantined=quarantined,
        )

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
        """Create a source row and its initial token in one audit transaction."""
        return self.tokens.create_row_with_token(
            run_id,
            source_node_id,
            row_index,
            data,
            source_row_index=source_row_index,
            ingest_sequence=ingest_sequence,
            row_id=row_id,
            token_id=token_id,
            quarantined=quarantined,
            coordination_token=coordination_token,
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
        """Connection-accepting rows+tokens insert: composes into the caller's transaction."""
        return self.tokens.insert_row_with_token_on(
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

    def create_token(
        self,
        row_id: str,
        *,
        token_id: str | None = None,
        branch_name: str | None = None,
        fork_group_id: str | None = None,
        join_group_id: str | None = None,
    ) -> Token:
        """Create a token (row instance in DAG path)."""
        return self.tokens.create_token(
            row_id,
            token_id=token_id,
            branch_name=branch_name,
            fork_group_id=fork_group_id,
            join_group_id=join_group_id,
        )

    def fork_token(
        self,
        parent_ref: TokenRef,
        row_id: str,
        branches: list[str],
        *,
        step_in_pipeline: int | None = None,
    ) -> tuple[list[Token], str]:
        """Fork a token to multiple branches."""
        return self.tokens.fork_token(parent_ref, row_id, branches, step_in_pipeline=step_in_pipeline)

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
        """Coalesce multiple tokens into one (join operation)."""
        return self.tokens.coalesce_tokens(
            parent_refs,
            row_id,
            merged_payload,
            coalesce_node_id=coalesce_node_id,
            parent_state_ids=parent_state_ids,
            merged_contract=merged_contract,
            step_in_pipeline=step_in_pipeline,
        )

    def finalize_coalesce_effect(
        self,
        *,
        merged: Token,
        parent_completions: Sequence[CoalesceParentCompletion],
    ) -> None:
        """Atomically terminalize the parents of one materialized coalesce."""
        self.tokens.finalize_coalesce_effect(merged=merged, parent_completions=parent_completions)

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
        """Expand a token into multiple child tokens (deaggregation)."""
        return self.tokens.expand_token(
            parent_ref,
            row_id,
            child_payloads,
            output_contract=output_contract,
            step_in_pipeline=step_in_pipeline,
            parent_path=parent_path,
            parent_batch_id=parent_batch_id,
        )

    # ── Token outcome recording (TokenOutcomeRepository) ───────────────────

    def lock_token_outcome_dependencies(self, refs: Sequence[TokenRef], *, conn: Connection) -> None:
        """Prelock token dependencies before a composed outcome transaction mutates states."""
        self.outcomes.lock_token_outcome_dependencies(refs, conn=conn)

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
        """Validate discriminator fields for the (outcome, path) pair."""
        self.outcomes._validate_outcome_fields(
            outcome,
            path,
            sink_name=sink_name,
            batch_id=batch_id,
            fork_group_id=fork_group_id,
            join_group_id=join_group_id,
            expand_group_id=expand_group_id,
            error_hash=error_hash,
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
        """Validate ADR-019 real-time cross-table invariants."""
        self.outcomes._validate_cross_table_invariants(
            ref,
            outcome,
            path,
            sink_name=sink_name,
            sink_node_id=sink_node_id,
            artifact_id=artifact_id,
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
        conn: Connection | None = None,
    ) -> str:
        """Record a token's (outcome, path) audit terminal in the audit trail."""
        return self.outcomes.record_token_outcome(
            ref,
            outcome,
            path,
            sink_name=sink_name,
            sink_node_id=sink_node_id,
            artifact_id=artifact_id,
            batch_id=batch_id,
            fork_group_id=fork_group_id,
            join_group_id=join_group_id,
            expand_group_id=expand_group_id,
            error_hash=error_hash,
            context=context,
            conn=conn,
        )

    def find_orphaned_transient_parents(self, run_id: str) -> list[SQLAlchemyRow[Any]]:
        """Find I1a parent tokens with no child token outcome witnesses."""
        return self.outcomes.find_orphaned_transient_parents(run_id)

    def find_orphaned_batch_consumptions(self, run_id: str) -> list[str]:
        """Find I1b batch IDs consumed by tokens whose batch did not complete."""
        return self.outcomes.find_orphaned_batch_consumptions(run_id)

    def sweep_deferred_invariants_or_crash(self, run_id: str) -> None:
        """Sweep ADR-019 deferred I1a/I1b invariants at a stable run boundary."""
        self.outcomes.sweep_deferred_invariants_or_crash(run_id)

    def get_token_outcome(self, token_id: str) -> TokenOutcome | None:
        """Get the terminal outcome for a token."""
        return self.outcomes.get_token_outcome(token_id)

    def get_token_outcomes_for_row(self, run_id: str, row_id: str) -> list[TokenOutcome]:
        """Get all token outcomes for a row in a single query."""
        return self.outcomes.get_token_outcomes_for_row(run_id, row_id)

    # ── Graph recording (GraphAuditRepository) ─────────────────────────────

    def _sanitize_node_config_for_audit(self, config: Mapping[str, object], *, plugin_name: str | None) -> Mapping[str, object]:
        """Return an audit-safe node config with secrets fingerprinted."""
        from elspeth.core.config import sanitize_node_config_for_audit

        return sanitize_node_config_for_audit(config, plugin_name=plugin_name)

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
        """Register a node in the execution graph."""
        return self.graph.register_node(
            run_id,
            plugin_name,
            node_type,
            plugin_version,
            config,
            node_id=node_id,
            sequence=sequence,
            schema_hash=schema_hash,
            determinism=determinism,
            schema_config=schema_config,
            source_file_hash=source_file_hash,
            input_contract=input_contract,
            output_contract=output_contract,
        )

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
        """Register an edge in the execution graph."""
        return self.graph.register_edge(run_id, from_node_id, to_node_id, label, mode, edge_id=edge_id)

    def get_node(self, node_id: str, run_id: str) -> Node | None:
        """Get a node by its composite primary key (node_id, run_id)."""
        return self.graph.get_node(node_id, run_id)

    def get_nodes(self, run_id: str) -> list[Node]:
        """Get all nodes for a run."""
        return self.graph.get_nodes(run_id)

    def get_node_contracts(
        self, run_id: str, node_id: str, *, allow_missing: bool = False
    ) -> tuple[SchemaContract | None, SchemaContract | None]:
        """Get input and output contracts for a node."""
        return self.graph.get_node_contracts(run_id, node_id, allow_missing=allow_missing)

    def get_edges(self, run_id: str) -> list[Edge]:
        """Get all edges for a run."""
        return self.graph.get_edges(run_id)

    def get_edge(self, edge_id: str) -> Edge:
        """Get a single edge by ID."""
        return self.graph.get_edge(edge_id)

    def get_edge_map(self, run_id: str) -> dict[tuple[str, str], str]:
        """Get edge mapping for a run (from_node_id, label) -> edge_id."""
        return self.graph.get_edge_map(run_id)

    def update_node_output_contract(
        self,
        run_id: str,
        node_id: str,
        contract: SchemaContract,
    ) -> None:
        """Update a node's output_contract after first-row inference or schema evolution."""
        self.graph.update_node_output_contract(run_id, node_id, contract)

    # ── Error recording (ErrorAuditRepository) ─────────────────────────────

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
        """Record a validation error in the audit trail."""
        return self.errors.record_validation_error(
            run_id,
            node_id,
            row_data,
            error,
            schema_mode,
            destination,
            row_id=row_id,
            contract_violation=contract_violation,
        )

    def link_validation_error_to_row(
        self,
        *,
        run_id: str,
        error_id: str,
        row_id: str,
    ) -> None:
        """Attach a persisted quarantine row to an existing validation error."""
        self.errors.link_validation_error_to_row(run_id=run_id, error_id=error_id, row_id=row_id)

    def record_transform_error(
        self,
        ref: TokenRef,
        transform_id: str,
        row_data: Mapping[str, object] | PipelineRow,
        error_details: TransformErrorReason,
        destination: str,
    ) -> str:
        """Record a transform processing error in the audit trail."""
        return self.errors.record_transform_error(ref, transform_id, row_data, error_details, destination)

    def get_validation_errors_for_row(
        self,
        run_id: str,
        row_hash: str | None = None,
        *,
        row_id: str | None = None,
    ) -> list[ValidationErrorRecord]:
        """Get validation errors for a row by stable row linkage or legacy hash."""
        return self.errors.get_validation_errors_for_row(run_id, row_hash, row_id=row_id)

    def get_validation_errors_for_run(self, run_id: str) -> list[ValidationErrorRecord]:
        """Get all validation errors for a run."""
        return self.errors.get_validation_errors_for_run(run_id)

    def get_transform_errors_for_token(self, token_id: str) -> list[TransformErrorRecord]:
        """Get transform errors for a specific token."""
        return self.errors.get_transform_errors_for_token(token_id)

    def get_transform_errors_for_run(self, run_id: str) -> list[TransformErrorRecord]:
        """Get all transform errors for a run."""
        return self.errors.get_transform_errors_for_run(run_id)
