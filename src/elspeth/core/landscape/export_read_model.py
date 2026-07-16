"""One-connection repeatable read model for immutable audit exports."""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC
from typing import Any

from sqlalchemy import and_, select
from sqlalchemy.engine import Connection, Engine

from elspeth.contracts import RunStatus, SecretResolution
from elspeth.contracts.audit_export import AuditExportTerminalWitness
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.plugin_policy_audit import WebPluginPolicyEvidence
from elspeth.core.landscape.model_loaders import (
    ArtifactLoader,
    BatchLoader,
    BatchMemberLoader,
    CallLoader,
    EdgeLoader,
    NodeLoader,
    NodeStateLoader,
    OperationLoader,
    RoutingEventLoader,
    RowLoader,
    RunLoader,
    SchedulerEventLoader,
    TokenLoader,
    TokenOutcomeLoader,
    TokenParentLoader,
    TransformErrorLoader,
    ValidationErrorLoader,
)
from elspeth.core.landscape.schema import (
    artifacts_table,
    batch_members_table,
    batches_table,
    calls_table,
    edges_table,
    node_states_table,
    nodes_table,
    operations_table,
    routing_events_table,
    rows_table,
    run_attributions_table,
    run_web_plugin_policy_table,
    runs_table,
    scheduler_events_table,
    secret_resolutions_table,
    token_outcomes_table,
    token_parents_table,
    tokens_table,
    transform_errors_table,
    validation_errors_table,
)

_QUERY_CHUNK_SIZE = 500
_EXPORT_TERMINAL = frozenset({RunStatus.COMPLETED, RunStatus.COMPLETED_WITH_FAILURES, RunStatus.EMPTY})


def _chunks(values: Sequence[str]) -> Iterator[Sequence[str]]:
    for offset in range(0, len(values), _QUERY_CHUNK_SIZE):
        yield values[offset : offset + _QUERY_CHUNK_SIZE]


class ConnectionBoundExportReadModel:
    """Exporter query adapter that can execute only on one supplied connection."""

    def __init__(self, connection: Connection) -> None:
        if not isinstance(connection, Connection):
            raise TypeError("connection must be a SQLAlchemy Connection")
        self._connection = connection
        self._run_loader = RunLoader()
        self._node_loader = NodeLoader()
        self._edge_loader = EdgeLoader()
        self._operation_loader = OperationLoader()
        self._call_loader = CallLoader()
        self._validation_error_loader = ValidationErrorLoader()
        self._transform_error_loader = TransformErrorLoader()
        self._row_loader = RowLoader()
        self._token_loader = TokenLoader()
        self._token_parent_loader = TokenParentLoader()
        self._token_outcome_loader = TokenOutcomeLoader()
        self._scheduler_event_loader = SchedulerEventLoader()
        self._node_state_loader = NodeStateLoader()
        self._routing_event_loader = RoutingEventLoader()
        self._batch_loader = BatchLoader()
        self._batch_member_loader = BatchMemberLoader()
        self._artifact_loader = ArtifactLoader()

    @property
    def connection(self) -> Connection:
        return self._connection

    @property
    def dialect_name(self) -> str:
        return self._connection.dialect.name

    def get_export_terminal_witness(self, run_id: str) -> AuditExportTerminalWitness:
        row = self._connection.execute(
            select(runs_table.c.run_id, runs_table.c.status, runs_table.c.completed_at).where(runs_table.c.run_id == run_id)
        ).one_or_none()
        if row is None:
            raise ValueError(f"Run not found: {run_id}")
        try:
            status = RunStatus(row.status)
        except ValueError as exc:
            raise AuditIntegrityError(f"run {run_id!r} has an invalid persisted status") from exc
        if status not in _EXPORT_TERMINAL or row.completed_at is None:
            raise AuditIntegrityError(f"run {run_id!r} is not immutable export-terminal")
        completed_at = row.completed_at
        if completed_at.tzinfo is None or completed_at.utcoffset() is None:
            completed_at = completed_at.replace(tzinfo=UTC)
        return AuditExportTerminalWitness(
            source_run_id=row.run_id,
            source_status=status,
            source_completed_at=completed_at,
        )

    def get_run(self, run_id: str) -> Any | None:
        row = self._connection.execute(select(runs_table).where(runs_table.c.run_id == run_id)).one_or_none()
        return None if row is None else self._run_loader.load(row)

    def get_run_attribution(self, run_id: str) -> tuple[str, str] | None:
        row = self._connection.execute(
            select(
                run_attributions_table.c.initiated_by_user_id,
                run_attributions_table.c.auth_provider_type,
            ).where(run_attributions_table.c.run_id == run_id)
        ).one_or_none()
        if row is None:
            return None
        if type(row.initiated_by_user_id) is not str or not row.initiated_by_user_id:
            raise AuditIntegrityError("run attribution user ID is corrupt")
        if type(row.auth_provider_type) is not str or not row.auth_provider_type:
            raise AuditIntegrityError("run attribution provider type is corrupt")
        return row.initiated_by_user_id, row.auth_provider_type

    def get_web_plugin_policy_evidence(self, run_id: str) -> Any | None:
        row = self._connection.execute(
            select(run_web_plugin_policy_table).where(run_web_plugin_policy_table.c.run_id == run_id)
        ).one_or_none()
        if row is None:
            return None
        try:
            return WebPluginPolicyEvidence(
                schema_version=row.schema_version,
                policy_hash=row.policy_hash,
                snapshot_hash=row.snapshot_hash,
                authorized_plugin_ids=tuple(json.loads(row.authorized_plugin_ids_json)),
                available_plugin_ids=tuple(json.loads(row.available_plugin_ids_json)),
                control_modes=tuple(tuple(item) for item in json.loads(row.control_modes_json)),
                selected_implementations=tuple(tuple(item) for item in json.loads(row.selected_implementations_json)),
                selected_profile_aliases=tuple(tuple(item) for item in json.loads(row.selected_profile_aliases_json)),
                plugin_code_identities=tuple(tuple(item) for item in json.loads(row.plugin_code_identities_json)),
                binding_generation_fingerprint=row.binding_generation_fingerprint,
                decision_codes=tuple(json.loads(row.decision_codes_json)),
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise AuditIntegrityError(f"Web plugin-policy evidence is corrupt for run {run_id}") from exc

    def get_secret_resolutions_for_run(self, run_id: str) -> list[Any]:
        rows = self._connection.execute(
            select(secret_resolutions_table)
            .where(secret_resolutions_table.c.run_id == run_id)
            .order_by(secret_resolutions_table.c.timestamp)
        ).fetchall()
        return [
            SecretResolution(
                resolution_id=row.resolution_id,
                run_id=row.run_id,
                timestamp=row.timestamp,
                env_var_name=row.env_var_name,
                source=row.source,
                vault_url=row.vault_url,
                secret_name=row.secret_name,
                fingerprint=row.fingerprint,
                resolution_latency_ms=row.resolution_latency_ms,
            )
            for row in rows
        ]

    def get_nodes(self, run_id: str) -> list[Any]:
        rows = self._connection.execute(
            select(nodes_table)
            .where(nodes_table.c.run_id == run_id)
            .order_by(nodes_table.c.sequence_in_pipeline.nullslast(), nodes_table.c.registered_at, nodes_table.c.node_id)
        ).fetchall()
        return [self._node_loader.load(row) for row in rows]

    def get_edges(self, run_id: str) -> list[Any]:
        rows = self._connection.execute(
            select(edges_table).where(edges_table.c.run_id == run_id).order_by(edges_table.c.created_at, edges_table.c.edge_id)
        ).fetchall()
        return [self._edge_loader.load(row) for row in rows]

    def get_operations_for_run(self, run_id: str) -> list[Any]:
        rows = self._connection.execute(
            select(operations_table).where(operations_table.c.run_id == run_id).order_by(operations_table.c.started_at)
        ).fetchall()
        return [self._operation_loader.load(row) for row in rows]

    def get_all_operation_calls_for_run(self, run_id: str) -> list[Any]:
        rows = self._connection.execute(
            select(calls_table)
            .join(operations_table, calls_table.c.operation_id == operations_table.c.operation_id)
            .where(operations_table.c.run_id == run_id)
            .order_by(calls_table.c.operation_id, calls_table.c.call_index)
        ).fetchall()
        return [self._call_loader.load(row) for row in rows]

    def get_validation_errors_for_run(self, run_id: str) -> list[Any]:
        rows = self._connection.execute(
            select(validation_errors_table).where(validation_errors_table.c.run_id == run_id).order_by(validation_errors_table.c.created_at)
        ).fetchall()
        return [self._validation_error_loader.load(row) for row in rows]

    def get_transform_errors_for_run(self, run_id: str) -> list[Any]:
        rows = self._connection.execute(
            select(transform_errors_table).where(transform_errors_table.c.run_id == run_id).order_by(transform_errors_table.c.created_at)
        ).fetchall()
        return [self._transform_error_loader.load(row) for row in rows]

    def iter_rows_for_run(self, run_id: str, *, batch_size: int) -> Iterator[list[Any]]:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive exact integer")
        last_ingest_sequence: int | None = None
        while True:
            query = select(rows_table).where(rows_table.c.run_id == run_id)
            if last_ingest_sequence is not None:
                query = query.where(rows_table.c.ingest_sequence > last_ingest_sequence)
            rows = self._connection.execute(query.order_by(rows_table.c.ingest_sequence).limit(batch_size)).fetchall()
            if not rows:
                return
            last_ingest_sequence = rows[-1].ingest_sequence
            if last_ingest_sequence is None:
                raise AuditIntegrityError("export row has NULL ingest_sequence")
            yield [self._row_loader.load(row) for row in rows]
            if len(rows) < batch_size:
                return

    def get_tokens_for_rows(self, run_id: str, row_ids: list[str]) -> list[Any]:
        result: list[Any] = []
        for chunk in _chunks(row_ids):
            rows = self._connection.execute(
                select(tokens_table)
                .where(tokens_table.c.run_id == run_id, tokens_table.c.row_id.in_(chunk))
                .order_by(tokens_table.c.row_id, tokens_table.c.created_at, tokens_table.c.token_id)
            ).fetchall()
            result.extend(self._token_loader.load(row) for row in rows)
        return result

    def get_token_parents_for_tokens(self, token_ids: list[str]) -> list[Any]:
        result: list[Any] = []
        for chunk in _chunks(token_ids):
            rows = self._connection.execute(
                select(token_parents_table)
                .where(token_parents_table.c.token_id.in_(chunk))
                .order_by(token_parents_table.c.token_id, token_parents_table.c.ordinal)
            ).fetchall()
            result.extend(self._token_parent_loader.load(row) for row in rows)
        return result

    def get_token_outcomes_for_tokens(self, run_id: str, token_ids: list[str]) -> list[Any]:
        result: list[Any] = []
        for chunk in _chunks(token_ids):
            rows = self._connection.execute(
                select(token_outcomes_table)
                .where(token_outcomes_table.c.run_id == run_id, token_outcomes_table.c.token_id.in_(chunk))
                .order_by(token_outcomes_table.c.token_id, token_outcomes_table.c.recorded_at)
            ).fetchall()
            result.extend(self._token_outcome_loader.load(row) for row in rows)
        return result

    def get_scheduler_events_for_tokens(self, run_id: str, token_ids: list[str]) -> list[Any]:
        result: list[Any] = []
        for chunk in _chunks(token_ids):
            rows = self._connection.execute(
                select(scheduler_events_table)
                .where(scheduler_events_table.c.run_id == run_id, scheduler_events_table.c.token_id.in_(chunk))
                .order_by(scheduler_events_table.c.recorded_at, scheduler_events_table.c.event_id)
            ).fetchall()
            result.extend(self._scheduler_event_loader.load(row) for row in rows)
        return result

    def get_node_states_for_tokens(self, run_id: str, token_ids: list[str]) -> list[Any]:
        result: list[Any] = []
        for chunk in _chunks(token_ids):
            rows = self._connection.execute(
                select(node_states_table)
                .where(node_states_table.c.run_id == run_id, node_states_table.c.token_id.in_(chunk))
                .order_by(node_states_table.c.token_id, node_states_table.c.step_index, node_states_table.c.attempt)
            ).fetchall()
            result.extend(self._node_state_loader.load(row) for row in rows)
        return result

    def get_routing_events_for_states(self, state_ids: list[str]) -> list[Any]:
        rows: list[Any] = []
        for chunk in _chunks(state_ids):
            rows.extend(
                self._connection.execute(
                    select(routing_events_table, node_states_table.c.step_index, node_states_table.c.attempt)
                    .join(
                        node_states_table,
                        and_(
                            routing_events_table.c.state_id == node_states_table.c.state_id,
                            routing_events_table.c.run_id == node_states_table.c.run_id,
                        ),
                    )
                    .where(routing_events_table.c.state_id.in_(chunk))
                ).fetchall()
            )
        rows.sort(key=lambda row: (row.step_index, row.attempt, row.state_id, row.ordinal, row.event_id))
        return [self._routing_event_loader.load(row) for row in rows]

    def get_calls_for_states(self, state_ids: list[str]) -> list[Any]:
        rows: list[Any] = []
        for chunk in _chunks(state_ids):
            rows.extend(
                self._connection.execute(
                    select(calls_table, node_states_table.c.step_index, node_states_table.c.attempt)
                    .join(node_states_table, calls_table.c.state_id == node_states_table.c.state_id)
                    .where(calls_table.c.state_id.in_(chunk))
                ).fetchall()
            )
        rows.sort(key=lambda row: (row.step_index, row.attempt, row.state_id, row.call_index))
        return [self._call_loader.load(row) for row in rows]

    def get_batches(self, run_id: str) -> list[Any]:
        rows = self._connection.execute(
            select(batches_table).where(batches_table.c.run_id == run_id).order_by(batches_table.c.created_at, batches_table.c.batch_id)
        ).fetchall()
        return [self._batch_loader.load(row) for row in rows]

    def get_all_batch_members_for_run(self, run_id: str) -> list[Any]:
        rows = self._connection.execute(
            select(batch_members_table)
            .join(
                batches_table,
                (batch_members_table.c.batch_id == batches_table.c.batch_id) & (batch_members_table.c.run_id == batches_table.c.run_id),
            )
            .where(batches_table.c.run_id == run_id)
            .order_by(batch_members_table.c.batch_id, batch_members_table.c.ordinal)
        ).fetchall()
        return [self._batch_member_loader.load(row) for row in rows]

    def get_artifacts(self, run_id: str) -> list[Any]:
        rows = self._connection.execute(
            select(artifacts_table)
            .where(artifacts_table.c.run_id == run_id)
            .order_by(artifacts_table.c.created_at, artifacts_table.c.artifact_id)
        ).fetchall()
        return [self._artifact_loader.load(row) for row in rows]


@contextmanager
def open_export_read_transaction(engine: Engine) -> Iterator[ConnectionBoundExportReadModel]:
    """Open one explicit, stable export snapshot on a dedicated connection."""

    if not isinstance(engine, Engine):
        raise TypeError("engine must be a SQLAlchemy Engine")
    connection = engine.connect()
    transaction = None
    try:
        if connection.dialect.name == "postgresql":
            connection = connection.execution_options(isolation_level="REPEATABLE READ")
            transaction = connection.begin()
        elif connection.dialect.name == "sqlite":
            # Landscape's SQLite ``begin`` event emits an explicit DEFERRED
            # ``BEGIN`` (pysqlite autocommit is disabled), so this is both a
            # SQLAlchemy-owned transaction and a real DB snapshot before the
            # initial registry lookup.
            transaction = connection.begin()
        else:
            transaction = connection.begin()
        yield ConnectionBoundExportReadModel(connection)
        transaction.commit()
    except BaseException:
        if transaction is not None and transaction.is_active:
            transaction.rollback()
        raise
    finally:
        connection.close()


__all__ = ["AuditExportTerminalWitness", "ConnectionBoundExportReadModel", "open_export_read_transaction"]
