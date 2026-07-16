"""Landscape audit trail exporter.

Exports complete audit data for a run in a format suitable for
compliance review and legal inquiry.

Export records are SELF-CONTAINED: they include the full resolved
configuration, not just hashes. This allows third-party auditors to
understand exactly what configuration drove each decision without
requiring access to the original database.
"""

import hashlib
import hmac
import json
from collections import defaultdict
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any, Protocol

from elspeth.contracts import (
    BatchMember,
    Call,
    NodeState,
    RoutingEvent,
    Row,
    Token,
    TokenOutcome,
    TokenParent,
)
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.export_records import (
    BatchExportRecord,
    BatchMemberExportRecord,
    CallExportRecord,
    EdgeExportRecord,
    ExportRecord,
    NodeExportRecord,
    OperationExportRecord,
    RoutingEventExportRecord,
    RowExportRecord,
    RunExportRecord,
    SchedulerEventExportRecord,
    SecretResolutionExportRecord,
    TokenExportRecord,
    TokenOutcomeExportRecord,
    TokenParentExportRecord,
    TransformErrorExportRecord,
    ValidationErrorExportRecord,
    WebPluginPolicyExportRecord,
)
from elspeth.contracts.freeze import deep_thaw
from elspeth.core.canonical import canonical_json
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.export_mappers import artifact_to_export_record, node_state_to_export_record
from elspeth.core.landscape.factory import RecorderFactory


class ExportReadModel(Protocol):
    """Read-only repository surface needed by ``LandscapeExporter``."""

    def get_run(self, run_id: str) -> Any | None: ...

    def get_run_attribution(self, run_id: str) -> tuple[str, str] | None: ...

    def get_web_plugin_policy_evidence(self, run_id: str) -> Any | None: ...

    def get_secret_resolutions_for_run(self, run_id: str) -> list[Any]: ...

    def get_nodes(self, run_id: str) -> list[Any]: ...

    def get_edges(self, run_id: str) -> list[Any]: ...

    def get_operations_for_run(self, run_id: str) -> list[Any]: ...

    def get_all_operation_calls_for_run(self, run_id: str) -> list[Any]: ...

    def get_validation_errors_for_run(self, run_id: str) -> list[Any]: ...

    def get_transform_errors_for_run(self, run_id: str) -> list[Any]: ...

    def iter_rows_for_run(self, run_id: str, *, batch_size: int) -> Iterator[list[Any]]: ...

    def get_tokens_for_rows(self, run_id: str, row_ids: list[str]) -> list[Any]: ...

    def get_token_parents_for_tokens(self, token_ids: list[str]) -> list[Any]: ...

    def get_token_outcomes_for_tokens(self, run_id: str, token_ids: list[str]) -> list[Any]: ...

    def get_scheduler_events_for_tokens(self, run_id: str, token_ids: list[str]) -> list[Any]: ...

    def get_node_states_for_tokens(self, run_id: str, token_ids: list[str]) -> list[Any]: ...

    def get_routing_events_for_states(self, state_ids: list[str]) -> list[Any]: ...

    def get_calls_for_states(self, state_ids: list[str]) -> list[Any]: ...

    def get_batches(self, run_id: str) -> list[Any]: ...

    def get_all_batch_members_for_run(self, run_id: str) -> list[Any]: ...

    def get_artifacts(self, run_id: str) -> list[Any]: ...


class RecorderFactoryExportReadModel:
    """Adapter from the repository factory bundle to the exporter read model."""

    def __init__(self, factory: RecorderFactory) -> None:
        self._factory = factory

    def get_run(self, run_id: str) -> Any | None:
        return self._factory.run_lifecycle.get_run(run_id)

    def get_run_attribution(self, run_id: str) -> tuple[str, str] | None:
        return self._factory.run_lifecycle.get_run_attribution(run_id)

    def get_web_plugin_policy_evidence(self, run_id: str) -> Any | None:
        return self._factory.run_lifecycle.get_web_plugin_policy_evidence(run_id)

    def get_secret_resolutions_for_run(self, run_id: str) -> list[Any]:
        return self._factory.run_lifecycle.get_secret_resolutions_for_run(run_id)

    def get_nodes(self, run_id: str) -> list[Any]:
        return self._factory.data_flow.get_nodes(run_id)

    def get_edges(self, run_id: str) -> list[Any]:
        return self._factory.data_flow.get_edges(run_id)

    def get_operations_for_run(self, run_id: str) -> list[Any]:
        return self._factory.execution.get_operations_for_run(run_id)

    def get_all_operation_calls_for_run(self, run_id: str) -> list[Any]:
        return self._factory.execution.get_all_operation_calls_for_run(run_id)

    def get_validation_errors_for_run(self, run_id: str) -> list[Any]:
        return self._factory.data_flow.get_validation_errors_for_run(run_id)

    def get_transform_errors_for_run(self, run_id: str) -> list[Any]:
        return self._factory.data_flow.get_transform_errors_for_run(run_id)

    def iter_rows_for_run(self, run_id: str, *, batch_size: int) -> Iterator[list[Any]]:
        return self._factory.query.iter_rows_for_run(run_id, batch_size=batch_size)

    def get_tokens_for_rows(self, run_id: str, row_ids: list[str]) -> list[Any]:
        return self._factory.query.get_tokens_for_rows(run_id, row_ids)

    def get_token_parents_for_tokens(self, token_ids: list[str]) -> list[Any]:
        return self._factory.query.get_token_parents_for_tokens(token_ids)

    def get_token_outcomes_for_tokens(self, run_id: str, token_ids: list[str]) -> list[Any]:
        return self._factory.query.get_token_outcomes_for_tokens(run_id, token_ids)

    def get_scheduler_events_for_tokens(self, run_id: str, token_ids: list[str]) -> list[Any]:
        return self._factory.query.get_scheduler_events_for_tokens(run_id, token_ids)

    def get_node_states_for_tokens(self, run_id: str, token_ids: list[str]) -> list[Any]:
        return self._factory.query.get_node_states_for_tokens(run_id, token_ids)

    def get_routing_events_for_states(self, state_ids: list[str]) -> list[Any]:
        return self._factory.query.get_routing_events_for_states(state_ids)

    def get_calls_for_states(self, state_ids: list[str]) -> list[Any]:
        return self._factory.query.get_calls_for_states(state_ids)

    def get_batches(self, run_id: str) -> list[Any]:
        return self._factory.execution.get_batches(run_id)

    def get_all_batch_members_for_run(self, run_id: str) -> list[Any]:
        return self._factory.execution.get_all_batch_members_for_run(run_id)

    def get_artifacts(self, run_id: str) -> list[Any]:
        return self._factory.execution.get_artifacts(run_id)


class LandscapeExporter:
    """Export Landscape audit data for a run.

    Produces a flat sequence of records suitable for CSV/JSON export.
    Each record has a 'record_type' field indicating its category.

    Record types:
    - run: Run metadata (one per export)
    - secret_resolution: Key Vault secret provenance (run-level)
    - node: Registered plugins
    - edge: Graph edges
    - operation: Source/sink I/O operations
    - validation_error: Source validation failures
    - transform_error: Transform processing failures
    - row: Source rows
    - token: Row instances
    - token_parent: Token lineage for forks/joins
    - token_outcome: Terminal state for tokens
    - scheduler_event: Durable scheduler state transitions and lease attribution
    - node_state: Processing records
    - routing_event: Routing decisions
    - call: External calls (may have state_id OR operation_id)
    - batch: Aggregation batches
    - batch_member: Batch membership
    - artifact: Sink outputs

    Memory envelope: the row family (rows, tokens, token_parents,
    token_outcomes, scheduler_events, node_states, routing_events, and
    state-parented calls) streams in bounded row batches, so its memory
    cost is O(row_batch_size), not O(run). The run-structure families
    (nodes, edges, operations + their calls, validation/transform errors,
    batches + members, artifacts) are each materialized as one full list;
    they scale with pipeline structure and error/batch counts rather than
    row volume. export_run_grouped() is the exception: it deliberately
    materializes the entire export into one dict.

    Example:
        db = LandscapeDB.from_url("sqlite:///audit.db")
        exporter = LandscapeExporter(db)

        # Export to JSON lines
        for record in exporter.export_run(run_id):
            json_file.write(json.dumps(record) + "\\n")

        # Export to CSV without keeping every record in memory
        for record_type, record in exporter.iter_run_records_by_type(run_id):
            write_csv_record(record_type, record)
    """

    def __init__(
        self,
        db: LandscapeDB,
        signing_key: bytes | None = None,
        *,
        include_raw_error_rows: bool = False,
        row_batch_size: int = 500,
        read_model: ExportReadModel | None = None,
    ) -> None:
        """Initialize exporter with database connection.

        Args:
            db: LandscapeDB instance to export from
            signing_key: Optional HMAC key for signing exported records.
                        Required if sign=True is passed to export_run().
            include_raw_error_rows: Include raw failing-row payloads
                (``row_data_json``) in validation_error / transform_error
                records. Default False (elspeth-384184c6ab): the export is
                an external artifact and every other record type already
                exports hashes/refs instead of raw payloads; error records
                default to ``row_hash``-only correlation, with the raw row
                remaining in the audit database.
            row_batch_size: Rows per streamed batch for the row-family
                sections of the export (elspeth-3ae79a4775). Bounds export
                memory: at most this many rows — plus their tokens, node
                states, and related child records — are resident at once.
                The exported record sequence is identical for every value.
            read_model: Optional narrow read surface for export queries.
                Defaults to a read-model adapter over ``RecorderFactory(db)``.

        Raises:
            ValueError: If row_batch_size < 1
        """
        if row_batch_size < 1:
            raise ValueError(f"row_batch_size must be >= 1, got {row_batch_size}")
        self._db = db
        self._read_model = read_model if read_model is not None else RecorderFactoryExportReadModel(RecorderFactory(db))
        self._signing_key = signing_key
        self._include_raw_error_rows = include_raw_error_rows
        self._row_batch_size = row_batch_size

    @staticmethod
    def _parse_tier1_json(raw_json: str, field_name: str, context: str) -> Any:
        """Parse JSON from Tier 1 audit data, crashing with context on corruption."""
        try:
            return json.loads(raw_json)
        except json.JSONDecodeError as exc:
            raise AuditIntegrityError(
                f"Corrupt {field_name} for {context}: database corruption (Tier 1 violation). Parse error: {exc}"
            ) from exc

    def _sign_record(self, record: dict[str, Any]) -> str:
        """Compute HMAC-SHA256 signature for a record.

        Args:
            record: Record dict to sign (must not contain 'signature' key)

        Returns:
            Hex-encoded HMAC-SHA256 signature

        Raises:
            ValueError: If signing key not configured
        """
        if self._signing_key is None:
            raise ValueError("Signing key not configured")

        # Canonical JSON ensures consistent hash
        canonical = canonical_json(record)
        return hmac.new(
            self._signing_key,
            canonical.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def export_run(
        self,
        run_id: str,
        sign: bool = False,
    ) -> Iterator[dict[str, Any]]:
        """Export all audit data for a run.

        Yields flat dict records with 'record_type' field.
        Order: run -> nodes -> edges -> rows -> tokens -> states -> batches -> artifacts

        Args:
            run_id: The run ID to export
            sign: If True, add HMAC signature to each record and emit
                  a final manifest with running hash chain.

        Yields:
            Dict records with 'record_type' and relevant fields.
            If sign=True, includes 'signature' field and final manifest.

        Raises:
            ValueError: If run_id is not found, or sign=True without signing_key
        """
        if sign and self._signing_key is None:
            raise ValueError("Signing requested but no signing_key provided")

        running_hash = hashlib.sha256()
        record_count = 0

        for typed_record in self._iter_records(run_id):
            # Widen to dict[str, Any] — export_run may add "signature" key
            record: dict[str, Any] = typed_record  # type: ignore[assignment]
            if sign:
                record["signature"] = self._sign_record(record)
                # Update running hash with signed record
                running_hash.update(record["signature"].encode())

            record_count += 1
            yield record

        # Emit manifest if signing
        if sign:
            manifest = {
                "record_type": "manifest",
                "run_id": run_id,
                "record_count": record_count,
                "final_hash": running_hash.hexdigest(),
                "hash_algorithm": "sha256",
                "signature_algorithm": "hmac-sha256",
                "exported_at": datetime.now(UTC).isoformat(),
            }
            manifest["signature"] = self._sign_record(manifest)
            yield manifest

    def iter_run_records_by_type(
        self,
        run_id: str,
        sign: bool = False,
    ) -> Iterator[tuple[str, dict[str, Any]]]:
        """Export audit data as a stream of ``(record_type, record)`` pairs.

        This is the bounded-memory grouping surface for callers that need to
        route records by type, such as multi-file CSV export. It preserves the
        same deterministic order and signing behavior as :meth:`export_run`
        without building a full ``record_type -> list[records]`` map.
        """
        for record in self.export_run(run_id, sign=sign):
            yield record["record_type"], record

    def _iter_records(self, run_id: str) -> Iterator[ExportRecord]:
        """Internal: iterate over raw records (no signing).

        Query strategy: run-structure collections (nodes, edges, operations,
        errors, batches, artifacts) load with one batch query per family —
        the Bug 76r N+1 fix. The row family (rows, tokens, and their child
        records) streams in bounded row batches (elspeth-3ae79a4775): rows
        page via keyset pagination and each batch loads its children with
        set-scoped queries, so memory is bounded by row_batch_size instead
        of run size, at O(row_count / row_batch_size) queries.

        Args:
            run_id: The run ID to export

        Yields:
            Dict records with 'record_type' field

        Raises:
            ValueError: If run_id is not found
        """
        # Run metadata
        run = self._read_model.get_run(run_id)
        if run is None:
            raise ValueError(f"Run not found: {run_id}")
        attribution = self._read_model.get_run_attribution(run_id)

        run_record: RunExportRecord = {
            "record_type": "run",
            "run_id": run.run_id,
            "status": run.status.value,
            "started_at": run.started_at.isoformat() if run.started_at else None,
            "completed_at": run.completed_at.isoformat() if run.completed_at else None,
            "canonical_version": run.canonical_version,
            "config_hash": run.config_hash,
            "initiated_by_user_id": attribution[0] if attribution is not None else None,
            "auth_provider_type": attribution[1] if attribution is not None else None,
            # Full resolved settings for audit trail portability (not just hash)
            "settings": self._parse_tier1_json(run.settings_json, "settings_json", f"run {run_id}"),
            "reproducibility_grade": run.reproducibility_grade.value if run.reproducibility_grade is not None else None,
        }
        yield run_record

        policy_evidence = self._read_model.get_web_plugin_policy_evidence(run_id)
        if policy_evidence is not None:
            policy_record: WebPluginPolicyExportRecord = {
                "record_type": "web_plugin_policy",
                "run_id": run_id,
                "schema_version": policy_evidence.schema_version,
                "policy_hash": policy_evidence.policy_hash,
                "snapshot_hash": policy_evidence.snapshot_hash,
                "authorized_plugin_ids": list(policy_evidence.authorized_plugin_ids),
                "available_plugin_ids": list(policy_evidence.available_plugin_ids),
                "control_modes": [list(item) for item in policy_evidence.control_modes],
                "selected_implementations": [list(item) for item in policy_evidence.selected_implementations],
                "selected_profile_aliases": [list(item) for item in policy_evidence.selected_profile_aliases],
                "plugin_code_identities": [list(item) for item in policy_evidence.plugin_code_identities],
                "binding_generation_fingerprint": policy_evidence.binding_generation_fingerprint,
                "decision_codes": list(policy_evidence.decision_codes),
            }
            yield policy_record

        # Secret resolutions (run-level provenance for Key Vault secrets)
        for resolution in self._read_model.get_secret_resolutions_for_run(run_id):
            secret_record: SecretResolutionExportRecord = {
                "record_type": "secret_resolution",
                "run_id": run_id,
                "resolution_id": resolution.resolution_id,
                "timestamp": resolution.timestamp,
                "env_var_name": resolution.env_var_name,
                "source": resolution.source,
                "vault_url": resolution.vault_url,
                "secret_name": resolution.secret_name,
                "fingerprint": resolution.fingerprint,
                "resolution_latency_ms": resolution.resolution_latency_ms,
            }
            yield secret_record

        # Nodes
        for node in self._read_model.get_nodes(run_id):
            node_record: NodeExportRecord = {
                "record_type": "node",
                "run_id": run_id,
                "node_id": node.node_id,
                "plugin_name": node.plugin_name,
                "node_type": node.node_type.value,
                "plugin_version": node.plugin_version,
                "source_file_hash": node.source_file_hash,
                "determinism": node.determinism.value,
                "config_hash": node.config_hash,
                # Full resolved config for audit trail portability (not just hash)
                "config": self._parse_tier1_json(node.config_json, "config_json", f"node {node.node_id} in run {run_id}"),
                "schema_hash": node.schema_hash,
                "schema_mode": node.schema_mode,
                "schema_fields": deep_thaw(node.schema_fields) if node.schema_fields is not None else None,
                "sequence_in_pipeline": node.sequence_in_pipeline,
                "registered_at": node.registered_at.isoformat(),
            }
            yield node_record

        # Edges
        for edge in self._read_model.get_edges(run_id):
            edge_record: EdgeExportRecord = {
                "record_type": "edge",
                "run_id": run_id,
                "edge_id": edge.edge_id,
                "from_node_id": edge.from_node_id,
                "to_node_id": edge.to_node_id,
                "label": edge.label,
                "default_mode": edge.default_mode.value,
                "created_at": edge.created_at.isoformat(),
            }
            yield edge_record

        # Operations (source loads, sink writes)
        all_operations = self._read_model.get_operations_for_run(run_id)

        # Batch query: Pre-load all operation-parented calls (N+1 fix)
        all_op_calls = self._read_model.get_all_operation_calls_for_run(run_id)
        op_calls_by_operation: defaultdict[str, list[Call]] = defaultdict(list)
        for call in all_op_calls:
            if not call.operation_id:
                raise AuditIntegrityError(
                    f"Operation-parented call '{call.call_id}' has no operation_id. Run: {run_id}. This violates the Call XOR constraint."
                )
            op_calls_by_operation[call.operation_id].append(call)

        for operation in all_operations:
            operation_record: OperationExportRecord = {
                "record_type": "operation",
                "run_id": run_id,
                "operation_id": operation.operation_id,
                "node_id": operation.node_id,
                "operation_type": operation.operation_type,
                "sink_effect_id": operation.sink_effect_id,
                "status": operation.status,
                "started_at": operation.started_at.isoformat() if operation.started_at else None,
                "completed_at": operation.completed_at.isoformat() if operation.completed_at else None,
                "duration_ms": operation.duration_ms,
                "error_message": operation.error_message,
                "input_data_ref": operation.input_data_ref,
                "input_data_hash": operation.input_data_hash,
                "output_data_ref": operation.output_data_ref,
                "output_data_hash": operation.output_data_hash,
            }
            yield operation_record

            # External calls for this operation (from pre-loaded dict). An
            # explicit `in` guard (not `.get(key, default)`, which trips R1)
            # avoids defaultdict inflating itself with a throwaway empty list
            # for every operation that has no calls.
            for call in op_calls_by_operation[operation.operation_id] if operation.operation_id in op_calls_by_operation else ():
                op_call_record: CallExportRecord = {
                    "record_type": "call",
                    "run_id": run_id,
                    "call_id": call.call_id,
                    "state_id": None,  # Operation calls don't have state_id
                    "operation_id": call.operation_id,
                    "call_index": call.call_index,
                    "call_type": call.call_type.value,
                    "status": call.status.value,
                    "request_hash": call.request_hash,
                    "response_hash": call.response_hash,
                    "resolved_prompt_template_hash": call.resolved_prompt_template_hash,
                    "latency_ms": call.latency_ms,
                    "request_ref": call.request_ref,
                    "response_ref": call.response_ref,
                    "error_json": call.error_json,
                    "created_at": call.created_at.isoformat() if call.created_at else None,
                }
                yield op_call_record

        # Source validation and transform error audit evidence.
        # Repository getters provide deterministic created_at ordering.
        # Raw-row minimization (elspeth-384184c6ab): row_data_json is
        # redacted to None unless the operator opted in — row_hash remains
        # for correlation back to the audit DB, matching the hash/ref
        # discipline every other exported record type already follows.
        for validation_error in self._read_model.get_validation_errors_for_run(run_id):
            validation_error_record: ValidationErrorExportRecord = {
                "record_type": "validation_error",
                "run_id": run_id,
                "error_id": validation_error.error_id,
                "node_id": validation_error.node_id,
                "row_id": validation_error.row_id,
                "row_hash": validation_error.row_hash,
                "row_data_json": validation_error.row_data_json if self._include_raw_error_rows else None,
                "error": validation_error.error,
                "schema_mode": validation_error.schema_mode,
                "destination": validation_error.destination,
                "created_at": validation_error.created_at.isoformat(),
                "violation_type": validation_error.violation_type,
                "original_field_name": validation_error.original_field_name,
                "normalized_field_name": validation_error.normalized_field_name,
                "expected_type": validation_error.expected_type,
                "actual_type": validation_error.actual_type,
            }
            yield validation_error_record

        for transform_error in self._read_model.get_transform_errors_for_run(run_id):
            transform_error_record: TransformErrorExportRecord = {
                "record_type": "transform_error",
                "run_id": run_id,
                "error_id": transform_error.error_id,
                "token_id": transform_error.token_id,
                "transform_id": transform_error.transform_id,
                "row_hash": transform_error.row_hash,
                "row_data_json": transform_error.row_data_json if self._include_raw_error_rows else None,
                "error_details_json": transform_error.error_details_json,
                "destination": transform_error.destination,
                "created_at": transform_error.created_at.isoformat(),
            }
            yield transform_error_record

        # === Row family: streamed in bounded row batches (elspeth-3ae79a4775) ===
        # Bug 76r replaced per-entity N+1 queries with full-run preloads; that
        # fixed the query count but left export memory proportional to the
        # entire run. The row family now streams: rows page through keyset
        # pagination and child collections are batch-loaded per row batch, so
        # memory scales with row_batch_size while the query count stays
        # O(row_count / row_batch_size).
        for row_batch in self._read_model.iter_rows_for_run(run_id, batch_size=self._row_batch_size):
            yield from self._iter_row_batch_records(run_id, row_batch)

        yield from self._iter_batch_and_artifact_records(run_id)

    def _iter_row_batch_records(self, run_id: str, row_batch: list[Row]) -> Iterator[ExportRecord]:
        """Yield row-family records for one bounded batch of rows.

        Loads the batch's child collections (tokens, token parents, token
        outcomes, scheduler events, node states, routing events, state calls)
        with set-scoped batch queries and groups them into per-parent lookup
        dicts, then emits records nested as row -> token -> (parents,
        outcomes, scheduler events, states -> (routing events, calls)). Every
        set-scoped query preserves the per-parent ordering of its full-run
        counterpart, so the emitted record sequence — and therefore every
        record signature and the manifest hash chain — is independent of
        row_batch_size.
        """
        row_ids = [row.row_id for row in row_batch]
        batch_tokens = self._read_model.get_tokens_for_rows(run_id, row_ids)
        tokens_by_row: defaultdict[str, list[Token]] = defaultdict(list)
        for token in batch_tokens:
            tokens_by_row[token.row_id].append(token)

        token_ids = [token.token_id for token in batch_tokens]
        parents_by_token: defaultdict[str, list[TokenParent]] = defaultdict(list)
        for parent in self._read_model.get_token_parents_for_tokens(token_ids):
            parents_by_token[parent.token_id].append(parent)

        outcomes_by_token: defaultdict[str, list[TokenOutcome]] = defaultdict(list)
        for outcome in self._read_model.get_token_outcomes_for_tokens(run_id, token_ids):
            outcomes_by_token[outcome.token_id].append(outcome)

        scheduler_events_by_token: defaultdict[str, list[Any]] = defaultdict(list)
        for scheduler_event in self._read_model.get_scheduler_events_for_tokens(run_id, token_ids):
            scheduler_events_by_token[scheduler_event.token_id].append(scheduler_event)

        batch_states = self._read_model.get_node_states_for_tokens(run_id, token_ids)
        states_by_token: defaultdict[str, list[NodeState]] = defaultdict(list)
        for state in batch_states:
            states_by_token[state.token_id].append(state)

        state_ids = [state.state_id for state in batch_states]
        events_by_state: defaultdict[str, list[RoutingEvent]] = defaultdict(list)
        for event in self._read_model.get_routing_events_for_states(state_ids):
            events_by_state[event.state_id].append(event)

        calls_by_state: defaultdict[str, list[Call]] = defaultdict(list)
        for call in self._read_model.get_calls_for_states(state_ids):
            if not call.state_id:
                raise AuditIntegrityError(
                    f"State-parented call '{call.call_id}' has no state_id. Run: {run_id}. This violates the Call XOR constraint."
                )
            calls_by_state[call.state_id].append(call)

        for row in row_batch:
            if row.source_row_index is None:
                raise AuditIntegrityError(
                    f"Row '{row.row_id}' has no source_row_index. Run: {run_id}. This violates the multi-source row identity contract."
                )
            if row.ingest_sequence is None:
                raise AuditIntegrityError(
                    f"Row '{row.row_id}' has no ingest_sequence. Run: {run_id}. This violates the multi-source row identity contract."
                )
            row_record: RowExportRecord = {
                "record_type": "row",
                "run_id": run_id,
                "row_id": row.row_id,
                "row_index": row.row_index,
                "source_row_index": row.source_row_index,
                "ingest_sequence": row.ingest_sequence,
                "source_node_id": row.source_node_id,
                "source_data_hash": row.source_data_hash,
                "source_data_ref": row.source_data_ref,
                "created_at": row.created_at.isoformat(),
            }
            yield row_record

            # Tokens for this row (from pre-loaded dict). An explicit `in`
            # guard (not `.get(key, default)`, which trips R1) avoids
            # defaultdict inflating itself with a throwaway empty list for
            # every row that has no tokens.
            for token in tokens_by_row[row.row_id] if row.row_id in tokens_by_row else ():
                token_record: TokenExportRecord = {
                    "record_type": "token",
                    "run_id": run_id,
                    "token_id": token.token_id,
                    "row_id": token.row_id,
                    "step_in_pipeline": token.step_in_pipeline,
                    "branch_name": token.branch_name,
                    "fork_group_id": token.fork_group_id,
                    "join_group_id": token.join_group_id,
                    "expand_group_id": token.expand_group_id,
                    "created_at": token.created_at.isoformat(),
                }
                yield token_record

                # Token parents (from pre-loaded dict). An explicit `in`
                # guard (not `.get(key, default)`, which trips R1) avoids
                # defaultdict inflating itself with a throwaway empty list for
                # every token that has no parents.
                for parent in parents_by_token[token.token_id] if token.token_id in parents_by_token else ():
                    token_parent_record: TokenParentExportRecord = {
                        "record_type": "token_parent",
                        "run_id": run_id,
                        "token_id": parent.token_id,
                        "parent_token_id": parent.parent_token_id,
                        "ordinal": parent.ordinal,
                    }
                    yield token_parent_record

                # Token outcomes (from pre-loaded dict). An explicit `in`
                # guard (not `.get(key, default)`, which trips R1) avoids
                # defaultdict inflating itself with a throwaway empty list for
                # every token that has no outcomes.
                for outcome in outcomes_by_token[token.token_id] if token.token_id in outcomes_by_token else ():
                    token_outcome_record: TokenOutcomeExportRecord = {
                        "record_type": "token_outcome",
                        "run_id": run_id,
                        "outcome_id": outcome.outcome_id,
                        "token_id": outcome.token_id,
                        "outcome": outcome.outcome.value if outcome.outcome is not None else None,
                        "path": outcome.path.value,
                        "completed": outcome.completed,
                        "recorded_at": outcome.recorded_at.isoformat(),
                        "sink_name": outcome.sink_name,
                        "batch_id": outcome.batch_id,
                        "fork_group_id": outcome.fork_group_id,
                        "join_group_id": outcome.join_group_id,
                        "expand_group_id": outcome.expand_group_id,
                        "error_hash": outcome.error_hash,
                        "context_json": outcome.context_json,
                        "expected_branches_json": outcome.expected_branches_json,
                    }
                    yield token_outcome_record

                # Scheduler transition events (from pre-loaded dict). An
                # explicit `in` guard (not `.get(key, default)`, which trips
                # R1) avoids defaultdict inflating itself with a throwaway
                # empty list for every token that has no scheduler events.
                for event in scheduler_events_by_token[token.token_id] if token.token_id in scheduler_events_by_token else ():
                    scheduler_event_record: SchedulerEventExportRecord = {
                        "record_type": "scheduler_event",
                        "run_id": run_id,
                        "event_id": event.event_id,
                        "token_id": event.token_id,
                        "work_item_id": event.work_item_id,
                        "node_id": event.node_id,
                        "event_type": event.event_type.value,
                        "from_status": event.from_status.value if event.from_status is not None else None,
                        "to_status": event.to_status.value,
                        "from_lease_owner": event.from_lease_owner,
                        "to_lease_owner": event.to_lease_owner,
                        "from_lease_expires_at": (
                            event.from_lease_expires_at.isoformat() if event.from_lease_expires_at is not None else None
                        ),
                        "to_lease_expires_at": (event.to_lease_expires_at.isoformat() if event.to_lease_expires_at is not None else None),
                        "from_attempt": event.from_attempt,
                        "to_attempt": event.to_attempt,
                        "recorded_at": event.recorded_at.isoformat(),
                        "caller_owner": event.caller_owner,
                        "context_json": event.context_json,
                    }
                    yield scheduler_event_record

                # Node states for this token (from pre-loaded dict). An
                # explicit `in` guard (not `.get(key, default)`, which trips
                # R1) avoids defaultdict inflating itself with a throwaway
                # empty list for every token that has no node states.
                for state in states_by_token[token.token_id] if token.token_id in states_by_token else ():
                    yield node_state_to_export_record(run_id, state)

                    # Routing events for this state (from pre-loaded dict).
                    # An explicit `in` guard (not `.get(key, default)`, which
                    # trips R1) avoids defaultdict inflating itself with a
                    # throwaway empty list for every state that has no
                    # routing events.
                    for event in events_by_state[state.state_id] if state.state_id in events_by_state else ():
                        routing_event_record: RoutingEventExportRecord = {
                            "record_type": "routing_event",
                            "run_id": run_id,
                            "event_id": event.event_id,
                            "state_id": event.state_id,
                            "edge_id": event.edge_id,
                            "routing_group_id": event.routing_group_id,
                            "ordinal": event.ordinal,
                            "mode": event.mode.value,
                            "reason_hash": event.reason_hash,
                            "reason_ref": event.reason_ref,
                            "created_at": event.created_at.isoformat() if event.created_at else None,
                        }
                        yield routing_event_record

                    # External calls for this state (from pre-loaded dict).
                    # An explicit `in` guard (not `.get(key, default)`, which
                    # trips R1) avoids defaultdict inflating itself with a
                    # throwaway empty list for every state that has no calls.
                    for call in calls_by_state[state.state_id] if state.state_id in calls_by_state else ():
                        state_call_record: CallExportRecord = {
                            "record_type": "call",
                            "run_id": run_id,
                            "call_id": call.call_id,
                            "state_id": call.state_id,
                            "operation_id": None,  # State calls don't have operation_id
                            "call_index": call.call_index,
                            "call_type": call.call_type.value,
                            "status": call.status.value,
                            "request_hash": call.request_hash,
                            "response_hash": call.response_hash,
                            "resolved_prompt_template_hash": call.resolved_prompt_template_hash,
                            "latency_ms": call.latency_ms,
                            "request_ref": call.request_ref,
                            "response_ref": call.response_ref,
                            "error_json": call.error_json,
                            "created_at": call.created_at.isoformat() if call.created_at else None,
                        }
                        yield state_call_record

    def _iter_batch_and_artifact_records(self, run_id: str) -> Iterator[ExportRecord]:
        """Yield batch, batch_member, and artifact records for the run."""
        # Batches
        all_batches = self._read_model.get_batches(run_id)

        # Batch query: Pre-load all batch members (N+1 fix)
        all_batch_members = self._read_model.get_all_batch_members_for_run(run_id)
        members_by_batch: defaultdict[str, list[BatchMember]] = defaultdict(list)
        for member in all_batch_members:
            members_by_batch[member.batch_id].append(member)

        for batch in all_batches:
            batch_record: BatchExportRecord = {
                "record_type": "batch",
                "run_id": run_id,
                "batch_id": batch.batch_id,
                "aggregation_node_id": batch.aggregation_node_id,
                "attempt": batch.attempt,
                "status": batch.status.value,
                "trigger_type": batch.trigger_type.value if batch.trigger_type is not None else None,
                "trigger_reason": batch.trigger_reason,
                "created_at": (batch.created_at.isoformat() if batch.created_at else None),
                "completed_at": (batch.completed_at.isoformat() if batch.completed_at else None),
            }
            yield batch_record

            # Batch members (from pre-loaded dict). An explicit `in` guard
            # (not `.get(key, default)`, which trips R1) avoids defaultdict
            # inflating itself with a throwaway empty list for every batch
            # that has no members.
            for member in members_by_batch[batch.batch_id] if batch.batch_id in members_by_batch else ():
                batch_member_record: BatchMemberExportRecord = {
                    "record_type": "batch_member",
                    "run_id": member.run_id,
                    "batch_id": member.batch_id,
                    "token_id": member.token_id,
                    "ordinal": member.ordinal,
                }
                yield batch_member_record

        # Artifacts
        for artifact in self._read_model.get_artifacts(run_id):
            yield artifact_to_export_record(run_id, artifact)

    def export_run_grouped(
        self,
        run_id: str,
        sign: bool = False,
    ) -> dict[str, list[dict[str, Any]]]:
        """Export all audit data for a run, grouped by record type.

        This method is useful for CSV export where each record type needs
        its own file (since record types have different schemas).

        Args:
            run_id: The run ID to export
            sign: If True, add HMAC signature to each record

        Returns:
            Dict mapping record_type -> list of records.
            Keys are in deterministic order for signature stability.

        Raises:
            ValueError: If run_id is not found, or sign=True without signing_key
        """
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for record_type, record in self.iter_run_records_by_type(run_id, sign=sign):
            groups[record_type].append(record)

        # Return as regular dict with deterministic key order
        # (Python 3.7+ dicts maintain insertion order, and export_run
        # yields records in a consistent type order)
        return dict(groups)
