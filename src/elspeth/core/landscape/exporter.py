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
from typing import Any

from elspeth.contracts import NodeStateCompleted, NodeStateOpen, NodeStatePending
from elspeth.core.canonical import canonical_json
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.recorder import LandscapeRecorder


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
    - row: Source rows
    - token: Row instances
    - token_parent: Token lineage for forks/joins
    - node_state: Processing records
    - routing_event: Routing decisions
    - call: External calls (may have state_id OR operation_id)
    - batch: Aggregation batches
    - batch_member: Batch membership
    - artifact: Sink outputs

    Example:
        db = LandscapeDB.from_url("sqlite:///audit.db")
        exporter = LandscapeExporter(db)

        # Export to JSON lines
        for record in exporter.export_run(run_id):
            json_file.write(json.dumps(record) + "\\n")

        # Export to CSV (group by record_type)
        records = list(exporter.export_run(run_id))
        for rtype in ["run", "node", "edge", "row", "token"]:
            typed_records = [r for r in records if r["record_type"] == rtype]
            write_csv(f"{rtype}.csv", typed_records)
    """

    def __init__(
        self,
        db: LandscapeDB,
        signing_key: bytes | None = None,
    ) -> None:
        """Initialize exporter with database connection.

        Args:
            db: LandscapeDB instance to export from
            signing_key: Optional HMAC key for signing exported records.
                        Required if sign=True is passed to export_run().
        """
        self._db = db
        self._recorder = LandscapeRecorder(db)
        self._signing_key = signing_key

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

        for record in self._iter_records(run_id):
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

    def _iter_records(self, run_id: str) -> Iterator[dict[str, Any]]:
        """Internal: iterate over raw records (no signing).

        Bug 76r fix: Uses batch queries to pre-load all data, avoiding N+1 pattern.
        Previous implementation did ~25,000 queries for 1000 rows with 5 states each.
        New implementation does ~10 queries regardless of data size.

        Args:
            run_id: The run ID to export

        Yields:
            Dict records with 'record_type' field

        Raises:
            ValueError: If run_id is not found
        """
        # Run metadata
        run = self._recorder.get_run(run_id)
        if run is None:
            raise ValueError(f"Run not found: {run_id}")

        yield {
            "record_type": "run",
            "run_id": run.run_id,
            "status": run.status.value,
            "started_at": run.started_at.isoformat() if run.started_at else None,
            "completed_at": run.completed_at.isoformat() if run.completed_at else None,
            "canonical_version": run.canonical_version,
            "config_hash": run.config_hash,
            # Full resolved settings for audit trail portability (not just hash)
            "settings": json.loads(run.settings_json),
            "reproducibility_grade": run.reproducibility_grade,
        }

        # Secret resolutions (run-level provenance for Key Vault secrets)
        for resolution in self._recorder.get_secret_resolutions_for_run(run_id):
            yield {
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

        # Nodes
        for node in self._recorder.get_nodes(run_id):
            yield {
                "record_type": "node",
                "run_id": run_id,
                "node_id": node.node_id,
                "plugin_name": node.plugin_name,
                "node_type": node.node_type.value,
                "plugin_version": node.plugin_version,
                "determinism": node.determinism.value,
                "config_hash": node.config_hash,
                # Full resolved config for audit trail portability (not just hash)
                "config": json.loads(node.config_json),
                "schema_hash": node.schema_hash,
                "schema_mode": node.schema_mode,
                "schema_fields": node.schema_fields,
                "sequence_in_pipeline": node.sequence_in_pipeline,
            }

        # Edges
        for edge in self._recorder.get_edges(run_id):
            yield {
                "record_type": "edge",
                "run_id": run_id,
                "edge_id": edge.edge_id,
                "from_node_id": edge.from_node_id,
                "to_node_id": edge.to_node_id,
                "label": edge.label,
                "default_mode": edge.default_mode.value,
            }

        # Operations (source loads, sink writes)
        all_operations = self._recorder.get_operations_for_run(run_id)

        # Batch query: Pre-load all operation-parented calls (N+1 fix)
        all_op_calls = self._recorder.get_all_operation_calls_for_run(run_id)
        op_calls_by_operation: dict[str, list[Any]] = defaultdict(list)
        for call in all_op_calls:
            if call.operation_id:
                op_calls_by_operation[call.operation_id].append(call)

        for operation in all_operations:
            yield {
                "record_type": "operation",
                "run_id": run_id,
                "operation_id": operation.operation_id,
                "node_id": operation.node_id,
                "operation_type": operation.operation_type,
                "status": operation.status,
                "started_at": operation.started_at.isoformat() if operation.started_at else None,
                "completed_at": operation.completed_at.isoformat() if operation.completed_at else None,
                "duration_ms": operation.duration_ms,
                "error_message": operation.error_message,
                # BUG #9: Add payload reference fields
                "input_data_ref": operation.input_data_ref,
                "input_data_hash": operation.input_data_hash,
                "output_data_ref": operation.output_data_ref,
                "output_data_hash": operation.output_data_hash,
            }

            # External calls for this operation (from pre-loaded dict)
            for call in op_calls_by_operation.get(operation.operation_id, []):
                yield {
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
                    "latency_ms": call.latency_ms,
                    # BUG #9: Add payload references, error, and timestamp
                    "request_ref": call.request_ref,
                    "response_ref": call.response_ref,
                    "error_json": call.error_json,
                    "created_at": call.created_at.isoformat() if call.created_at else None,
                }

        # === Bug 76r fix: Pre-load all row-related data with batch queries ===
        # This replaces the N+1 pattern where nested loops issued per-entity queries.
        # Now we do 5 batch queries and build lookup dicts in memory.

        # Batch query 1: All tokens for this run
        all_tokens = self._recorder.get_all_tokens_for_run(run_id)
        tokens_by_row: dict[str, list[Any]] = defaultdict(list)
        for token in all_tokens:
            tokens_by_row[token.row_id].append(token)

        # Batch query 2: All token parents for this run
        all_parents = self._recorder.get_all_token_parents_for_run(run_id)
        parents_by_token: dict[str, list[Any]] = defaultdict(list)
        for parent in all_parents:
            parents_by_token[parent.token_id].append(parent)

        # Batch query 3: All node states for this run
        all_states = self._recorder.get_all_node_states_for_run(run_id)
        states_by_token: dict[str, list[Any]] = defaultdict(list)
        for state in all_states:
            states_by_token[state.token_id].append(state)

        # Batch query 4: All routing events for this run
        all_routing_events = self._recorder.get_all_routing_events_for_run(run_id)
        events_by_state: dict[str, list[Any]] = defaultdict(list)
        for event in all_routing_events:
            events_by_state[event.state_id].append(event)

        # Batch query 5: All state-parented calls for this run
        all_calls = self._recorder.get_all_calls_for_run(run_id)
        calls_by_state: dict[str, list[Any]] = defaultdict(list)
        for call in all_calls:
            if call.state_id:  # Should always be true for state-parented calls
                calls_by_state[call.state_id].append(call)

        # Now iterate through rows using pre-loaded data (no more per-entity queries)
        for row in self._recorder.get_rows(run_id):
            yield {
                "record_type": "row",
                "run_id": run_id,
                "row_id": row.row_id,
                "row_index": row.row_index,
                "source_node_id": row.source_node_id,
                "source_data_hash": row.source_data_hash,
            }

            # Tokens for this row (from pre-loaded dict)
            for token in tokens_by_row.get(row.row_id, []):
                yield {
                    "record_type": "token",
                    "run_id": run_id,
                    "token_id": token.token_id,
                    "row_id": token.row_id,
                    "step_in_pipeline": token.step_in_pipeline,
                    "branch_name": token.branch_name,
                    "fork_group_id": token.fork_group_id,
                    "join_group_id": token.join_group_id,
                    "expand_group_id": token.expand_group_id,
                }

                # Token parents (from pre-loaded dict)
                for parent in parents_by_token.get(token.token_id, []):
                    yield {
                        "record_type": "token_parent",
                        "run_id": run_id,
                        "token_id": parent.token_id,
                        "parent_token_id": parent.parent_token_id,
                        "ordinal": parent.ordinal,
                    }

                # Node states for this token (from pre-loaded dict)
                for state in states_by_token.get(token.token_id, []):
                    # Handle discriminated union types
                    if isinstance(state, NodeStateOpen):
                        yield {
                            "record_type": "node_state",
                            "run_id": run_id,
                            "state_id": state.state_id,
                            "token_id": state.token_id,
                            "node_id": state.node_id,
                            "step_index": state.step_index,
                            "attempt": state.attempt,
                            "status": state.status.value,
                            "input_hash": state.input_hash,
                            "output_hash": None,
                            "duration_ms": None,
                            "started_at": state.started_at.isoformat(),
                            "completed_at": None,
                            # BUG #9: Add context, error, and success reason fields
                            "context_before_json": state.context_before_json,
                            "context_after_json": None,  # OPEN states don't have after context
                            "error_json": None,  # OPEN states aren't failed
                            "success_reason_json": None,  # OPEN states aren't completed
                        }
                    elif isinstance(state, NodeStatePending):
                        yield {
                            "record_type": "node_state",
                            "run_id": run_id,
                            "state_id": state.state_id,
                            "token_id": state.token_id,
                            "node_id": state.node_id,
                            "step_index": state.step_index,
                            "attempt": state.attempt,
                            "status": state.status.value,
                            "input_hash": state.input_hash,
                            "output_hash": None,
                            "duration_ms": state.duration_ms,
                            "started_at": state.started_at.isoformat(),
                            "completed_at": state.completed_at.isoformat(),
                            # BUG #9: Add context, error, and success reason fields
                            "context_before_json": state.context_before_json,
                            "context_after_json": state.context_after_json,
                            "error_json": None,  # PENDING states aren't failed
                            "success_reason_json": None,  # PENDING states aren't completed yet
                        }
                    elif isinstance(state, NodeStateCompleted):
                        yield {
                            "record_type": "node_state",
                            "run_id": run_id,
                            "state_id": state.state_id,
                            "token_id": state.token_id,
                            "node_id": state.node_id,
                            "step_index": state.step_index,
                            "attempt": state.attempt,
                            "status": state.status.value,
                            "input_hash": state.input_hash,
                            "output_hash": state.output_hash,
                            "duration_ms": state.duration_ms,
                            "started_at": state.started_at.isoformat(),
                            "completed_at": state.completed_at.isoformat(),
                            # BUG #9: Add context, error, and success reason fields
                            "context_before_json": state.context_before_json,
                            "context_after_json": state.context_after_json,
                            "error_json": None,  # COMPLETED states aren't failed
                            "success_reason_json": state.success_reason_json,
                        }
                    else:  # NodeStateFailed
                        yield {
                            "record_type": "node_state",
                            "run_id": run_id,
                            "state_id": state.state_id,
                            "token_id": state.token_id,
                            "node_id": state.node_id,
                            "step_index": state.step_index,
                            "attempt": state.attempt,
                            "status": state.status.value,
                            "input_hash": state.input_hash,
                            "output_hash": state.output_hash,
                            "duration_ms": state.duration_ms,
                            "started_at": state.started_at.isoformat(),
                            "completed_at": state.completed_at.isoformat(),
                            # BUG #9: Add context, error, and success reason fields
                            "context_before_json": state.context_before_json,
                            "context_after_json": state.context_after_json,
                            "error_json": state.error_json,
                            "success_reason_json": None,  # FAILED states aren't completed
                        }

                    # Routing events for this state (from pre-loaded dict)
                    for event in events_by_state.get(state.state_id, []):
                        yield {
                            "record_type": "routing_event",
                            "run_id": run_id,
                            "event_id": event.event_id,
                            "state_id": event.state_id,
                            "edge_id": event.edge_id,
                            "routing_group_id": event.routing_group_id,
                            "ordinal": event.ordinal,
                            "mode": event.mode.value,
                            "reason_hash": event.reason_hash,
                            # BUG #9: Add payload reference and timestamp
                            "reason_ref": event.reason_ref,
                            "created_at": event.created_at.isoformat() if event.created_at else None,
                        }

                    # External calls for this state (from pre-loaded dict)
                    for call in calls_by_state.get(state.state_id, []):
                        yield {
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
                            "latency_ms": call.latency_ms,
                            # BUG #9: Add payload references, error, and timestamp
                            "request_ref": call.request_ref,
                            "response_ref": call.response_ref,
                            "error_json": call.error_json,
                            "created_at": call.created_at.isoformat() if call.created_at else None,
                        }

        # Batches
        all_batches = self._recorder.get_batches(run_id)

        # Batch query: Pre-load all batch members (N+1 fix)
        all_batch_members = self._recorder.get_all_batch_members_for_run(run_id)
        members_by_batch: dict[str, list[Any]] = defaultdict(list)
        for member in all_batch_members:
            members_by_batch[member.batch_id].append(member)

        for batch in all_batches:
            yield {
                "record_type": "batch",
                "run_id": run_id,
                "batch_id": batch.batch_id,
                "aggregation_node_id": batch.aggregation_node_id,
                "attempt": batch.attempt,
                "status": batch.status.value,
                "trigger_type": batch.trigger_type,
                "trigger_reason": batch.trigger_reason,
                "created_at": (batch.created_at.isoformat() if batch.created_at else None),
                "completed_at": (batch.completed_at.isoformat() if batch.completed_at else None),
            }

            # Batch members (from pre-loaded dict)
            for member in members_by_batch.get(batch.batch_id, []):
                yield {
                    "record_type": "batch_member",
                    "run_id": run_id,
                    "batch_id": member.batch_id,
                    "token_id": member.token_id,
                    "ordinal": member.ordinal,
                }

        # Artifacts
        for artifact in self._recorder.get_artifacts(run_id):
            yield {
                "record_type": "artifact",
                "run_id": run_id,
                "artifact_id": artifact.artifact_id,
                "sink_node_id": artifact.sink_node_id,
                "produced_by_state_id": artifact.produced_by_state_id,
                "artifact_type": artifact.artifact_type,
                "path_or_uri": artifact.path_or_uri,
                "content_hash": artifact.content_hash,
                "size_bytes": artifact.size_bytes,
            }

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

        for record in self.export_run(run_id, sign=sign):
            record_type = record["record_type"]
            groups[record_type].append(record)

        # Return as regular dict with deterministic key order
        # (Python 3.7+ dicts maintain insertion order, and export_run
        # yields records in a consistent type order)
        return dict(groups)
