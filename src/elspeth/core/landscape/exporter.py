"""Landscape audit trail exporter.

Exports complete audit data for a run in a format suitable for
compliance review and legal inquiry.
"""

import hashlib
import hmac
from collections import defaultdict
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

from elspeth.core.canonical import canonical_json
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.recorder import LandscapeRecorder


class LandscapeExporter:
    """Export Landscape audit data for a run.

    Produces a flat sequence of records suitable for CSV/JSON export.
    Each record has a 'record_type' field indicating its category.

    Record types:
    - run: Run metadata (one per export)
    - node: Registered plugins
    - edge: Graph edges
    - row: Source rows
    - token: Row instances
    - token_parent: Token lineage for forks/joins
    - node_state: Processing records
    - routing_event: Routing decisions
    - call: External calls
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
            "status": run.status,
            "started_at": run.started_at.isoformat() if run.started_at else None,
            "completed_at": run.completed_at.isoformat() if run.completed_at else None,
            "canonical_version": run.canonical_version,
            "config_hash": run.config_hash,
            "reproducibility_grade": run.reproducibility_grade,
        }

        # Nodes
        for node in self._recorder.get_nodes(run_id):
            yield {
                "record_type": "node",
                "run_id": run_id,
                "node_id": node.node_id,
                "plugin_name": node.plugin_name,
                "node_type": node.node_type,
                "plugin_version": node.plugin_version,
                "config_hash": node.config_hash,
                "schema_hash": node.schema_hash,
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
                "default_mode": edge.default_mode,
            }

        # Rows and their tokens/states
        for row in self._recorder.get_rows(run_id):
            yield {
                "record_type": "row",
                "run_id": run_id,
                "row_id": row.row_id,
                "row_index": row.row_index,
                "source_node_id": row.source_node_id,
                "source_data_hash": row.source_data_hash,
            }

            # Tokens for this row
            for token in self._recorder.get_tokens(row.row_id):
                yield {
                    "record_type": "token",
                    "run_id": run_id,
                    "token_id": token.token_id,
                    "row_id": token.row_id,
                    "step_in_pipeline": token.step_in_pipeline,
                    "branch_name": token.branch_name,
                    "fork_group_id": token.fork_group_id,
                    "join_group_id": token.join_group_id,
                }

                # Token parents (for fork/join lineage)
                for parent in self._recorder.get_token_parents(token.token_id):
                    yield {
                        "record_type": "token_parent",
                        "run_id": run_id,
                        "token_id": parent.token_id,
                        "parent_token_id": parent.parent_token_id,
                        "ordinal": parent.ordinal,
                    }

                # Node states for this token
                for state in self._recorder.get_node_states_for_token(token.token_id):
                    yield {
                        "record_type": "node_state",
                        "run_id": run_id,
                        "state_id": state.state_id,
                        "token_id": state.token_id,
                        "node_id": state.node_id,
                        "step_index": state.step_index,
                        "attempt": state.attempt,
                        "status": state.status,
                        "input_hash": state.input_hash,
                        "output_hash": state.output_hash,
                        "duration_ms": state.duration_ms,
                        "started_at": (
                            state.started_at.isoformat() if state.started_at else None
                        ),
                        "completed_at": (
                            state.completed_at.isoformat()
                            if state.completed_at
                            else None
                        ),
                    }

                    # Routing events for this state
                    for event in self._recorder.get_routing_events(state.state_id):
                        yield {
                            "record_type": "routing_event",
                            "run_id": run_id,
                            "event_id": event.event_id,
                            "state_id": event.state_id,
                            "edge_id": event.edge_id,
                            "routing_group_id": event.routing_group_id,
                            "ordinal": event.ordinal,
                            "mode": event.mode,
                            "reason_hash": event.reason_hash,
                        }

                    # External calls for this state
                    for call in self._recorder.get_calls(state.state_id):
                        yield {
                            "record_type": "call",
                            "run_id": run_id,
                            "call_id": call.call_id,
                            "state_id": call.state_id,
                            "call_index": call.call_index,
                            "call_type": call.call_type,
                            "status": call.status,
                            "request_hash": call.request_hash,
                            "response_hash": call.response_hash,
                            "latency_ms": call.latency_ms,
                        }

        # Batches
        for batch in self._recorder.get_batches(run_id):
            yield {
                "record_type": "batch",
                "run_id": run_id,
                "batch_id": batch.batch_id,
                "aggregation_node_id": batch.aggregation_node_id,
                "attempt": batch.attempt,
                "status": batch.status,
                "trigger_reason": batch.trigger_reason,
                "created_at": (
                    batch.created_at.isoformat() if batch.created_at else None
                ),
                "completed_at": (
                    batch.completed_at.isoformat() if batch.completed_at else None
                ),
            }

            # Batch members
            for member in self._recorder.get_batch_members(batch.batch_id):
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
