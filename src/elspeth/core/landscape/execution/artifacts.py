"""Sink artifact registration (split from ``ExecutionRepository``).

Owns the ``artifacts`` table: registration of files/URIs produced by sinks
and deterministic artifact reads for export.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.engine import Connection
from sqlalchemy.exc import SQLAlchemyError

from elspeth.contracts import Artifact
from elspeth.contracts.results import require_no_artifact_uri_credentials
from elspeth.core.ids import generate_id
from elspeth.core.landscape._database_ops import DatabaseOps
from elspeth.core.landscape._helpers import now
from elspeth.core.landscape.errors import LandscapeRecordError
from elspeth.core.landscape.model_loaders import ArtifactLoader
from elspeth.core.landscape.schema import artifacts_table


class ArtifactRepository:
    """Sink artifact registration for the audit trail."""

    def __init__(
        self,
        ops: DatabaseOps,
        *,
        artifact_loader: ArtifactLoader,
    ) -> None:
        self._ops = ops
        self._artifact_loader = artifact_loader

    def register_artifact(
        self,
        run_id: str,
        state_id: str,
        sink_node_id: str,
        artifact_type: str,
        path: str,
        content_hash: str,
        size_bytes: int,
        *,
        artifact_id: str | None = None,
        idempotency_key: str | None = None,
        conn: Connection | None = None,
    ) -> Artifact:
        """Register an artifact produced by a sink.

        Args:
            run_id: Run that produced this artifact
            state_id: Node state that produced this artifact
            sink_node_id: Sink node that wrote the artifact
            artifact_type: Type of artifact (csv, json, etc.)
            path: File path or URI
            content_hash: Hash of artifact content
            size_bytes: Size of artifact in bytes
            artifact_id: Optional artifact ID
            idempotency_key: Optional key for retry deduplication

        Returns:
            Artifact model
        """
        artifact_id = artifact_id or generate_id()
        timestamp = now()
        require_no_artifact_uri_credentials(path)

        artifact = Artifact(
            artifact_id=artifact_id,
            run_id=run_id,
            produced_by_state_id=state_id,
            sink_node_id=sink_node_id,
            artifact_type=artifact_type,
            path_or_uri=path,
            content_hash=content_hash,
            size_bytes=size_bytes,
            created_at=timestamp,
            idempotency_key=idempotency_key,
        )

        stmt = artifacts_table.insert().values(
            artifact_id=artifact.artifact_id,
            run_id=artifact.run_id,
            produced_by_state_id=artifact.produced_by_state_id,
            sink_node_id=artifact.sink_node_id,
            artifact_type=artifact.artifact_type,
            path_or_uri=artifact.path_or_uri,
            content_hash=artifact.content_hash,
            size_bytes=artifact.size_bytes,
            idempotency_key=artifact.idempotency_key,
            created_at=artifact.created_at,
        )
        if conn is None:
            self._ops.execute_insert(stmt)
        else:
            try:
                result = conn.execute(stmt)
            except SQLAlchemyError as exc:
                raise LandscapeRecordError(
                    f"register_artifact failed for state_id={state_id!r} — database rejected audit write: {type(exc).__name__}"
                ) from exc
            if result.rowcount == 0:
                raise LandscapeRecordError(f"register_artifact: zero rows affected for state_id={state_id!r} — audit write failed")

        return artifact

    def get_artifacts(
        self,
        run_id: str,
        *,
        sink_node_id: str | None = None,
    ) -> list[Artifact]:
        """Get artifacts for a run.

        Args:
            run_id: Run ID
            sink_node_id: Optional filter by sink

        Returns:
            List of Artifact models
        """
        query = select(artifacts_table).where(artifacts_table.c.run_id == run_id)

        if sink_node_id is not None:
            query = query.where(artifacts_table.c.sink_node_id == sink_node_id)

        # Order for deterministic export signatures
        query = query.order_by(artifacts_table.c.created_at, artifacts_table.c.artifact_id)
        rows = self._ops.execute_fetchall(query)
        return [self._artifact_loader.load(row) for row in rows]
