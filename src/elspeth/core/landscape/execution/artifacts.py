"""Sink artifact registration (split from ``ExecutionRepository``).

Owns the ``artifacts`` table: registration of files/URIs produced by sinks
and deterministic artifact reads for export.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sqlalchemy import Insert, select
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Connection, RowMapping
from sqlalchemy.exc import SQLAlchemyError

from elspeth.contracts import Artifact, ArtifactPublicationEvidenceKind
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
        sink_node_id: str,
        artifact_type: str,
        path: str,
        content_hash: str,
        size_bytes: int,
        *,
        state_id: str | None = None,
        sink_effect_id: str | None = None,
        artifact_id: str | None = None,
        idempotency_key: str | None = None,
        publication_performed: bool = True,
        publication_evidence_kind: ArtifactPublicationEvidenceKind | None = None,
        conn: Connection | None = None,
    ) -> Artifact:
        """Register an artifact produced by a sink.

        Args:
            run_id: Run that produced this artifact
            state_id: Legacy node state that produced this artifact
            sink_effect_id: Epoch-26 sink effect that produced this artifact
            sink_node_id: Sink node that wrote the artifact
            artifact_type: Type of artifact (csv, json, etc.)
            path: File path or URI
            content_hash: Hash of artifact content
            size_bytes: Size of artifact in bytes
            artifact_id: Optional artifact ID
            idempotency_key: Optional opaque logical-effect key. Within one
                run, an identical retry returns the original durable artifact;
                reuse for divergent linkage or descriptor fields fails closed.
                ``None`` preserves independent-insert behavior.

        Returns:
            The durable Artifact model. On an idempotent retry this is the
            original row, including its artifact ID and creation timestamp.
        """
        if (state_id is None) == (sink_effect_id is None):
            raise ValueError("register_artifact requires exactly one producer link")
        artifact_id = artifact_id or generate_id()
        require_no_artifact_uri_credentials(path)
        evidence_kind: ArtifactPublicationEvidenceKind = (
            ("legacy_returned" if state_id is not None else "returned") if publication_evidence_kind is None else publication_evidence_kind
        )

        artifact = Artifact(
            artifact_id=artifact_id,
            run_id=run_id,
            produced_by_state_id=state_id,
            sink_effect_id=sink_effect_id,
            sink_node_id=sink_node_id,
            artifact_type=artifact_type,
            path_or_uri=path,
            content_hash=content_hash,
            size_bytes=size_bytes,
            created_at=now(),
            idempotency_key=idempotency_key,
            publication_performed=publication_performed,
            publication_evidence_kind=evidence_kind,
        )

        values = {
            "artifact_id": artifact.artifact_id,
            "run_id": artifact.run_id,
            "produced_by_state_id": artifact.produced_by_state_id,
            "sink_effect_id": artifact.sink_effect_id,
            "sink_node_id": artifact.sink_node_id,
            "artifact_type": artifact.artifact_type,
            "path_or_uri": artifact.path_or_uri,
            "content_hash": artifact.content_hash,
            "size_bytes": artifact.size_bytes,
            "idempotency_key": artifact.idempotency_key,
            "publication_performed": artifact.publication_performed,
            "publication_evidence_kind": artifact.publication_evidence_kind,
            "created_at": artifact.created_at,
        }
        if conn is not None:
            return self._insert_or_fetch(conn, values)

        try:
            with self._ops.write_connection() as owned_conn:
                return self._insert_or_fetch(owned_conn, values)
        except SQLAlchemyError as exc:
            raise LandscapeRecordError(
                f"register_artifact failed for producer={artifact.producer_kind!r} — database rejected audit write: {type(exc).__name__}"
            ) from exc

    def _insert_or_fetch(self, conn: Connection, values: Mapping[str, Any]) -> Artifact:
        """Atomically insert a logical artifact effect or return its winner."""
        idempotency_key = values["idempotency_key"]
        inserted_artifact_id: str | None = None
        try:
            if idempotency_key is None:
                conn.execute(artifacts_table.insert().values(**values))
            else:
                # RETURNING is the cross-driver authority for insert-vs-conflict:
                # psycopg may report rowcount=-1 even when the insert succeeds.
                inserted_artifact_id = conn.execute(
                    self._idempotent_insert(conn, values).returning(artifacts_table.c.artifact_id)
                ).scalar_one_or_none()
        except SQLAlchemyError as exc:
            raise LandscapeRecordError(
                f"register_artifact failed for producer linkage — database rejected audit write: {type(exc).__name__}"
            ) from exc

        if inserted_artifact_id is not None and inserted_artifact_id != values["artifact_id"]:
            raise LandscapeRecordError(
                "register_artifact insert returned an artifact identity different from the proposed row; audit write is ambiguous"
            )

        if idempotency_key is None:
            identity_query = select(artifacts_table).where(artifacts_table.c.artifact_id == values["artifact_id"])
        else:
            identity_query = (
                select(artifacts_table)
                .where(artifacts_table.c.run_id == values["run_id"])
                .where(artifacts_table.c.idempotency_key == idempotency_key)
            )
        rows = conn.execute(identity_query).fetchmany(2)
        if not rows:
            raise LandscapeRecordError("register_artifact could not read its durable row after insert-or-fetch")
        if len(rows) > 1:
            raise LandscapeRecordError(
                "register_artifact idempotency lookup matched multiple durable rows; artifact logical-effect identity is ambiguous"
            )
        row = rows[0]

        if idempotency_key is not None:
            self._validate_idempotent_effect(row._mapping, values)
        return self._artifact_loader.load(row)

    @staticmethod
    def _idempotent_insert(conn: Connection, values: Mapping[str, Any]) -> Insert:
        """Build the backend-native conflict-safe insert for the partial key."""
        if conn.dialect.name == "sqlite":
            return (
                sqlite_insert(artifacts_table)
                .values(**values)
                .on_conflict_do_nothing(
                    index_elements=[artifacts_table.c.run_id, artifacts_table.c.idempotency_key],
                    index_where=artifacts_table.c.idempotency_key.is_not(None),
                )
            )
        if conn.dialect.name == "postgresql":
            return (
                postgresql_insert(artifacts_table)
                .values(**values)
                .on_conflict_do_nothing(
                    index_elements=[artifacts_table.c.run_id, artifacts_table.c.idempotency_key],
                    index_where=artifacts_table.c.idempotency_key.is_not(None),
                )
            )
        raise LandscapeRecordError(
            f"register_artifact idempotency is unsupported for database dialect {conn.dialect.name!r}; refusing an unfenced insert"
        )

    @staticmethod
    def _validate_idempotent_effect(existing: RowMapping, proposed: Mapping[str, Any]) -> None:
        """Reject reuse of one logical-effect key for divergent evidence."""
        effect_fields = (
            "produced_by_state_id",
            "sink_effect_id",
            "sink_node_id",
            "artifact_type",
            "path_or_uri",
            "content_hash",
            "size_bytes",
            "publication_performed",
            "publication_evidence_kind",
        )
        mismatches = [field for field in effect_fields if existing[field] != proposed[field]]
        if mismatches:
            raise LandscapeRecordError(
                "register_artifact idempotency conflict: the durable logical effect differs in "
                + ", ".join(mismatches)
                + "; existing artifact evidence was preserved unchanged"
            )

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
