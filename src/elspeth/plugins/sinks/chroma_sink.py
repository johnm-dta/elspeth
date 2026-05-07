"""ChromaDB vector store sink plugin.

Writes pipeline rows into a ChromaDB collection. Each row becomes a
document with ChromaDB's default embedding function.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import chromadb
import chromadb.api
import chromadb.errors
import structlog
from pydantic import BaseModel, Field, model_validator

from elspeth.contracts.diversion import SinkWriteResult
from elspeth.contracts.enums import CallStatus, CallType
from elspeth.contracts.errors import (
    AuditIntegrityError,
    DuplicateDocumentError,
    FrameworkBugError,
)
from elspeth.contracts.results import ArtifactDescriptor
from elspeth.contracts.url import SanitizedDatabaseUrl
from elspeth.core.canonical import canonical_json
from elspeth.plugins.infrastructure.base import BaseSink
from elspeth.plugins.infrastructure.clients.retrieval.connection import (
    ChromaConnectionConfig,
)
from elspeth.plugins.infrastructure.config_base import DataPluginConfig
from elspeth.plugins.infrastructure.schema_factory import create_schema_from_config

slog = structlog.get_logger(__name__)


class _ChromaPayloadRejection(Exception):
    """Chroma API rejected the payload with ValueError.

    Wraps ValueError from Chroma write calls (upsert/add) so the error handler
    can distinguish "Chroma rejected our data" (Tier 3) from "bug in our code"
    (framework error). Without this, a broad ValueError catch would suppress
    framework bugs in the surrounding code.
    """


if TYPE_CHECKING:
    from elspeth.contracts.contexts import LifecycleContext, SinkContext
    from elspeth.contracts.data import PluginSchema


class FieldMappingConfig(BaseModel):
    """Maps row field names to ChromaDB document concepts.

    Field values are names of row fields, not literal content.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    document_field: str = Field(description="Row field containing text to embed")
    id_field: str = Field(description="Row field containing document ID")
    metadata_fields: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Row fields to include as ChromaDB metadata",
    )


class ChromaSinkConfig(DataPluginConfig):
    """Configuration for ChromaDB vector store sink.

    Connection fields are flat (matching YAML config convention for sinks)
    and validated by constructing a ChromaConnectionConfig in the model
    validator. This is the same delegation pattern used by
    ChromaSearchProviderConfig.
    """

    _plugin_component_type: ClassVar[str | None] = "sink"

    collection: str = Field(description="ChromaDB collection name")
    mode: Literal["persistent", "client"] = Field(description="Connection mode")
    persist_directory: str | None = Field(default=None)
    host: str | None = Field(default=None)
    port: int = Field(default=8000, ge=1, le=65535)
    ssl: bool = Field(default=True)
    distance_function: Literal["cosine", "l2", "ip"] = Field(default="cosine")

    field_mapping: FieldMappingConfig = Field(description="Maps row fields to ChromaDB document/id/metadata")
    on_duplicate: Literal["overwrite", "skip", "error"] = Field(
        default="overwrite",
        description="Behaviour when a document ID already exists",
    )

    @model_validator(mode="after")
    def validate_connection(self) -> ChromaSinkConfig:
        """Delegate connection validation to ChromaConnectionConfig."""
        ChromaConnectionConfig(
            collection=self.collection,
            mode=self.mode,
            persist_directory=self.persist_directory,
            host=self.host,
            port=self.port,
            ssl=self.ssl,
            distance_function=self.distance_function,
        )
        return self

    @model_validator(mode="after")
    def validate_field_mapping_against_schema(self) -> ChromaSinkConfig:
        """Cross-reference field_mapping field names against schema_config.

        For fixed/flexible schemas, validates that referenced fields exist and
        have compatible types. For observed schemas, fields are unknown at config
        time so validation defers to runtime.
        """
        if self.schema_config.is_observed:
            return self

        fields = self.schema_config.fields
        if fields is None:
            return self

        field_types = {f.name: f.field_type for f in fields}
        fm = self.field_mapping

        # document_field and id_field must exist and be str-compatible
        for attr_name, label in [("document_field", "document_field"), ("id_field", "id_field")]:
            field_name = getattr(fm, attr_name)
            if field_name not in field_types:
                raise ValueError(
                    f"field_mapping.{label} references '{field_name}' which is not in the schema. Declared fields: {sorted(field_types)}"
                )
            ft = field_types[field_name]
            if ft not in ("str", "any"):
                raise ValueError(f"field_mapping.{label} references '{field_name}' which has type '{ft}' — ChromaDB requires str")

        # metadata_fields must exist and have ChromaDB-compatible types
        chroma_metadata_types = {"str", "int", "float", "bool", "any"}
        for mf in fm.metadata_fields:
            if mf not in field_types:
                raise ValueError(
                    f"field_mapping.metadata_fields references '{mf}' which is not in the schema. Declared fields: {sorted(field_types)}"
                )
            ft = field_types[mf]
            if ft not in chroma_metadata_types:
                raise ValueError(
                    f"field_mapping.metadata_fields references '{mf}' which has type '{ft}' — ChromaDB metadata requires str/int/float/bool"
                )

        return self


class ChromaSink(BaseSink):
    """Write pipeline rows into a ChromaDB collection.

    Each row maps to a ChromaDB document via the configured field_mapping.
    Content is hashed (canonical JSON of the actual payload sent) before
    write for audit integrity.

    Trust boundary: Row data arriving at this sink is Tier 2 (types validated
    upstream). ChromaDB itself is an external system — SDK errors
    (chromadb.errors.ChromaError) are caught as infrastructure failures;
    other exceptions crash through as plugin bugs per CLAUDE.md plugin
    ownership rules.
    """

    name = "chroma_sink"
    plugin_version = "1.0.0"
    source_file_hash: str | None = "sha256:89351e5be1d6cf3c"
    config_model = ChromaSinkConfig
    supports_resume = False

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._config = ChromaSinkConfig.from_dict(config, plugin_name=self.name)
        self._schema_class: type[PluginSchema] = create_schema_from_config(
            self._config.schema_config,
            "ChromaSinkRowSchema",
            allow_coercion=False,
        )
        self.input_schema = self._schema_class
        self.declared_required_fields = self._config.schema_config.get_effective_required_fields()

        self._client: chromadb.api.ClientAPI | None = None
        self._collection: chromadb.Collection | None = None
        self._telemetry_emit: Callable[[Any], None] | None = None
        self._total_written = 0
        self._total_bytes = 0

    def on_start(self, ctx: LifecycleContext) -> None:
        super().on_start(ctx)
        self._telemetry_emit = ctx.telemetry_emit

        if self._config.mode == "persistent":
            # Validated by ChromaConnectionConfig: persist_directory is not None
            if self._config.persist_directory is None:
                raise FrameworkBugError(
                    "ChromaSinkConfig.persist_directory is None in 'persistent' mode "
                    "— ChromaConnectionConfig validation should have rejected this"
                )
            self._client = chromadb.PersistentClient(
                path=self._config.persist_directory,
            )
        else:
            # Validated by ChromaConnectionConfig: host is not None
            if self._config.host is None:
                raise FrameworkBugError(
                    "ChromaSinkConfig.host is None in 'client' mode — ChromaConnectionConfig validation should have rejected this"
                )
            self._client = chromadb.HttpClient(
                host=self._config.host,
                port=self._config.port,
                ssl=self._config.ssl,
            )
            self._client.heartbeat()

        self._collection = self._client.get_or_create_collection(
            name=self._config.collection,
            metadata={"hnsw:space": self._config.distance_function},
        )

    @staticmethod
    def _compute_payload_hash(
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]] | None,
    ) -> tuple[str, int]:
        """Compute canonical hash and size for the actual payload being sent."""
        payload = canonical_json({"ids": ids, "documents": documents, "metadatas": metadatas})
        payload_bytes = payload.encode("utf-8")
        return hashlib.sha256(payload_bytes).hexdigest(), len(payload_bytes)

    def write(self, rows: list[dict[str, Any]], ctx: SinkContext) -> SinkWriteResult:
        if self._collection is None:
            raise FrameworkBugError("ChromaSink._collection is None — on_start() was not called before write()")
        collection = self._collection

        chroma_url = SanitizedDatabaseUrl.from_raw_url(f"chromadb://{self._config.collection}")

        if not rows:
            return SinkWriteResult(
                artifact=ArtifactDescriptor.for_database(
                    url=chroma_url,
                    table=self._config.collection,
                    content_hash=hashlib.sha256(b"").hexdigest(),
                    payload_size=0,
                    row_count=0,
                )
            )

        fm = self._config.field_mapping

        # Per-row extraction of required fields (id, document) and optional metadata.
        # Missing or non-string required fields are per-row data problems — divert
        # the row rather than aborting the entire batch.
        ids: list[str] = []
        documents: list[str] = []
        # Per-row metadata: non-empty dict if metadata was extracted, None if the row
        # had none of the configured metadata fields present. ChromaDB rejects empty
        # metadata dicts {}, so rows with no extractable metadata must be sent in a
        # separate batch with metadatas=None.
        per_row_metadata: list[dict[str, Any] | None] = []
        valid_indices: list[int] = []

        for i, row in enumerate(rows):
            # Required fields: id and document must be present and string-typed.
            try:
                raw_id = row[fm.id_field]
                raw_doc = row[fm.document_field]
            except KeyError as exc:
                self._divert_row(
                    row,
                    row_index=i,
                    reason=f"Missing required field: {exc}",
                )
                continue

            if not isinstance(raw_id, str):
                self._divert_row(
                    row,
                    row_index=i,
                    reason=f"Field '{fm.id_field}' must be str, got {type(raw_id).__name__}",
                )
                continue
            if not isinstance(raw_doc, str):
                self._divert_row(
                    row,
                    row_index=i,
                    reason=f"Field '{fm.document_field}' must be str, got {type(raw_doc).__name__}",
                )
                continue

            # Optional metadata fields — extract if configured, validate types.
            # ChromaDB accepts str|int|float|bool|None — anything else is a per-row
            # data problem, not a plugin bug.
            if fm.metadata_fields:
                meta: dict[str, Any] = {}
                bad_fields: dict[str, str] = {}
                for field in fm.metadata_fields:
                    try:
                        value = row[field]
                    except KeyError:
                        # Missing metadata field — skip it (metadata is optional per-field)
                        continue
                    if value is not None and not isinstance(value, (str, int, float, bool)):
                        bad_fields[field] = type(value).__name__
                    elif isinstance(value, float) and not math.isfinite(value):
                        bad_fields[field] = f"non-finite float ({value!r})"
                    else:
                        meta[field] = value
                if bad_fields:
                    self._divert_row(
                        row,
                        row_index=i,
                        reason=f"Invalid ChromaDB metadata types: {bad_fields}",
                    )
                    continue
                # ChromaDB rejects empty metadata dicts — mark as None so the row
                # can be sent in a metadata-free sub-batch instead of crashing.
                per_row_metadata.append(meta if meta else None)
            else:
                per_row_metadata.append(None)

            ids.append(raw_id)
            documents.append(raw_doc)
            valid_indices.append(i)

        # Partition rows by metadata availability.  ChromaDB rejects empty
        # metadata dicts {}, so rows where all configured metadata fields were
        # absent must be sent in a separate API call with metadatas=None.
        # When metadata_fields is not configured, per_row_metadata is all-None
        # and every row lands in the no-metadata partition.
        meta_ids: list[str] = []
        meta_documents: list[str] = []
        meta_metadatas: list[dict[str, Any]] = []
        nometa_ids: list[str] = []
        nometa_documents: list[str] = []

        for id_, doc, m in zip(ids, documents, per_row_metadata, strict=True):
            if m is not None:
                meta_ids.append(id_)
                meta_documents.append(doc)
                meta_metadatas.append(m)
            else:
                nometa_ids.append(id_)
                nometa_documents.append(doc)

        # Handle all-rejected case: nothing to write, return zero-write artifact
        if not ids:
            content_hash, payload_size = self._compute_payload_hash([], [], None)
            try:
                ctx.record_call(
                    call_type=CallType.VECTOR,
                    status=CallStatus.SUCCESS,
                    request_data={
                        "operation": self._config.on_duplicate.upper(),
                        "collection": self._config.collection,
                        "row_count": 0,
                        "batch_size": len(rows),
                    },
                    response_data={
                        "rows_written": 0,
                    },
                    latency_ms=0.0,
                    provider="chromadb",
                )
            except Exception as exc:
                raise AuditIntegrityError(
                    f"Failed to record metadata-rejected ChromaDB write to audit trail (collection={self._config.collection!r})."
                ) from exc
            return SinkWriteResult(
                artifact=ArtifactDescriptor.for_database(
                    url=chroma_url,
                    table=self._config.collection,
                    content_hash=content_hash,
                    payload_size=payload_size,
                    row_count=0,
                ),
                diversions=self._get_diversions(),
            )

        # Build the sub-batches to send.  Most batches are homogeneous (all
        # rows have metadata or none do), so we usually make a single call.
        # Mixed batches produce two calls.
        sub_batches: list[tuple[list[str], list[str], list[dict[str, Any]] | None]] = []
        if meta_ids:
            sub_batches.append((meta_ids, meta_documents, meta_metadatas))
        if nometa_ids:
            sub_batches.append((nometa_ids, nometa_documents, None))

        # Aggregate write-ids/documents/metadatas across sub-batches for audit.
        # These will be updated for skip/error mode to reflect actual payload sent.
        all_write_ids: list[str] = []
        all_write_documents: list[str] = []
        all_write_metadatas: list[dict[str, Any]] = []
        total_rows_written = 0
        total_rows_skipped = 0
        all_skipped_ids: list[str] = []

        start_time = time.perf_counter()
        try:
            for batch_ids, batch_docs, batch_metadatas in sub_batches:
                write_ids = batch_ids
                write_documents = batch_docs
                write_metadatas = batch_metadatas
                rows_written = len(batch_ids)

                if self._config.on_duplicate == "overwrite":
                    try:
                        collection.upsert(
                            ids=batch_ids,
                            documents=batch_docs,
                            metadatas=batch_metadatas,  # type: ignore[arg-type]  # chromadb stub Metadata vs dict[str, Any]
                        )
                    except ValueError as ve:
                        raise _ChromaPayloadRejection(str(ve)) from ve
                elif self._config.on_duplicate == "skip":
                    existing = collection.get(ids=batch_ids)
                    existing_ids = set(existing["ids"])
                    new_indices = [i for i, id_ in enumerate(batch_ids) if id_ not in existing_ids]

                    skipped_ids = [id_ for id_ in batch_ids if id_ in existing_ids]
                    all_skipped_ids.extend(skipped_ids)
                    total_rows_skipped += len(skipped_ids)

                    if new_indices:
                        write_ids = [batch_ids[i] for i in new_indices]
                        write_documents = [batch_docs[i] for i in new_indices]
                        write_metadatas = [batch_metadatas[i] for i in new_indices] if batch_metadatas is not None else None
                        rows_written = len(new_indices)
                        try:
                            collection.add(
                                ids=write_ids,
                                documents=write_documents,
                                metadatas=write_metadatas,  # type: ignore[arg-type]  # chromadb stub Metadata vs dict[str, Any]
                            )
                        except ValueError as ve:
                            raise _ChromaPayloadRejection(str(ve)) from ve
                    else:
                        write_ids = []
                        write_documents = []
                        write_metadatas = None
                        rows_written = 0
                elif self._config.on_duplicate == "error":
                    existing = collection.get(ids=batch_ids)
                    existing_ids = set(existing["ids"])
                    duplicates = [id_ for id_ in batch_ids if id_ in existing_ids]
                    if duplicates:
                        raise DuplicateDocumentError(
                            collection=self._config.collection,
                            duplicate_ids=duplicates,
                        )
                    try:
                        collection.add(
                            ids=batch_ids,
                            documents=batch_docs,
                            metadatas=batch_metadatas,  # type: ignore[arg-type]  # chromadb stub Metadata vs dict[str, Any]
                        )
                    except ValueError as ve:
                        raise _ChromaPayloadRejection(str(ve)) from ve

                all_write_ids.extend(write_ids)
                all_write_documents.extend(write_documents)
                if write_metadatas is not None:
                    all_write_metadatas.extend(write_metadatas)
                total_rows_written += rows_written

            latency_ms = (time.perf_counter() - start_time) * 1000
        except (chromadb.errors.ChromaError, DuplicateDocumentError, _ChromaPayloadRejection) as write_exc:
            latency_ms = (time.perf_counter() - start_time) * 1000
            try:
                ctx.record_call(
                    call_type=CallType.VECTOR,
                    status=CallStatus.ERROR,
                    request_data={
                        "operation": self._config.on_duplicate.upper(),
                        "collection": self._config.collection,
                        "row_count": len(rows),
                    },
                    error={
                        "type": type(write_exc).__name__,
                        "message": str(write_exc),
                    },
                    latency_ms=latency_ms,
                    provider="chromadb",
                )
            except Exception as audit_exc:
                raise AuditIntegrityError(
                    f"Failed to record failed ChromaDB write to audit trail "
                    f"(collection={self._config.collection!r}, original_error={type(write_exc).__name__}). "
                    f"Write failed AND audit record is missing."
                ) from audit_exc
            raise

        # Hash the actual payload sent, not the full batch (critical for skip mode).
        # When some rows had metadata and some didn't, the hash covers the metadatas
        # that were actually sent (or None if none were).
        hash_metadatas: list[dict[str, Any]] | None = all_write_metadatas if all_write_metadatas else None
        content_hash, payload_size = self._compute_payload_hash(all_write_ids, all_write_documents, hash_metadatas)

        diversions = self._get_diversions()

        try:
            response_data: dict[str, Any] = {"rows_written": total_rows_written}
            if total_rows_skipped > 0:
                response_data["rows_skipped"] = total_rows_skipped
                response_data["skipped_ids"] = all_skipped_ids

            request_data: dict[str, Any] = {
                "operation": self._config.on_duplicate.upper(),
                "collection": self._config.collection,
                "row_count": total_rows_written,
                "document_ids": all_write_ids,
            }
            if total_rows_skipped > 0 or diversions:
                request_data["batch_size"] = len(rows)

            ctx.record_call(
                call_type=CallType.VECTOR,
                status=CallStatus.SUCCESS,
                request_data=request_data,
                response_data=response_data,
                latency_ms=latency_ms,
                provider="chromadb",
            )
        except Exception as exc:
            raise AuditIntegrityError(
                f"Failed to record successful ChromaDB write to audit trail "
                f"(collection={self._config.collection!r}, row_count={total_rows_written}). "
                f"Write completed but audit record is missing."
            ) from exc

        self._total_written += total_rows_written
        self._total_bytes += payload_size

        return SinkWriteResult(
            artifact=ArtifactDescriptor.for_database(
                url=chroma_url,
                table=self._config.collection,
                content_hash=content_hash,
                payload_size=payload_size,
                row_count=total_rows_written,
            ),
            diversions=diversions,
        )

    def flush(self) -> None:
        """ChromaDB writes are synchronous in write() — no pending data to flush."""

    def on_complete(self, ctx: LifecycleContext) -> None:
        if self._telemetry_emit is not None:
            self._telemetry_emit(
                {
                    "event": "chroma_sink_complete",
                    "collection": self._config.collection,
                    "total_written": self._total_written,
                    "total_bytes": self._total_bytes,
                }
            )

    def close(self) -> None:
        if self._client is not None:
            self._client.clear_system_cache()
        self._client = None
        self._collection = None
