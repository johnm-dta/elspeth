"""BlobServiceImpl — filesystem-backed blob persistence."""

from __future__ import annotations

import hashlib
import hmac
import os
import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict, TypeVar, cast
from uuid import UUID, uuid4, uuid5

from opentelemetry import metrics
from sqlalchemy import Engine, func, select
from sqlalchemy.engine import Connection, Row
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from elspeth.contracts.enums import CreationModality
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.hashing import canonical_json
from elspeth.web.async_workers import run_sync_in_worker
from elspeth.web.blobs.protocol import (
    ALLOWED_MIME_TYPES,
    BLOB_CREATORS,
    BLOB_RUN_LINK_DIRECTIONS,
    BLOB_STATUSES,
    FINALIZE_BLOB_STATUSES,
    AllowedMimeType,
    BlobActiveRunError,
    BlobContentMissingError,
    BlobCreator,
    BlobFinalizationError,
    BlobFinalizationResult,
    BlobIntegrityError,
    BlobNotFoundError,
    BlobPendingProposalError,
    BlobQuotaExceededError,
    BlobRecord,
    BlobRunLinkDirection,
    BlobRunLinkRecord,
    BlobStateError,
    FinalizeBlobStatus,
    InlineCustodyRequest,
)
from elspeth.web.sessions.converters import pipeline_dict_from_record
from elspeth.web.sessions.locking import (
    acquire_session_advisory_xact_lock,
    locked_session_transaction,
    postgres_session_advisory_lock,
    sqlite_process_session_lock,
)
from elspeth.web.sessions.models import (
    blob_run_links_table,
    blobs_table,
    composition_states_table,
    runs_table,
    sessions_table,
)
from elspeth.web.sessions.proposal_blob_refs import pending_proposal_reference_id
from elspeth.web.sessions.protocol import CompositionStateRecord

_T = TypeVar("_T")

_BLOB_COPY_FORK_ORPHAN_ROWS_COUNTER = metrics.get_meter(__name__).create_counter("blob_copy_fork.orphan_rows_left_behind")

_INLINE_CUSTODY_NAMESPACE = UUID("8ef5fd65-8a90-5fe4-9084-eab5b9d2d2db")
_INLINE_CUSTODY_SCHEMA = "elspeth.inline-custody.v1"
_LOWERCASE_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class _NormalizedInlineCustodyFields(TypedDict):
    session_id: str
    filename: str
    mime_type: AllowedMimeType
    source_description: str | None
    creation_modality: CreationModality
    created_from_message_id: str
    creating_model_identifier: str | None
    creating_model_version: str | None
    creating_provider: str | None
    creating_composer_skill_hash: str | None
    creating_arguments_hash: str | None
    content_hash: str
    size_bytes: int


class _ExpectedBlobFields(TypedDict):
    session_id: str
    filename: str
    mime_type: AllowedMimeType
    source_description: str | None
    creation_modality: CreationModality
    created_from_message_id: str | None
    creating_model_identifier: str | None
    creating_model_version: str | None
    creating_provider: str | None
    creating_composer_skill_hash: str | None
    creating_arguments_hash: str | None
    content_hash: str
    size_bytes: int
    created_by: BlobCreator


_ACTIVE_RUN_COMPOSITION_COLUMNS = (
    runs_table.c.id.label("run_id"),
    composition_states_table.c.id.label("state_id"),
    composition_states_table.c.session_id.label("state_session_id"),
    composition_states_table.c.version.label("state_version"),
    composition_states_table.c.source,
    composition_states_table.c.nodes,
    composition_states_table.c.edges,
    composition_states_table.c.outputs,
    composition_states_table.c.metadata_,
    composition_states_table.c.is_valid,
    composition_states_table.c.validation_errors,
    composition_states_table.c.created_at,
    composition_states_table.c.derived_from_state_id,
    composition_states_table.c.composer_meta,
)


def _uuid_from_db(value: Any) -> UUID:
    return UUID(str(value))


def _active_run_pipeline_dict(active_run: Any) -> dict[str, Any]:
    """Convert an active-run join row to canonical runtime/YAML shape."""
    return pipeline_dict_from_record(
        CompositionStateRecord(
            id=_uuid_from_db(active_run.state_id),
            session_id=_uuid_from_db(active_run.state_session_id),
            version=active_run.state_version,
            source=active_run.source,
            nodes=active_run.nodes,
            edges=active_run.edges,
            outputs=active_run.outputs,
            metadata_=active_run.metadata_,
            is_valid=bool(active_run.is_valid),
            validation_errors=active_run.validation_errors,
            created_at=active_run.created_at,
            derived_from_state_id=_uuid_from_db(active_run.derived_from_state_id) if active_run.derived_from_state_id is not None else None,
            composer_meta=active_run.composer_meta,
        )
    )


def content_hash(data: bytes) -> str:
    """Compute SHA-256 hex digest of raw content bytes.

    This is the shared hash helper referenced by AD-5 and AD-7 in
    docs/plans/rc4.2-ux-remediation/2026-03-30-02-blob-manager-subplan.md.
    When a pipeline reads from a blob, the engine records the raw data
    hash in PayloadStore. Using the same algorithm here guarantees the
    hashes match when the bytes match. Output is SHA-256 hex, 64
    lowercase characters — the canonical form validated by
    ``_validate_finalize_hash`` at the write side and compared via
    ``hmac.compare_digest`` at the read side.
    """
    return hashlib.sha256(data).hexdigest()


def sanitize_filename(filename: str) -> str:
    """Extract a safe basename from a potentially malicious filename.

    Strips all directory components (path traversal protection) and
    rejects empty results or dot-only names.
    """
    sanitized = Path(filename).name
    if not sanitized or sanitized in (".", ".."):
        raise ValueError(f"Invalid filename: {filename!r}")
    # Cap length to leave room for UUID prefix in storage path
    if len(sanitized.encode("utf-8")) > 200:
        # Preserve the extension
        stem = Path(sanitized).stem
        suffix = Path(sanitized).suffix
        max_stem = 200 - len(suffix.encode("utf-8"))
        sanitized = stem.encode("utf-8")[:max_stem].decode("utf-8", errors="ignore") + suffix
    return sanitized


def _normalized_optional_text(value: str | None, *, field_name: str) -> str | None:
    if value is None:
        return None
    if type(value) is not str:
        raise TypeError(f"{field_name} must be str or None, got {type(value).__name__}")
    normalized = value.strip()
    return normalized if normalized else None


def _normalized_optional_sha256(value: str | None, *, field_name: str) -> str | None:
    if value is None:
        return None
    if type(value) is not str:
        raise TypeError(f"{field_name} must be str or None, got {type(value).__name__}")
    if _LOWERCASE_SHA256.fullmatch(value) is None:
        raise AuditIntegrityError(f"{field_name} must be an exact lowercase SHA-256 digest")
    return value


def _normalized_inline_custody_fields(request: InlineCustodyRequest) -> _NormalizedInlineCustodyFields:
    """Validate and normalize the identity-bearing custody fields."""
    if type(request.content) is not bytes:
        raise TypeError(f"InlineCustodyRequest.content must be bytes, got {type(request.content).__name__}")
    if type(request.session_id) is not UUID:
        raise TypeError(f"InlineCustodyRequest.session_id must be UUID, got {type(request.session_id).__name__}")
    if type(request.filename) is not str:
        raise TypeError(f"InlineCustodyRequest.filename must be str, got {type(request.filename).__name__}")
    filename = sanitize_filename(request.filename)
    untrusted_mime_type: object = request.mime_type
    if type(untrusted_mime_type) is not str:
        raise TypeError(f"InlineCustodyRequest.mime_type must be str, got {type(untrusted_mime_type).__name__}")
    mime_type_value = untrusted_mime_type.strip().lower()
    if mime_type_value not in ALLOWED_MIME_TYPES:
        raise RuntimeError(f"Invalid mime_type {mime_type_value!r} — not in the allowed MIME set")
    mime_type = cast(AllowedMimeType, mime_type_value)
    if type(request.creation_modality) is not CreationModality:
        raise TypeError(f"InlineCustodyRequest.creation_modality must be CreationModality, got {type(request.creation_modality).__name__}")
    message_id = _normalized_optional_text(request.created_from_message_id, field_name="created_from_message_id")
    if message_id is None:
        raise AuditIntegrityError("Inline custody requires a non-blank originating message id")
    description = _normalized_optional_text(request.source_description, field_name="source_description")
    model_identifier = _normalized_optional_text(request.creating_model_identifier, field_name="creating_model_identifier")
    model_version = _normalized_optional_text(request.creating_model_version, field_name="creating_model_version")
    provider = _normalized_optional_text(request.creating_provider, field_name="creating_provider")
    skill_hash = _normalized_optional_sha256(
        request.creating_composer_skill_hash,
        field_name="creating_composer_skill_hash",
    )
    arguments_hash = _normalized_optional_sha256(
        request.creating_arguments_hash,
        field_name="creating_arguments_hash",
    )
    llm_fields = (model_identifier, model_version, provider, skill_hash, arguments_hash)
    if request.creation_modality.requires_llm_provenance():
        if any(value is None for value in llm_fields):
            raise AuditIntegrityError("LLM-authored inline custody requires complete composer provenance")
    elif any(value is not None for value in llm_fields):
        raise AuditIntegrityError("Verbatim inline custody must not carry LLM composer provenance")
    return {
        "session_id": str(request.session_id),
        "filename": filename,
        "mime_type": mime_type,
        "source_description": description,
        "creation_modality": request.creation_modality,
        "created_from_message_id": message_id,
        "creating_model_identifier": model_identifier,
        "creating_model_version": model_version,
        "creating_provider": provider,
        "creating_composer_skill_hash": skill_hash,
        "creating_arguments_hash": arguments_hash,
        "content_hash": content_hash(request.content),
        "size_bytes": len(request.content),
    }


def inline_custody_blob_id(request: InlineCustodyRequest) -> UUID:
    """Return the domain-separated deterministic UUID5 for a custody request."""
    fields = _normalized_inline_custody_fields(request)
    identity = {
        "schema": _INLINE_CUSTODY_SCHEMA,
        "session_id": fields["session_id"],
        "originating_message_id": fields["created_from_message_id"],
        "filename": fields["filename"],
        "mime_type": fields["mime_type"],
        "description": fields["source_description"],
        "content_hash": fields["content_hash"],
        "creation_provenance": {
            "modality": fields["creation_modality"].value,
            "model_identifier": fields["creating_model_identifier"],
            "model_version": fields["creating_model_version"],
            "provider": fields["creating_provider"],
            "composer_skill_hash": fields["creating_composer_skill_hash"],
        },
    }
    return uuid5(_INLINE_CUSTODY_NAMESPACE, canonical_json(identity))


def _atomic_write_blob(storage: Path, content: bytes) -> None:
    storage.parent.mkdir(parents=True, exist_ok=True)
    _remove_blob_temp_artifacts(storage)
    temp_path = storage.with_name(f".{storage.name}.custody.tmp")
    fd = os.open(temp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, storage)
        _fsync_parent_directory(storage.parent)
    finally:
        temp_path.unlink(missing_ok=True)


def _fsync_parent_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _blob_temp_artifacts(storage: Path) -> tuple[Path, ...]:
    deterministic = storage.with_name(f".{storage.name}.custody.tmp")
    legacy = tuple(storage.parent.glob(f".{storage.name}.*.tmp")) if storage.parent.exists() else ()
    return tuple(dict.fromkeys((deterministic, *legacy)))


def _remove_blob_temp_artifacts(storage: Path) -> None:
    for path in _blob_temp_artifacts(storage):
        path.unlink(missing_ok=True)


def _validate_reusable_blob_row(
    row: Any,
    *,
    expected: _ExpectedBlobFields,
    blob_id: str,
    storage_path: Path,
) -> None:
    expected_fields = {
        "session_id": expected["session_id"],
        "filename": expected["filename"],
        "mime_type": expected["mime_type"],
        "size_bytes": expected["size_bytes"],
        "content_hash": expected["content_hash"],
        "storage_path": str(storage_path),
        "created_by": expected["created_by"],
        "source_description": expected["source_description"],
        "creation_modality": expected["creation_modality"].value,
        "created_from_message_id": expected["created_from_message_id"],
        "creating_model_identifier": expected["creating_model_identifier"],
        "creating_model_version": expected["creating_model_version"],
        "creating_provider": expected["creating_provider"],
        "creating_composer_skill_hash": expected["creating_composer_skill_hash"],
        "creating_arguments_hash": expected["creating_arguments_hash"],
    }
    for field_name, expected_value in expected_fields.items():
        if getattr(row, field_name) != expected_value:
            raise AuditIntegrityError(f"Inline custody blob {blob_id} has mismatched {field_name}")
    if row.status not in {"pending", "ready"}:
        raise AuditIntegrityError(f"Inline custody blob {blob_id} has invalid reuse status {row.status!r}")


@contextmanager
def _blob_phase_transaction(engine: Engine, held_connection: Connection | None) -> Iterator[Connection]:
    if held_connection is None:
        with engine.begin() as conn:
            yield conn
        return
    with held_connection.begin():
        yield held_connection


@contextmanager
def _blob_custody_session_lock(engine: Engine, session_id: str) -> Iterator[Connection | None]:
    dialect = engine.dialect.name
    if dialect == "sqlite":
        with sqlite_process_session_lock(engine, session_id):
            yield None
        return
    if dialect == "postgresql":
        with engine.connect() as conn, postgres_session_advisory_lock(conn, session_id):
            yield conn
        return
    raise NotImplementedError(f"Blob custody locking is not implemented for dialect {dialect}")


def _acquire_blob_phase_lock(conn: Connection, session_id: str) -> None:
    if conn.dialect.name == "postgresql":
        acquire_session_advisory_xact_lock(conn, session_id)


def _reserve_pending_blob(
    *,
    engine: Engine,
    held_connection: Connection | None,
    blob_id: str,
    storage: Path,
    expected: _ExpectedBlobFields,
    max_storage_per_session: int,
    idempotent: bool,
) -> tuple[Row[Any], bool]:
    session_id = expected["session_id"]
    with _blob_phase_transaction(engine, held_connection) as conn:
        _acquire_blob_phase_lock(conn, session_id)
        _lock_session_for_blob_quota(conn, session_id)
        row = conn.execute(select(blobs_table).where(blobs_table.c.id == blob_id)).first()
        if row is None:
            current_total = conn.execute(
                select(func.coalesce(func.sum(blobs_table.c.size_bytes), 0)).where(blobs_table.c.session_id == session_id)
            ).scalar()
            if type(current_total) is not int:
                raise AuditIntegrityError(f"Tier 1: COALESCE(SUM) returned {type(current_total).__name__}, expected int")
            if current_total + expected["size_bytes"] > max_storage_per_session:
                raise BlobQuotaExceededError(
                    session_id,
                    current_bytes=current_total,
                    limit_bytes=max_storage_per_session,
                )
            try:
                with conn.begin_nested():
                    conn.execute(
                        blobs_table.insert().values(
                            id=blob_id,
                            session_id=session_id,
                            filename=expected["filename"],
                            mime_type=expected["mime_type"],
                            size_bytes=expected["size_bytes"],
                            content_hash=expected["content_hash"],
                            storage_path=str(storage),
                            created_at=datetime.now(UTC),
                            created_by=expected["created_by"],
                            source_description=expected["source_description"],
                            status="pending",
                            creation_modality=expected["creation_modality"].value,
                            created_from_message_id=expected["created_from_message_id"],
                            creating_model_identifier=expected["creating_model_identifier"],
                            creating_model_version=expected["creating_model_version"],
                            creating_provider=expected["creating_provider"],
                            creating_composer_skill_hash=expected["creating_composer_skill_hash"],
                            creating_arguments_hash=expected["creating_arguments_hash"],
                        )
                    )
            except IntegrityError as exc:
                row = conn.execute(select(blobs_table).where(blobs_table.c.id == blob_id)).first()
                if row is None:
                    raise
                if not idempotent:
                    raise AuditIntegrityError(f"Unexpected duplicate blob id {blob_id}") from exc
                created_reservation = False
            else:
                created_reservation = True
                row = conn.execute(select(blobs_table).where(blobs_table.c.id == blob_id)).one()
        elif not idempotent:
            raise AuditIntegrityError(f"Unexpected duplicate blob id {blob_id}")
        else:
            created_reservation = False
        _validate_reusable_blob_row(row, expected=expected, blob_id=blob_id, storage_path=storage)
        return row, created_reservation


def _write_or_validate_reserved_blob(
    *,
    row: Row[Any],
    storage: Path,
    content: bytes,
    expected_hash: str,
    blob_id: str,
) -> bool:
    if storage.exists():
        existing_content = storage.read_bytes()
        actual_hash = content_hash(existing_content)
        if not hmac.compare_digest(existing_content, content) or not hmac.compare_digest(actual_hash, expected_hash):
            raise BlobIntegrityError(blob_id, expected=expected_hash, actual=actual_hash)
        return False
    if row.status == "ready":
        raise BlobContentMissingError(blob_id, storage_path=str(storage))
    _atomic_write_blob(storage, content)
    return True


def _finalize_reserved_blob(
    *,
    engine: Engine,
    held_connection: Connection | None,
    blob_id: str,
    storage: Path,
    expected: _ExpectedBlobFields,
) -> Row[Any]:
    session_id = expected["session_id"]
    with _blob_phase_transaction(engine, held_connection) as conn:
        _acquire_blob_phase_lock(conn, session_id)
        row = conn.execute(select(blobs_table).where(blobs_table.c.id == blob_id)).one()
        _validate_reusable_blob_row(row, expected=expected, blob_id=blob_id, storage_path=storage)
        if row.status == "pending":
            conn.execute(blobs_table.update().where(blobs_table.c.id == blob_id).values(status="ready"))
        final_row = conn.execute(select(blobs_table).where(blobs_table.c.id == blob_id)).one()
        _guard_blob_row_literals(final_row)
        return final_row


def _discard_nonidempotent_reservation(
    *,
    engine: Engine,
    held_connection: Connection | None,
    blob_id: str,
    session_id: str,
    storage: Path,
    remove_storage: bool,
) -> None:
    if remove_storage:
        storage.unlink(missing_ok=True)
    with _blob_phase_transaction(engine, held_connection) as conn:
        _acquire_blob_phase_lock(conn, session_id)
        conn.execute(blobs_table.delete().where(blobs_table.c.id == blob_id).where(blobs_table.c.status == "pending"))


def _persist_blob_content(
    *,
    engine: Engine,
    data_dir: Path,
    max_storage_per_session: int,
    blob_id: UUID,
    session_id: UUID | str,
    filename: str,
    content: bytes,
    mime_type: AllowedMimeType,
    created_by: BlobCreator,
    source_description: str | None,
    creation_modality: CreationModality,
    created_from_message_id: str | None,
    creating_model_identifier: str | None,
    creating_model_version: str | None,
    creating_provider: str | None,
    creating_composer_skill_hash: str | None,
    creating_arguments_hash: str | None,
    idempotent: bool,
) -> Row[Any]:
    """Persist one blob through committed reservation, file, and ready phases."""
    if type(blob_id) is not UUID:
        raise TypeError(f"blob_id must be UUID, got {type(blob_id).__name__}")
    if type(session_id) not in {UUID, str}:
        raise TypeError(f"session_id must be UUID or str, got {type(session_id).__name__}")
    if type(filename) is not str:
        raise TypeError(f"filename must be str, got {type(filename).__name__}")
    if type(content) is not bytes:
        raise TypeError(f"Blob content must be bytes, got {type(content).__name__}")
    untrusted_mime_type: object = mime_type
    if type(untrusted_mime_type) is not str:
        raise TypeError(f"mime_type must be str, got {type(untrusted_mime_type).__name__}")
    if untrusted_mime_type not in ALLOWED_MIME_TYPES:
        raise RuntimeError(f"Invalid mime_type {untrusted_mime_type!r} — not in the allowed MIME set")
    untrusted_created_by: object = created_by
    if type(untrusted_created_by) is not str:
        raise TypeError(f"created_by must be str, got {type(untrusted_created_by).__name__}")
    if untrusted_created_by not in BLOB_CREATORS:
        raise RuntimeError(f"Invalid created_by {untrusted_created_by!r} — must be one of {sorted(BLOB_CREATORS)}")
    if type(creation_modality) is not CreationModality:
        raise TypeError(f"creation_modality must be CreationModality, got {type(creation_modality).__name__}")
    source_description = _normalized_optional_text(source_description, field_name="source_description")
    created_from_message_id = _normalized_optional_text(created_from_message_id, field_name="created_from_message_id")
    creating_model_identifier = _normalized_optional_text(
        creating_model_identifier,
        field_name="creating_model_identifier",
    )
    creating_model_version = _normalized_optional_text(creating_model_version, field_name="creating_model_version")
    creating_provider = _normalized_optional_text(creating_provider, field_name="creating_provider")
    creating_composer_skill_hash = _normalized_optional_sha256(
        creating_composer_skill_hash,
        field_name="creating_composer_skill_hash",
    )
    creating_arguments_hash = _normalized_optional_sha256(
        creating_arguments_hash,
        field_name="creating_arguments_hash",
    )
    llm_provenance = (
        creating_model_identifier,
        creating_model_version,
        creating_provider,
        creating_composer_skill_hash,
        creating_arguments_hash,
    )
    if creation_modality.requires_llm_provenance():
        if created_from_message_id is None or any(value is None for value in llm_provenance):
            raise AuditIntegrityError("LLM-authored blob persistence requires complete composer provenance")
    elif any(value is not None for value in llm_provenance):
        raise AuditIntegrityError("Verbatim blob persistence must not carry LLM composer provenance")
    session_id_str = str(session_id)
    if not session_id_str or Path(session_id_str).name != session_id_str or session_id_str in {".", ".."}:
        raise AuditIntegrityError("session_id must be a non-empty opaque path segment")
    blob_id_str = str(blob_id)
    safe_filename = sanitize_filename(filename)
    storage = data_dir.expanduser().resolve() / "blobs" / session_id_str / f"{blob_id_str}_{safe_filename}"
    expected: _ExpectedBlobFields = {
        "session_id": session_id_str,
        "filename": safe_filename,
        "mime_type": mime_type,
        "size_bytes": len(content),
        "content_hash": content_hash(content),
        "created_by": created_by,
        "source_description": source_description,
        "creation_modality": creation_modality,
        "created_from_message_id": created_from_message_id,
        "creating_model_identifier": creating_model_identifier,
        "creating_model_version": creating_model_version,
        "creating_provider": creating_provider,
        "creating_composer_skill_hash": creating_composer_skill_hash,
        "creating_arguments_hash": creating_arguments_hash,
    }
    with _blob_custody_session_lock(engine, session_id_str) as held_connection:
        created_reservation = False
        created_storage = False
        try:
            row, created_reservation = _reserve_pending_blob(
                engine=engine,
                held_connection=held_connection,
                blob_id=blob_id_str,
                storage=storage,
                expected=expected,
                max_storage_per_session=max_storage_per_session,
                idempotent=idempotent,
            )
            storage_existed_before_write = storage.exists()
            try:
                created_storage = _write_or_validate_reserved_blob(
                    row=row,
                    storage=storage,
                    content=content,
                    expected_hash=expected["content_hash"],
                    blob_id=blob_id_str,
                )
            except Exception:
                created_storage = not storage_existed_before_write and storage.exists()
                raise
            return _finalize_reserved_blob(
                engine=engine,
                held_connection=held_connection,
                blob_id=blob_id_str,
                storage=storage,
                expected=expected,
            )
        except Exception:
            if not idempotent and created_reservation:
                _discard_nonidempotent_reservation(
                    engine=engine,
                    held_connection=held_connection,
                    blob_id=blob_id_str,
                    session_id=session_id_str,
                    storage=storage,
                    remove_storage=created_storage,
                )
            raise


def _option_value_references_blob(value: Any, blob_id: str, storage_path: str) -> bool:
    """Recursively inspect an option value for blob identity markers."""
    if type(value) is dict:
        if "blob_ref" in value and value["blob_ref"] == blob_id:
            return True
        if any(key in value and value[key] == storage_path for key in ("path", "file")):
            return True
        return any(_option_value_references_blob(child, blob_id, storage_path) for child in value.values())
    if type(value) is list:
        return any(_option_value_references_blob(child, blob_id, storage_path) for child in value)
    return False


def _options_reference_blob(options: Any, blob_id: str, storage_path: str, owner: str) -> bool:
    if options is None:
        return False
    if type(options) is not dict:
        raise AuditIntegrityError(f"Tier 1: composition_states.{owner}.options is {type(options).__name__}, expected dict")
    return _option_value_references_blob(options, blob_id, storage_path)


def _composition_references_blob(
    composition_state: Any,
    blob_id: str,
    storage_path: str,
) -> bool:
    """Check whether any runtime/YAML-shape composition section references a blob.

    ``composition_state`` must be the canonical pipeline dict emitted by
    ``generate_pipeline_dict()`` or ``pipeline_dict_from_record()``. It walks
    source options, node-collection options, and sink options for either a
    matching ``blob_ref`` marker or a path/file value matching ``storage_path``.

    Tier 1 guards: malformed present sections are DB/audit corruption, so they
    raise ``AuditIntegrityError`` instead of becoming silent false negatives.
    """
    if composition_state is None:
        return False
    if type(composition_state) is not dict:
        raise AuditIntegrityError(f"Tier 1: composition_states is {type(composition_state).__name__}, expected dict")

    if "sources" in composition_state:
        sources = composition_state["sources"]
        # Per ADR-025 §1, the canonical pipeline dict emits `sources` as a
        # non-null dict whenever any source is present. A null `sources` map
        # in a persisted composition_state is internal corruption (Tier 1).
        if sources is None:
            raise AuditIntegrityError("Tier 1: composition_states.sources is null, expected dict")
        if type(sources) is not dict:
            raise AuditIntegrityError(f"Tier 1: composition_states.sources is {type(sources).__name__}, expected dict")
        for source_name, source in sources.items():
            if source is None:
                raise AuditIntegrityError(f"Tier 1: composition_states.sources[{source_name!r}] is null, expected dict")
            if type(source) is not dict:
                raise AuditIntegrityError(f"Tier 1: composition_states.sources[{source_name!r}] is {type(source).__name__}, expected dict")
            source_options = source["options"] if "options" in source else None
            if _options_reference_blob(source_options, blob_id, storage_path, f"sources[{source_name!r}]"):
                return True

    for collection_key in ("transforms", "gates", "aggregations", "coalesce"):
        if collection_key not in composition_state:
            continue
        nodes = composition_state[collection_key]
        if nodes is None:
            continue
        if type(nodes) is not list:
            raise AuditIntegrityError(f"Tier 1: composition_states.{collection_key} is {type(nodes).__name__}, expected list")
        for index, node in enumerate(nodes):
            if type(node) is not dict:
                raise AuditIntegrityError(f"Tier 1: composition_states.{collection_key}[{index}] is {type(node).__name__}, expected dict")
            node_options = node["options"] if "options" in node else None
            if _options_reference_blob(node_options, blob_id, storage_path, f"{collection_key}[{index}]"):
                return True

    if "sinks" not in composition_state:
        return False
    sinks = composition_state["sinks"]
    if sinks is None:
        return False
    if type(sinks) is not dict:
        raise AuditIntegrityError(f"Tier 1: composition_states.sinks is {type(sinks).__name__}, expected dict")
    for sink_name, sink in sinks.items():
        if type(sink) is not dict:
            raise AuditIntegrityError(f"Tier 1: composition_states.sinks[{sink_name!r}] is {type(sink).__name__}, expected dict")
        sink_options = sink["options"] if "options" in sink else None
        if _options_reference_blob(sink_options, blob_id, storage_path, f"sinks[{sink_name!r}]"):
            return True
    return False


def _assert_blob_run_same_session(
    conn: Connection,
    *,
    blob_id: str,
    run_id: str,
    caller: str,
) -> None:
    """Offensive guard: blob and run must belong to the same session.

    ``link_blob_to_run()`` is an internal write boundary. A cross-session
    linkage is a caller bug, not user input, so crash with RuntimeError
    before persisting contradictory ownership into ``blob_run_links``.
    """
    blob_session_id = conn.execute(select(blobs_table.c.session_id).where(blobs_table.c.id == blob_id)).scalar()
    if blob_session_id is None:
        raise RuntimeError(f"{caller}: blob_id={blob_id!r} does not exist")

    run_session_id = conn.execute(select(runs_table.c.session_id).where(runs_table.c.id == run_id)).scalar()
    if run_session_id is None:
        raise RuntimeError(f"{caller}: run_id={run_id!r} does not exist")

    if blob_session_id != run_session_id:
        raise RuntimeError(
            f"{caller}: blob_id={blob_id!r} belongs to session "
            f"{blob_session_id!r}, run_id={run_id!r} belongs to session "
            f"{run_session_id!r} — cross-session reference is a contract violation"
        )


def _session_quota_lock_statement(session_id_str: str) -> Any:
    """Build the per-session row lock used to serialize quota writers."""
    return select(sessions_table.c.id).where(sessions_table.c.id == session_id_str).with_for_update()


def _lock_session_for_blob_quota(conn: Connection, session_id_str: str) -> None:
    """Lock the owning session row before a quota read/write sequence.

    On PostgreSQL this emits ``SELECT ... FOR UPDATE`` and serializes all
    same-session blob quota writers. SQLite ignores the row-lock clause, but
    its coarse write serialization already preserves the current behavior.
    """
    locked = conn.execute(_session_quota_lock_statement(session_id_str)).first()
    if locked is None:
        raise RuntimeError(f"Blob quota lock target session {session_id_str!r} does not exist")


def _guard_blob_row_literals(row: Any) -> None:
    """Validate closed-set blob row fields at the DB read boundary."""
    # Tier 1 read guards — BlobRecord's fields are declared as closed
    # Literal types, but the DB can be tampered with via direct SQL
    # or a migration bug. Crash on any value outside the enum so the
    # audit trail never silently returns a record whose static type
    # is a lie. Aligns with the frozenset CHECK constraints in
    # web/sessions/models.py (ck_blobs_status, ck_blobs_created_by)
    # and the MIME allowlist enforced at create_blob().
    #
    # Explicit raise (not ``assert``): ``python -O`` strips asserts,
    # so an optimised interpreter would silently pass a tampered row
    # through these guards. AuditIntegrityError is the contract for
    # Tier 1 DB-corruption conditions and survives ``-O`` execution.
    if row.status not in BLOB_STATUSES:
        raise AuditIntegrityError(f"Tier 1: blobs.status is {row.status!r}, expected one of {sorted(BLOB_STATUSES)}")
    if row.created_by not in BLOB_CREATORS:
        raise AuditIntegrityError(f"Tier 1: blobs.created_by is {row.created_by!r}, expected one of {sorted(BLOB_CREATORS)}")
    if row.mime_type not in ALLOWED_MIME_TYPES:
        raise AuditIntegrityError(f"Tier 1: blobs.mime_type is {row.mime_type!r}, not in the allowed MIME set")
    # Tier 1 guard for the closed CreationModality enum (Phase 5a Task 2.5).
    # Mirrors the ck_blobs_creation_modality DB CHECK; this Python guard
    # catches tampered or migration-bug-introduced rows that bypassed the
    # DB layer (e.g. a manual SQLite UPDATE).  AuditIntegrityError keeps
    # the audit-trail correctness invariant — the read path never silently
    # returns a record whose static type is a lie.
    if row.creation_modality not in {m.value for m in CreationModality}:
        raise AuditIntegrityError(
            f"Tier 1: blobs.creation_modality is {row.creation_modality!r}, expected one of {sorted(m.value for m in CreationModality)}"
        )


def _row_to_blob_record(row: Any) -> BlobRecord:
    """Convert a blobs row into a guarded BlobRecord."""
    _guard_blob_row_literals(row)
    return BlobRecord(
        id=UUID(row.id),
        session_id=UUID(row.session_id),
        filename=row.filename,
        mime_type=row.mime_type,
        size_bytes=row.size_bytes,
        content_hash=row.content_hash,
        storage_path=row.storage_path,
        created_at=row.created_at,
        created_by=row.created_by,
        source_description=row.source_description,
        status=row.status,
        # Tier 1 read: ``creation_modality`` has already been checked
        # against the closed CreationModality enum in
        # ``_guard_blob_row_literals``; coerce to the enum so consumers
        # get the typed value rather than the bare wire-format string.
        creation_modality=CreationModality(row.creation_modality),
        created_from_message_id=row.created_from_message_id,
        creating_model_identifier=row.creating_model_identifier,
        creating_model_version=row.creating_model_version,
        creating_provider=row.creating_provider,
        creating_composer_skill_hash=row.creating_composer_skill_hash,
        creating_arguments_hash=row.creating_arguments_hash,
    )


class BlobServiceImpl:
    """Filesystem-backed blob service.

    Follows the same async-over-sync pattern as SessionServiceImpl:
    all public methods are async, database I/O runs in a thread pool
    executor via _run_sync().
    """

    def __init__(self, engine: Engine, data_dir: Path, max_storage_per_session: int = 500 * 1024 * 1024) -> None:
        self._engine = engine
        self._data_dir = data_dir.expanduser().resolve()
        self._max_storage_per_session = max_storage_per_session

    async def _run_sync(self, func: Callable[[], _T]) -> _T:
        return await run_sync_in_worker(func)

    def _now(self) -> datetime:
        return datetime.now(UTC)

    def _blob_dir(self, session_id: str) -> Path:
        return self._data_dir / "blobs" / session_id

    def _storage_path(self, session_id: str, blob_id: str, filename: str) -> Path:
        return self._blob_dir(session_id) / f"{blob_id}_{filename}"

    def _row_to_record(self, row: Any) -> BlobRecord:
        return _row_to_blob_record(row)

    def _row_to_link_record(self, row: Any) -> BlobRunLinkRecord:
        # Tier 1 read guard — mirrors the ck_blob_run_links_direction
        # CHECK constraint.  A row with a bogus direction would leave
        # BlobRunLinkRecord.direction (typed BlobRunLinkDirection)
        # carrying a value outside its Literal set.  Explicit raise (not
        # ``assert``) so the guard survives ``python -O``.
        if row.direction not in BLOB_RUN_LINK_DIRECTIONS:
            raise AuditIntegrityError(
                f"Tier 1: blob_run_links.direction is {row.direction!r}, expected one of {sorted(BLOB_RUN_LINK_DIRECTIONS)}"
            )
        return BlobRunLinkRecord(
            blob_id=UUID(row.blob_id),
            run_id=UUID(row.run_id),
            direction=row.direction,
        )

    def _enforce_ready_finalize_quota(
        self,
        conn: Connection,
        *,
        blob_id_str: str,
        session_id_str: str,
        status: FinalizeBlobStatus,
        size_bytes: int | None,
    ) -> None:
        """Enforce session storage quota when a pending blob becomes ready."""
        if status != "ready" or size_bytes is None or size_bytes <= 0:
            return
        _lock_session_for_blob_quota(conn, session_id_str)
        current_total = conn.execute(
            select(func.coalesce(func.sum(blobs_table.c.size_bytes), 0)).where(
                blobs_table.c.session_id == session_id_str,
                blobs_table.c.id != blob_id_str,
            )
        ).scalar()
        # COALESCE guarantees an exact int; bool/subclasses or any other type
        # are Tier 1 anomalies. Explicit raise so the guard survives python -O.
        if type(current_total) is not int:
            raise AuditIntegrityError(f"Tier 1: COALESCE(SUM) returned {type(current_total).__name__}, expected int")
        if current_total + size_bytes > self._max_storage_per_session:
            raise BlobQuotaExceededError(
                session_id_str,
                current_bytes=current_total,
                limit_bytes=self._max_storage_per_session,
            )

    async def create_blob(
        self,
        session_id: UUID,
        filename: str,
        content: bytes,
        mime_type: AllowedMimeType,
        created_by: BlobCreator = "user",
        source_description: str | None = None,
    ) -> BlobRecord:
        """Create a blob from content bytes."""
        if created_by not in BLOB_CREATORS:
            raise RuntimeError(f"Invalid created_by {created_by!r} — must be one of {sorted(BLOB_CREATORS)}")
        if mime_type not in ALLOWED_MIME_TYPES:
            raise RuntimeError(f"Invalid mime_type {mime_type!r} — not in the allowed MIME set")
        blob_id = uuid4()
        row = await self._run_sync(
            lambda: _persist_blob_content(
                engine=self._engine,
                data_dir=self._data_dir,
                max_storage_per_session=self._max_storage_per_session,
                blob_id=blob_id,
                session_id=session_id,
                filename=filename,
                content=content,
                mime_type=mime_type,
                created_by=created_by,
                source_description=source_description,
                creation_modality=CreationModality.VERBATIM,
                created_from_message_id=None,
                creating_model_identifier=None,
                creating_model_version=None,
                creating_provider=None,
                creating_composer_skill_hash=None,
                creating_arguments_hash=None,
                idempotent=False,
            )
        )
        return _row_to_blob_record(row)

    async def reserve_inline_custody(self, request: InlineCustodyRequest) -> BlobRecord:
        """Idempotently materialize one composer inline source."""
        fields = _normalized_inline_custody_fields(request)
        blob_id = inline_custody_blob_id(request)
        row = await self._run_sync(
            lambda: _persist_blob_content(
                engine=self._engine,
                data_dir=self._data_dir,
                max_storage_per_session=self._max_storage_per_session,
                blob_id=blob_id,
                session_id=request.session_id,
                filename=fields["filename"],
                content=request.content,
                mime_type=fields["mime_type"],
                created_by="assistant",
                source_description=fields["source_description"],
                creation_modality=request.creation_modality,
                created_from_message_id=fields["created_from_message_id"],
                creating_model_identifier=fields["creating_model_identifier"],
                creating_model_version=fields["creating_model_version"],
                creating_provider=fields["creating_provider"],
                creating_composer_skill_hash=fields["creating_composer_skill_hash"],
                creating_arguments_hash=fields["creating_arguments_hash"],
                idempotent=True,
            )
        )
        return _row_to_blob_record(row)

    async def create_pending_blob(
        self,
        session_id: UUID,
        filename: str,
        mime_type: AllowedMimeType,
        created_by: BlobCreator = "pipeline",
        source_description: str | None = None,
    ) -> BlobRecord:
        """Reserve a pending output blob."""
        # Programmer-bug guard on Literal-typed parameter.  Explicit raise
        # so the check survives ``python -O`` (mirrors create_blob()).
        if created_by not in BLOB_CREATORS:
            raise RuntimeError(f"Invalid created_by {created_by!r} — must be one of {sorted(BLOB_CREATORS)}")
        if mime_type not in ALLOWED_MIME_TYPES:
            raise RuntimeError(f"Invalid mime_type {mime_type!r} — not in the allowed MIME set")
        safe_filename = sanitize_filename(filename)
        blob_id = str(uuid4())
        session_id_str = str(session_id)
        storage = self._storage_path(session_id_str, blob_id, safe_filename)

        def _sync() -> BlobRecord:
            # Ensure directory exists (file will be written by sink later)
            storage.parent.mkdir(parents=True, exist_ok=True)

            now = self._now()
            with self._engine.begin() as conn:
                conn.execute(
                    blobs_table.insert().values(
                        id=blob_id,
                        session_id=session_id_str,
                        filename=safe_filename,
                        mime_type=mime_type,
                        size_bytes=0,
                        content_hash=None,
                        storage_path=str(storage),
                        created_at=now,
                        created_by=created_by,
                        source_description=source_description,
                        status="pending",
                        # Pending output blobs (pipeline-produced): the
                        # content is filled in later by the sink writer;
                        # provenance for these is the pipeline run, not
                        # a chat message, so the chat-message FK is NULL
                        # and the modality is VERBATIM (the run wrote
                        # exactly the bytes the sink emitted; no LLM
                        # authored them).  Phase 5a Task 2.5.
                        creation_modality=CreationModality.VERBATIM.value,
                        created_from_message_id=None,
                        creating_model_identifier=None,
                        creating_model_version=None,
                        creating_provider=None,
                        creating_composer_skill_hash=None,
                        creating_arguments_hash=None,
                    )
                )

            return BlobRecord(
                id=UUID(blob_id),
                session_id=session_id,
                filename=safe_filename,
                mime_type=mime_type,
                size_bytes=0,
                content_hash=None,
                storage_path=str(storage),
                created_at=now,
                created_by=created_by,
                source_description=source_description,
                status="pending",
                creation_modality=CreationModality.VERBATIM,
                created_from_message_id=None,
                creating_model_identifier=None,
                creating_model_version=None,
                creating_provider=None,
                creating_composer_skill_hash=None,
                creating_arguments_hash=None,
            )

        return await self._run_sync(_sync)

    async def finalize_blob(
        self,
        blob_id: UUID,
        status: FinalizeBlobStatus,
        size_bytes: int | None = None,
        content_hash: str | None = None,
    ) -> BlobRecord:
        """Update a pending blob to ready or error after execution."""
        blob_id_str = str(blob_id)
        # Runtime guard for dynamic callers — the Literal narrowing gives
        # static callers the correct shape, but the Protocol boundary is
        # still called by code that mypy may not fully verify (tests,
        # factory-constructed services).  Keep the check as a belt.
        if status not in FINALIZE_BLOB_STATUSES:
            raise RuntimeError(f"Invalid finalize status '{status}' — must be one of {sorted(FINALIZE_BLOB_STATUSES)}")

        def _sync() -> BlobRecord:
            with self._engine.begin() as conn:
                row = conn.execute(select(blobs_table).where(blobs_table.c.id == blob_id_str)).first()
                if row is None:
                    raise BlobNotFoundError(blob_id_str)

                if row.status != "pending":
                    raise BlobStateError(
                        blob_id_str,
                        message=f"Cannot finalize blob {blob_id_str} — status is '{row.status}', expected 'pending'",
                    )
                # Hash-format validation runs after the state check so a
                # callers confused by a stale blob hear about the lifecycle
                # problem first.  See _validate_finalize_hash() docstring.
                _validate_finalize_hash(blob_id_str, status, content_hash)
                self._enforce_ready_finalize_quota(
                    conn,
                    blob_id_str=blob_id_str,
                    session_id_str=row.session_id,
                    status=status,
                    size_bytes=size_bytes,
                )

                updates: dict[str, Any] = {"status": status}
                if size_bytes is not None:
                    updates["size_bytes"] = size_bytes
                if content_hash is not None:
                    updates["content_hash"] = content_hash

                conn.execute(blobs_table.update().where(blobs_table.c.id == blob_id_str).values(**updates))

                updated = conn.execute(select(blobs_table).where(blobs_table.c.id == blob_id_str)).first()
                if updated is None:
                    raise RuntimeError(f"Blob {blob_id_str} vanished during finalize — concurrent deletion?")
                return self._row_to_record(updated)

        return await self._run_sync(_sync)

    async def get_blob(self, blob_id: UUID) -> BlobRecord:
        """Get blob metadata."""
        blob_id_str = str(blob_id)

        def _sync() -> BlobRecord:
            with self._engine.connect() as conn:
                row = conn.execute(select(blobs_table).where(blobs_table.c.id == blob_id_str)).first()
                if row is None:
                    raise BlobNotFoundError(blob_id_str)
                return self._row_to_record(row)

        return await self._run_sync(_sync)

    async def list_blobs(
        self,
        session_id: UUID,
        limit: int | None = 50,
        offset: int = 0,
    ) -> list[BlobRecord]:
        """List blobs for a session, newest first."""
        session_id_str = str(session_id)

        def _sync() -> list[BlobRecord]:
            with self._engine.connect() as conn:
                rows = conn.execute(
                    select(blobs_table)
                    .where(blobs_table.c.session_id == session_id_str)
                    .order_by(blobs_table.c.created_at.desc())
                    .limit(limit)
                    .offset(offset)
                ).fetchall()
                return [self._row_to_record(r) for r in rows]

        return await self._run_sync(_sync)

    async def delete_blob(self, blob_id: UUID) -> None:
        """Delete blob metadata and backing file."""
        blob_id_str = str(blob_id)

        def _sync() -> None:
            with self._engine.connect() as lookup_conn:
                session_id = lookup_conn.execute(select(blobs_table.c.session_id).where(blobs_table.c.id == blob_id_str)).scalar()
            if session_id is None:
                raise BlobNotFoundError(blob_id_str)
            with locked_session_transaction(self._engine, session_id) as conn:
                # Re-read only after the shared lock: custody/proposal/delete
                # decisions for this session must observe one serial order.
                row = conn.execute(select(blobs_table).where(blobs_table.c.id == blob_id_str)).first()
                if row is None:
                    raise BlobNotFoundError(blob_id_str)

                retaining_proposal_id = pending_proposal_reference_id(
                    conn,
                    session_id=row.session_id,
                    blob_id=blob_id_str,
                )
                if retaining_proposal_id is not None:
                    raise BlobPendingProposalError(blob_id_str, proposal_id=retaining_proposal_id)

                # Active-run guard (two checks):
                #
                # 1. Explicit link: blob_run_links already points at an active run.
                active_link = conn.execute(
                    select(blob_run_links_table)
                    .join(
                        runs_table,
                        blob_run_links_table.c.run_id == runs_table.c.id,
                    )
                    .where(blob_run_links_table.c.blob_id == blob_id_str)
                    .where(runs_table.c.status.in_(["pending", "running"]))
                ).first()
                if active_link is not None:
                    raise BlobActiveRunError(blob_id_str, run_id=active_link.run_id)

                # 2. Pre-link window: _execute_locked() creates the run record
                #    before link_blob_to_run() inserts the blob_run_links row.
                #    During that gap the explicit-link check above sees nothing,
                #    but the backing file is about to be needed.
                #
                #    Scoped to THIS blob: join runs → composition_states and
                #    check whether the active run's canonical pipeline dict
                #    references this blob via blob_ref OR via a path/file that
                #    matches this blob's storage_path. Runs whose state doesn't
                #    touch this blob must not block unrelated blob deletions.
                active_run = conn.execute(
                    select(*_ACTIVE_RUN_COMPOSITION_COLUMNS)
                    .join(
                        composition_states_table,
                        runs_table.c.state_id == composition_states_table.c.id,
                    )
                    .where(runs_table.c.session_id == row.session_id)
                    .where(runs_table.c.status.in_(["pending", "running"]))
                ).first()
                if active_run is not None and _composition_references_blob(
                    _active_run_pipeline_dict(active_run),
                    blob_id_str,
                    row.storage_path,
                ):
                    raise BlobActiveRunError(blob_id_str, run_id=active_run.run_id)

                # Delete backing file first — orphaned DB row is recoverable,
                # orphaned file with no metadata is not
                storage = Path(row.storage_path)
                if storage.exists():
                    storage.unlink()
                _remove_blob_temp_artifacts(storage)

                # Delete metadata (cascades to blob_run_links)
                conn.execute(blobs_table.delete().where(blobs_table.c.id == blob_id_str))

        await self._run_sync(_sync)

    async def read_blob_content(self, blob_id: UUID) -> bytes:
        """Read the raw content of a blob.

        Enforces two invariants before returning bytes:

        1. **Lifecycle guard**: only ``ready`` blobs are readable.
           Pending blobs have no finalized content; error blobs
           represent failed runs whose output is not trustworthy.

        2. **Integrity verification**: a ready blob must still have a
           backing file on disk, and its bytes must match the stored
           ``content_hash``. Missing bytes or hash mismatch indicate
           filesystem corruption, silent data loss, tampering, or a
           write-path bug — all Tier 1 anomalies.
        """
        blob_id_str = str(blob_id)

        def _sync() -> bytes:
            with self._engine.connect() as conn:
                row = conn.execute(select(blobs_table).where(blobs_table.c.id == blob_id_str)).first()
                if row is None:
                    raise BlobNotFoundError(blob_id_str)

                # Lifecycle guard — only ready blobs have finalized content
                if row.status != "ready":
                    raise BlobStateError(
                        blob_id_str,
                        message=f"Cannot read blob {blob_id_str} — status is '{row.status}', expected 'ready'",
                    )

                storage = Path(row.storage_path)
                if not storage.exists():
                    raise BlobContentMissingError(blob_id_str, storage_path=row.storage_path)

                data = storage.read_bytes()

                # Integrity verification — Tier 1: our data must be pristine.
                # A ready blob must always have a content_hash — it is set
                # by create_blob() and required by _finalize_blob_sync()
                # when transitioning to ready.  NULL here is a DB anomaly.
                # Explicit raise so the guard survives ``python -O``.
                if row.content_hash is None:
                    raise AuditIntegrityError(
                        f"Tier 1: ready blob {blob_id_str} has NULL content_hash — DB integrity anomaly, cannot verify"
                    )
                actual = content_hash(data)
                if not hmac.compare_digest(actual, row.content_hash):
                    raise BlobIntegrityError(blob_id_str, expected=row.content_hash, actual=actual)

                return data

        return await self._run_sync(_sync)

    async def read_blob_preview(self, blob_id: UUID, *, limit_bytes: int) -> tuple[bytes, bool]:
        """Read a bounded prefix of a ready blob for inline UI preview.

        This shares the full-content lifecycle/missing-file guards but does
        not verify the full SHA-256 digest, because doing so would require
        reading the whole blob and defeat the preview endpoint's resource cap.
        """
        if limit_bytes < 1:
            raise ValueError("limit_bytes must be >= 1")

        blob_id_str = str(blob_id)

        def _sync() -> tuple[bytes, bool]:
            with self._engine.connect() as conn:
                row = conn.execute(select(blobs_table).where(blobs_table.c.id == blob_id_str)).first()
                if row is None:
                    raise BlobNotFoundError(blob_id_str)

                if row.status != "ready":
                    raise BlobStateError(
                        blob_id_str,
                        message=f"Cannot preview blob {blob_id_str} — status is '{row.status}', expected 'ready'",
                    )

                storage = Path(row.storage_path)
                if not storage.exists():
                    raise BlobContentMissingError(blob_id_str, storage_path=row.storage_path)

                with storage.open("rb") as handle:
                    data = handle.read(limit_bytes + 1)
                return data[:limit_bytes], len(data) > limit_bytes

        return await self._run_sync(_sync)

    async def link_blob_to_run(
        self,
        blob_id: UUID,
        run_id: UUID,
        direction: BlobRunLinkDirection,
    ) -> None:
        """Record a blob-to-run linkage."""
        if direction not in BLOB_RUN_LINK_DIRECTIONS:
            raise RuntimeError(f"Invalid link direction '{direction}' — must be one of {sorted(BLOB_RUN_LINK_DIRECTIONS)}")

        def _sync() -> None:
            with self._engine.begin() as conn:
                _assert_blob_run_same_session(
                    conn,
                    blob_id=str(blob_id),
                    run_id=str(run_id),
                    caller="BlobServiceImpl.link_blob_to_run",
                )
                existing = conn.execute(
                    select(blob_run_links_table.c.blob_id)
                    .where(blob_run_links_table.c.blob_id == str(blob_id))
                    .where(blob_run_links_table.c.run_id == str(run_id))
                    .where(blob_run_links_table.c.direction == direction)
                ).first()
                if existing is not None:
                    return
                conn.execute(
                    blob_run_links_table.insert().values(
                        blob_id=str(blob_id),
                        run_id=str(run_id),
                        direction=direction,
                    )
                )

        await self._run_sync(_sync)

    async def get_blob_run_links(
        self,
        blob_id: UUID,
    ) -> list[BlobRunLinkRecord]:
        """Get all run links for a blob."""
        blob_id_str = str(blob_id)

        def _sync() -> list[BlobRunLinkRecord]:
            with self._engine.connect() as conn:
                rows = conn.execute(select(blob_run_links_table).where(blob_run_links_table.c.blob_id == blob_id_str)).fetchall()
                return [self._row_to_link_record(r) for r in rows]

        return await self._run_sync(_sync)

    # Per-blob operational errors that should not abort the finalization
    # loop.  BlobStateError covers status-guard conditions (blob already
    # finalized by a concurrent call).  RuntimeError is deliberately
    # excluded — it covers the Tier 1 "blob vanished mid-transaction"
    # anomaly, which must propagate.  Programmer bugs (TypeError,
    # AttributeError, AssertionError) also propagate per offensive
    # programming policy.
    _PER_BLOB_SUPPRESSED: tuple[type[BaseException], ...] = (
        BlobNotFoundError,
        BlobStateError,
        OSError,
        SQLAlchemyError,
    )

    async def finalize_run_output_blobs(
        self,
        run_id: UUID,
        success: bool,
    ) -> BlobFinalizationResult:
        """Finalize pending output blobs for a completed/failed run.

        On success: compute content_hash and size_bytes from the backing
        file, set status to 'ready'. If the file wasn't written, mark
        as 'error'.
        On failure: delete the backing file (if any) and set status to
        'error', leaving size/hash as None.  This ensures the filesystem
        matches the DB metadata and prevents orphaned files from escaping
        quota accounting.

        Processes each blob independently — a per-blob operational error
        does not abort finalization of remaining blobs.  Failed blobs are
        transitioned to ``error`` status on a best-effort basis.

        Returns a BlobFinalizationResult with both successfully finalized
        blobs and per-blob error records.
        """
        run_id_str = str(run_id)

        def _sync() -> BlobFinalizationResult:
            with self._engine.connect() as conn:
                rows = conn.execute(
                    select(blobs_table)
                    .join(
                        blob_run_links_table,
                        blob_run_links_table.c.blob_id == blobs_table.c.id,
                    )
                    .where(blob_run_links_table.c.run_id == run_id_str)
                    .where(blob_run_links_table.c.direction == "output")
                    .where(blobs_table.c.status == "pending")
                ).fetchall()

            finalized: list[BlobRecord] = []
            errors: list[BlobFinalizationError] = []
            for row in rows:
                outcome = self._finalize_one_output_blob(UUID(row.id), Path(row.storage_path), success=success)
                if isinstance(outcome, BlobRecord):
                    finalized.append(outcome)
                else:
                    errors.extend(outcome)
            return BlobFinalizationResult(finalized=finalized, errors=errors)

        return await self._run_sync(_sync)

    def _finalize_one_output_blob(
        self,
        blob_id: UUID,
        storage: Path,
        *,
        success: bool,
    ) -> BlobRecord | list[BlobFinalizationError]:
        """Finalize a single output blob, returning an explicit per-blob outcome.

        This is the per-item boundary for ``finalize_run_output_blobs``.
        Filesystem and database faults are genuine I/O boundaries (Tier 3 in
        the web-component sense — the disk and DB are external to our authored
        values).  Rather than swallow such a fault, this method **returns** an
        explicit list of :class:`BlobFinalizationError` records so the batch
        caller can record them in ``BlobFinalizationResult.errors`` and proceed
        to the next blob.  Programmer bugs (TypeError, AttributeError,
        AssertionError) and the Tier 1 "blob vanished mid-transaction"
        RuntimeError are NOT in ``_PER_BLOB_SUPPRESSED`` and so propagate.
        """
        try:
            if success:
                if storage.exists():
                    file_bytes = storage.read_bytes()
                    try:
                        record = self._finalize_blob_sync(
                            blob_id,
                            "ready",
                            size_bytes=len(file_bytes),
                            content_hash_val=content_hash(file_bytes),
                        )
                    except BlobQuotaExceededError:
                        # Run succeeded but this blob would breach the
                        # session quota — mark as error so the run
                        # finalization isn't aborted entirely.
                        # Delete the backing file to prevent untracked
                        # disk growth from repeated over-quota outputs.
                        if storage.exists():
                            storage.unlink()
                        record = self._finalize_blob_sync(blob_id, "error")
                else:
                    record = self._finalize_blob_sync(blob_id, "error")
            else:
                # Run failed — delete the backing file so the
                # filesystem matches the DB metadata (size_bytes=0,
                # content_hash=None).  Without this, repeated
                # failed runs can grow disk usage without bound
                # while quota accounting sees only zero-byte
                # error rows.
                if storage.exists():
                    storage.unlink()
                record = self._finalize_blob_sync(blob_id, "error")
            return record
        except self._PER_BLOB_SUPPRESSED as exc:
            # Best-effort: transition the failed blob to "error" so it does
            # not remain permanently pending.  Return explicit error records
            # (never a silent swallow) describing the primary fault and any
            # recovery fault, so the batch caller surfaces both to auditors.
            blob_errors = [
                BlobFinalizationError(
                    blob_id=blob_id,
                    exc_type=type(exc).__name__,
                    detail=str(exc),
                )
            ]
            recovery_exc = self._best_effort_mark_blob_error(blob_id)
            if recovery_exc is not None:
                blob_errors.append(
                    BlobFinalizationError(
                        blob_id=blob_id,
                        exc_type=f"RecoveryFailed[{type(recovery_exc).__name__}]",
                        detail=str(recovery_exc),
                    )
                )
            return blob_errors

    def _best_effort_mark_blob_error(self, blob_id: UUID) -> SQLAlchemyError | OSError | None:
        """Transition a still-pending blob to ``error`` status, best effort.

        The ``WHERE status='pending'`` makes this a no-op if the blob was
        already finalized or deleted.  Returns the DB/IO fault if the update
        itself failed (so the caller records a ``RecoveryFailed[...]`` audit
        entry) or ``None`` on success.  Narrow to DB/IO faults — programmer
        bugs (TypeError, AttributeError, AssertionError) must propagate per
        offensive-programming policy.
        """
        try:
            with self._engine.begin() as err_conn:
                err_conn.execute(
                    blobs_table.update()
                    .where(blobs_table.c.id == str(blob_id))
                    .where(blobs_table.c.status == "pending")
                    .values(status="error")
                )
        except (SQLAlchemyError, OSError) as rec_exc:
            return rec_exc
        return None

    async def copy_blobs_for_fork(
        self,
        source_session_id: UUID,
        target_session_id: UUID,
    ) -> dict[UUID, BlobRecord]:
        """Copy all ready blobs from source session to target session.

        Pre-checks total source blob size against the target session's
        quota before copying any files. This eliminates partial-write
        scenarios — either all blobs are copied or none are.

        On any failure during the copy loop, cleans up files already
        written before re-raising.
        """
        source_blobs = await self.list_blobs(source_session_id, limit=None)
        ready_blobs = [b for b in source_blobs if b.status == "ready"]

        if not ready_blobs:
            return {}

        # Pre-check: will the total source blob size fit in the target quota?
        total_source_bytes = sum(b.size_bytes for b in ready_blobs)
        target_session_id_str = str(target_session_id)

        def _check_quota() -> int:
            with self._engine.connect() as conn:
                current = conn.execute(
                    select(func.coalesce(func.sum(blobs_table.c.size_bytes), 0)).where(blobs_table.c.session_id == target_session_id_str)
                ).scalar()
                # COALESCE guarantees an exact int; bool/subclasses or any
                # other type are Tier 1 anomalies. Explicit raise so the
                # guard survives ``python -O``.
                if type(current) is not int:
                    raise AuditIntegrityError(f"Tier 1: COALESCE(SUM) returned {type(current).__name__}, expected int")
                return current

        current_usage = await self._run_sync(_check_quota)
        if current_usage + total_source_bytes > self._max_storage_per_session:
            raise BlobQuotaExceededError(
                target_session_id_str,
                current_bytes=current_usage,
                limit_bytes=self._max_storage_per_session,
            )

        # Copy blobs — clean up partial writes on any failure.
        # Build old_id → new_blob mapping for source reference rewriting.
        blob_map: dict[UUID, BlobRecord] = {}
        copied: list[BlobRecord] = []
        try:
            for blob in ready_blobs:
                content = await self.read_blob_content(blob.id)
                new_blob = await self.create_blob(
                    session_id=target_session_id,
                    filename=blob.filename,
                    content=content,
                    mime_type=blob.mime_type,
                    created_by=blob.created_by,
                    source_description=f"copied from session fork (original: {blob.id})",
                )
                copied.append(new_blob)
                blob_map[blob.id] = new_blob
        except Exception as primary_exc:
            # Clean up both files AND database rows for any blobs already
            # committed. create_blob() commits each blob atomically, so
            # without this cleanup the forked session would have "ready"
            # blob metadata pointing at files we're about to delete.
            #
            # Cleanup failures must NOT be silently swallowed: a failed
            # delete_blob leaves an orphan DB row in the target session
            # that auditors would interpret as a successfully copied blob.
            # Mirror the RecoveryFailed[...] convention used by
            # ``BlobServiceImpl.finalize_run_output_blobs`` (the per-blob
            # error-record path inside its nested ``_sync`` closure): narrow
            # the catch to (SQLAlchemyError, OSError) — programmer bugs must
            # propagate — collect every cleanup failure, and attach them
            # as notes on primary_exc.  The fallback file unlink stays for
            # disk-quota recovery, but the DB-row orphan is now visible
            # to operators reading the traceback.  Bare `raise` re-raises
            # primary_exc (sys.exc_info() reverts after each nested except),
            # preserving the original copy failure as the headline.
            cleanup_failures: list[tuple[UUID, BaseException]] = []
            for written_blob in copied:
                cleanup_exc = await self._cleanup_forked_blob(written_blob)
                if cleanup_exc is not None:
                    cleanup_failures.append((written_blob.id, cleanup_exc))
            for orphan_id, recorded_exc in cleanup_failures:
                primary_exc.add_note(
                    f"RecoveryFailed[{type(recorded_exc).__name__}]: "
                    f"could not delete partially-copied blob {orphan_id} from "
                    f"target session {target_session_id_str} "
                    f"({recorded_exc}). "
                    f"Storage file was unlinked, but the DB row remains and "
                    f"will appear as a 'ready' blob in the target session — "
                    f"manual cleanup of blobs.id={orphan_id} required."
                )
                _BLOB_COPY_FORK_ORPHAN_ROWS_COUNTER.add(
                    1,
                    {
                        "orphan_blob_id": str(orphan_id),
                        "target_session_id": target_session_id_str,
                        "exc_type": type(recorded_exc).__name__,
                    },
                )
            raise

        return blob_map

    async def _cleanup_forked_blob(self, written_blob: BlobRecord) -> SQLAlchemyError | OSError | None:
        """Delete a partially-copied fork blob, returning any cleanup fault.

        ``delete_blob`` performs filesystem + database I/O — a genuine
        external boundary.  On a narrow DB/IO fault this **returns** the
        exception (the caller records a ``RecoveryFailed[...]`` note + counter
        so the orphaned DB row is visible to auditors) after a fallback file
        unlink for disk-quota recovery; on success it returns ``None``.
        Programmer bugs (TypeError, AttributeError, AssertionError) are not
        caught and propagate per offensive-programming policy.
        """
        try:
            await self.delete_blob(written_blob.id)
        except (SQLAlchemyError, OSError) as cleanup_exc:
            storage = Path(written_blob.storage_path)
            if storage.exists():
                storage.unlink(missing_ok=True)
            return cleanup_exc
        return None

    def _finalize_blob_sync(
        self,
        blob_id: UUID,
        status: FinalizeBlobStatus,
        size_bytes: int | None = None,
        content_hash_val: str | None = None,
    ) -> BlobRecord:
        """Synchronous single-blob finalize for use inside _run_sync closures."""
        blob_id_str = str(blob_id)
        # Invalid status is a programmer bug at a Protocol boundary, not a
        # per-blob operational condition.  RuntimeError propagates past
        # _PER_BLOB_SUPPRESSED so the loop in finalize_run_output_blobs
        # crashes loudly instead of silently converting the caller's typo
        # into an "error" record the auditor cannot distinguish from a
        # genuine run failure.  Mirrors the RuntimeError in finalize_blob().
        if status not in FINALIZE_BLOB_STATUSES:
            raise RuntimeError(f"Invalid finalize status '{status}' — must be one of {sorted(FINALIZE_BLOB_STATUSES)}")
        # Single source of truth for the ready-requires-valid-hash rule.
        # See _validate_finalize_hash() docstring.
        _validate_finalize_hash(blob_id_str, status, content_hash_val)

        with self._engine.begin() as conn:
            row = conn.execute(select(blobs_table).where(blobs_table.c.id == blob_id_str)).first()
            if row is None:
                raise BlobNotFoundError(blob_id_str)
            if row.status != "pending":
                raise BlobStateError(
                    blob_id_str,
                    message=f"Cannot finalize blob {blob_id_str} — status is '{row.status}', expected 'pending'",
                )

            self._enforce_ready_finalize_quota(
                conn,
                blob_id_str=blob_id_str,
                session_id_str=row.session_id,
                status=status,
                size_bytes=size_bytes,
            )

            updates: dict[str, Any] = {"status": status}
            if size_bytes is not None:
                updates["size_bytes"] = size_bytes
            if content_hash_val is not None:
                updates["content_hash"] = content_hash_val

            conn.execute(blobs_table.update().where(blobs_table.c.id == blob_id_str).values(**updates))
            updated = conn.execute(select(blobs_table).where(blobs_table.c.id == blob_id_str)).first()
            if updated is None:
                raise RuntimeError(f"Blob {blob_id_str} vanished during finalize — concurrent deletion?")
            return self._row_to_record(updated)


# SHA-256 hex digest: exactly 64 lowercase hex characters.  Must match
# FilesystemPayloadStore's validator (core/payload_store.py) — a blob
# whose content_hash round-trips through the audit trail must use the
# same canonical form everywhere.  Used with ``fullmatch`` (NOT
# ``match``) because Python's ``$`` anchor matches at end-of-string OR
# just before a final ``\n``, so the naive ``^[a-f0-9]{64}$`` pattern
# would accept ``"a" * 64 + "\n"`` — letting a newline-terminated hash
# slip past the pre-check and land at the DB CHECK as an opaque
# IntegrityError rather than the structured BlobStateError this
# validator is supposed to raise.
_SHA256_HEX_PATTERN = re.compile(r"[a-f0-9]{64}")


def _validate_finalize_hash(
    blob_id_str: str,
    status: FinalizeBlobStatus,
    content_hash_val: str | None,
) -> None:
    """Service-layer pre-check for the ``ready`` content_hash invariant.

    This is the FIRST of two walls enforcing the Tier-1 integrity
    contract that makes ``read_blob_content`` verifiable (AD-5/AD-7 in
    docs/plans/rc4.2-ux-remediation/2026-03-30-02-blob-manager-subplan.md).
    A ``ready`` blob MUST carry a SHA-256 hex digest; before this
    pre-check existed, a caller could finalize with a bogus string like
    ``"abc123"`` and the DB would happily store it, leaving a ``ready``
    row whose hash cannot be produced by any real bytes on disk.

    Division of responsibility
    --------------------------
    This function is the SERVICE-LAYER pre-check. It runs on every
    ``finalize_blob`` / ``_finalize_blob_sync`` write-path call and
    raises :class:`BlobStateError` — a structured, caller-friendly
    diagnostic — before any SQL is issued. The DB-level CHECK
    constraint ``ck_blobs_ready_hash`` is the AUTHORITATIVE guard: it
    closes the same invariant for any writer that bypasses this service
    (direct SQL or an ORM call path that skips finalize). If these two
    guards disagree, the DB CHECK wins and the service pre-check is the
    bug.

    Keeping both guards means a service regression surfaces as a clean
    BlobStateError at the write-path entry point (easy to debug),
    while a writer that skips the service still cannot corrupt the
    audit trail. The shape rule is kept in agreement between the two
    sites by design — the current session schema declares the DB-side
    guard, and the tests in
    ``tests/unit/web/blobs/test_service.py::TestBlobsReadyHashDBConstraint``
    pin the DB guard independently of this one.
    """
    if status != "ready":
        return
    if content_hash_val is None:
        raise BlobStateError(
            blob_id_str,
            message=f"Tier 1: cannot finalize blob {blob_id_str} as 'ready' without content_hash — audit integrity requires a hash",
        )
    # ``fullmatch`` (not ``match``) — see the _SHA256_HEX_PATTERN comment
    # above for why ``^...$`` + ``match`` admits trailing newlines.
    if not _SHA256_HEX_PATTERN.fullmatch(content_hash_val):
        raise BlobStateError(
            blob_id_str,
            message=f"Tier 1: content_hash must be 64 lowercase hex characters (SHA-256), got {content_hash_val!r}",
        )
