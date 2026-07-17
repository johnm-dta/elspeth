"""Custody-safe materialization for canonical ``set_pipeline`` proposals.

The preliminary candidate builder is deliberately side-effect free: it may
validate ``source.inline_blob`` using prepared bytes and a provisional path,
but it cannot publish those bytes.  This module owns the narrow transition
from that acceptable draft to exact proposal arguments containing only a
deterministic ``source.blob_id``.
"""

from __future__ import annotations

import hmac
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from uuid import UUID

from sqlalchemy import Engine

from elspeth.contracts.blobs import AllowedMimeType, BlobRecord, InlineCustodyRequest
from elspeth.contracts.enums import CreationModality
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.freeze import freeze_fields
from elspeth.core.canonical import stable_hash
from elspeth.web.blobs.service import (
    BlobServiceImpl,
    content_hash,
    inline_custody_blob_id,
    sanitize_filename,
)

if TYPE_CHECKING:
    from elspeth.web.composer.tools.blobs import _PreparedBlobCreate


_AUDIT_REDACTION_MARKER = "[redacted inline content held for custody]"


def _detached_json(value: Any) -> Any:
    """Detach a decoded JSON tree without coercing or canonicalizing it."""
    if type(value) is dict:
        return {key: _detached_json(child) for key, child in value.items()}
    if type(value) is list:
        return [_detached_json(child) for child in value]
    return value


def inline_custody_audit_projection(arguments: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return an audit-safe projection before candidate validation begins.

    Malformed drafts still open a dispatch audit.  Replacing the entire
    untrusted ``content`` value up front prevents an invalid draft from
    leaking bytes through ``DispatchAudit`` before the redaction manifest can
    validate the surrounding shape.
    """
    projected_value = _detached_json(dict(arguments))
    if type(projected_value) is not dict:
        raise AuditIntegrityError("Pipeline custody audit projection must remain an object")
    projected = projected_value

    def _redact(value: Any) -> None:
        if type(value) is dict:
            for key, child in value.items():
                if key == "inline_blob":
                    value[key] = _AUDIT_REDACTION_MARKER
                else:
                    _redact(child)
        elif type(value) is list:
            for child in value:
                _redact(child)

    _redact(projected)
    return projected


def inline_custody_manifest_redaction_input(arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Restore a schema-valid shell solely for manifest redaction.

    Dispatch auditing removes the whole untrusted ``inline_blob`` value before
    validation. The manifest still needs a structurally valid value so it can
    redact other sensitive pipeline fields; this shell contains no user bytes
    and is projected back to the whole-value marker before persistence.
    """
    restored = _detached_json(dict(arguments))
    if type(restored) is not dict:
        raise AuditIntegrityError("Pipeline custody manifest input must remain an object")

    def _restore(value: Any) -> None:
        if type(value) is dict:
            for key, child in value.items():
                if key == "inline_blob" and child == _AUDIT_REDACTION_MARKER:
                    value[key] = {
                        "filename": "redacted-inline-content",
                        "mime_type": "text/plain",
                        "content": _AUDIT_REDACTION_MARKER,
                    }
                else:
                    _restore(child)
        elif type(value) is list:
            for child in value:
                _restore(child)

    _restore(restored)
    return restored


def _contains_inline_blob(value: Any) -> bool:
    if type(value) is dict:
        return "inline_blob" in value or any(_contains_inline_blob(child) for child in value.values())
    if type(value) is list:
        return any(_contains_inline_blob(child) for child in value)
    return False


@dataclass(frozen=True, slots=True)
class PipelineCustodyPreparation:
    """Pure result produced before custody performs filesystem or DB I/O."""

    arguments: Mapping[str, Any]
    request: InlineCustodyRequest
    blob_id: UUID

    def __post_init__(self) -> None:
        freeze_fields(self, "arguments")


def _validate_prepared_inline_blob(
    inline_blob: Mapping[str, Any],
    prepared: _PreparedBlobCreate,
) -> None:
    """Fail closed unless candidate bytes exactly match removed arguments."""
    inline_content = inline_blob["content"] if "content" in inline_blob else None
    inline_mime_type = inline_blob["mime_type"] if "mime_type" in inline_blob else None
    inline_filename = inline_blob["filename"] if "filename" in inline_blob else None
    inline_description = inline_blob["description"] if "description" in inline_blob else None
    if type(inline_content) is not str:
        raise AuditIntegrityError("Pipeline custody inline content must be a string")
    if type(inline_mime_type) is not str:
        raise AuditIntegrityError("Pipeline custody inline MIME type must be a string")
    if type(inline_filename) is not str:
        raise AuditIntegrityError("Pipeline custody inline filename must be a string")
    if inline_description is not None and type(inline_description) is not str:
        raise AuditIntegrityError("Pipeline custody inline description must be a string or null")
    try:
        inline_bytes = inline_content.encode("utf-8")
        inline_safe_filename = sanitize_filename(inline_filename)
    except (UnicodeEncodeError, ValueError) as exc:
        raise AuditIntegrityError("Pipeline custody inline candidate metadata is invalid") from exc

    prepared_hash = content_hash(prepared.content_bytes)
    if type(prepared.content_hash) is not str or not hmac.compare_digest(prepared.content_hash, prepared_hash):
        raise AuditIntegrityError("Pipeline custody prepared content hash does not match prepared bytes")
    if not hmac.compare_digest(inline_bytes, prepared.content_bytes):
        raise AuditIntegrityError("Pipeline custody prepared bytes do not match inline arguments")
    if inline_mime_type != prepared.mime_type:
        raise AuditIntegrityError("Pipeline custody prepared MIME type does not match inline arguments")
    if inline_safe_filename != prepared.filename:
        raise AuditIntegrityError("Pipeline custody prepared filename does not match inline arguments")
    if inline_description != prepared.description:
        raise AuditIntegrityError("Pipeline custody prepared description does not match inline arguments")


def prepare_pipeline_custody(
    arguments: Mapping[str, Any],
    prepared: _PreparedBlobCreate,
    *,
    session_id: str,
) -> PipelineCustodyPreparation:
    """Replace the one supported inline source with a deterministic blob id.

    ``creating_arguments_hash`` is intentionally excluded from the UUID5
    identity.  The safe arguments must contain the UUID before their hash can
    be computed, so the request is first used only to derive that UUID and is
    then rebound to the hash of the final, custody-safe arguments.
    """
    try:
        session_uuid = UUID(session_id)
    except (TypeError, ValueError) as exc:
        raise AuditIntegrityError("Pipeline custody session_id must be a canonical UUID string") from exc
    if str(session_uuid) != session_id:
        raise AuditIntegrityError("Pipeline custody session_id must be a canonical UUID string")

    safe_arguments = _detached_json(dict(arguments))
    source = safe_arguments["source"] if "source" in safe_arguments else None
    if type(source) is not dict:
        raise AuditIntegrityError("Pipeline custody requires canonical set_pipeline source arguments")
    inline_blob = source["inline_blob"] if "inline_blob" in source else None
    if type(inline_blob) is not dict:
        raise AuditIntegrityError("Pipeline custody requires canonical source.inline_blob arguments")
    _validate_prepared_inline_blob(inline_blob, prepared)
    if "blob_id" in source:
        raise AuditIntegrityError("Pipeline custody refuses source arguments containing both blob_id and inline_blob")
    options = source["options"] if "options" in source else None
    if options is not None and type(options) is not dict:
        raise AuditIntegrityError("Pipeline custody requires canonical source.options arguments")
    if type(options) is dict and "blob_ref" in options:
        raise AuditIntegrityError("Pipeline custody refuses caller-authored source.options.blob_ref")
    if type(prepared.description) not in {str, type(None)}:
        raise AuditIntegrityError("Prepared inline custody description must be str or None")

    provisional_request = InlineCustodyRequest(
        session_id=session_uuid,
        filename=prepared.filename,
        content=prepared.content_bytes,
        mime_type=cast(AllowedMimeType, prepared.mime_type),
        source_description=prepared.description,
        creation_modality=prepared.creation_modality,
        created_from_message_id=prepared.created_from_message_id or "",
        creating_model_identifier=prepared.creating_model_identifier,
        creating_model_version=prepared.creating_model_version,
        creating_provider=prepared.creating_provider,
        creating_composer_skill_hash=prepared.creating_composer_skill_hash,
        creating_arguments_hash=prepared.creating_arguments_hash,
    )
    blob_id = inline_custody_blob_id(provisional_request)

    del source["inline_blob"]
    source["blob_id"] = str(blob_id)
    if _contains_inline_blob(safe_arguments):
        raise AuditIntegrityError("Pipeline custody rewrite left an inline_blob member in proposal arguments")
    safe_arguments_hash = stable_hash(safe_arguments)
    final_arguments_hash = safe_arguments_hash if prepared.creation_modality is not CreationModality.VERBATIM else None
    request = replace(
        provisional_request,
        creating_arguments_hash=final_arguments_hash,
    )
    if inline_custody_blob_id(request) != blob_id:
        raise AuditIntegrityError("Pipeline custody identity changed after safe argument hashing")
    return PipelineCustodyPreparation(
        arguments=safe_arguments,
        request=request,
        blob_id=blob_id,
    )


async def finalize_pipeline_custody(
    preparation: PipelineCustodyPreparation,
    *,
    engine: Engine,
    data_dir: str | Path,
    max_storage_per_session: int,
) -> BlobRecord:
    """Materialize a prepared inline source through the shared blob service."""
    service = BlobServiceImpl(
        engine,
        Path(data_dir),
        max_storage_per_session=max_storage_per_session,
    )
    record = await service.reserve_inline_custody(preparation.request)
    if record.id != preparation.blob_id:
        raise AuditIntegrityError("Inline custody returned a blob id different from the prepared proposal")
    return record
