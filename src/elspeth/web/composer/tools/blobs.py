"""Composer blob-storage plane — session-scoped binary blob handlers.

Hosts:

- Tool handlers for blob CRUD: ``_execute_create_blob`` / ``_execute_update_blob``
  / ``_execute_delete_blob`` / ``_execute_get_blob_content`` /
  ``_handle_list_blobs`` / ``_handle_get_blob_metadata``.
- Quota / lock state (``_BLOB_QUOTA_BYTES``, ``_SESSION_BLOB_LOCKS``).
- Storage primitives (``_prepare_blob_create`` / ``_persist_prepared_blob_create`` /
  ``_sync_get_blob`` / ``_sync_list_blobs`` / ``_check_blob_quota``).
- Blob DTOs (``BlobToolRecord`` / ``BlobCreatePayload`` / ``_PreparedBlobCreate``)
  and in-transaction signal exceptions (``_BlobQuotaExceededInTxn`` /
  ``_BlobUpdateBlockedByActiveRun``).
- Tool-classification name sets and predicates live in
  ``elspeth.web.composer.tools.discovery``; the trailing comment in this file
  points to that module.

Patch-target stability: tests that bind ``_BLOB_QUOTA_BYTES`` /
``_check_blob_quota`` / ``_sync_get_blob`` by full dotted path must target this
module (``elspeth.web.composer.tools.blobs.<name>``), not the package facade —
helpers here resolve those names via their local module namespace.
"""

from __future__ import annotations

import hmac
import os
import tempfile
import threading
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict, cast
from uuid import UUID, uuid4

from pydantic import ValidationError as PydanticValidationError
from sqlalchemy import Engine, delete, func, select, update

from elspeth.contracts.blobs_inline import (
    ALLOWED_CONTENT_ENCODINGS,
    BlobInlineRef,
    ContentEncoding,
)
from elspeth.contracts.enums import CreationModality, is_llm_authored_creation_modality
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.freeze import deep_thaw
from elspeth.contracts.trust_boundary import trust_boundary
from elspeth.web.blobs.protocol import BlobIntegrityError
from elspeth.web.blobs.service import (
    _ACTIVE_RUN_COMPOSITION_COLUMNS,
    _active_run_pipeline_dict,
    _composition_references_blob,
    _guard_blob_row_literals,
    _lock_session_for_blob_quota,
    content_hash,
    sanitize_filename,
)
from elspeth.web.composer.protocol import ToolArgumentError
from elspeth.web.composer.redaction import (
    CreateBlobArgumentsModel,
    UpdateBlobArgumentsModel,
)
from elspeth.web.composer.state import (
    CompositionState,
)
from elspeth.web.composer.tools._common import (
    ToolContext,
    ToolResult,
    _discovery_result,
    _failure_result,
    _mutation_result,
    _runtime_owned_llm_option_error,
)
from elspeth.web.composer.tools.declarations import (
    ToolDeclaration,
    ToolKind,
)
from elspeth.web.interpretation_state import INTERPRETATION_REQUIREMENTS_KEY
from elspeth.web.sessions.models import blob_run_links_table, blobs_table, composition_states_table, runs_table


class BlobToolRecord(TypedDict):
    """Closed dict shape returned by composer blob discovery helpers.

    Inline-blob provenance fields mirror the columns introduced on
    ``blobs_table``: ``creation_modality`` carries the
    closed-enum string (wire form), ``created_from_message_id`` binds to
    the originating chat message, and the five ``creating_*`` fields
    carry LLM-provenance for the three LLM-authored modalities.
    """

    id: str
    session_id: str
    filename: str
    mime_type: str
    size_bytes: int
    content_hash: str | None
    storage_path: str
    created_by: str
    source_description: str | None
    status: str
    creation_modality: str
    created_from_message_id: str | None
    creating_model_identifier: str | None
    creating_model_version: str | None
    creating_provider: str | None
    creating_composer_skill_hash: str | None
    creating_arguments_hash: str | None


class BlobCreatePayload(TypedDict):
    """Closed dict shape for the create_blob tool's success result data."""

    blob_id: str
    filename: str
    mime_type: str
    size_bytes: int
    content_hash: str


def _blob_row_to_tool_dict(row: Any) -> BlobToolRecord:
    """Serialize a validated blobs row to the tool-layer dict shape."""
    _guard_blob_row_literals(row)
    return {
        "id": row.id,
        "session_id": row.session_id,
        "filename": row.filename,
        "mime_type": row.mime_type,
        "size_bytes": row.size_bytes,
        "content_hash": row.content_hash,
        "storage_path": row.storage_path,
        "created_by": row.created_by,
        "source_description": row.source_description,
        "status": row.status,
        # Inline-blob provenance. The Tier 1 guard in
        # ``_guard_blob_row_literals`` already validated
        # ``creation_modality`` against the closed CreationModality enum.
        "creation_modality": row.creation_modality,
        "created_from_message_id": row.created_from_message_id,
        "creating_model_identifier": row.creating_model_identifier,
        "creating_model_version": row.creating_model_version,
        "creating_provider": row.creating_provider,
        "creating_composer_skill_hash": row.creating_composer_skill_hash,
        "creating_arguments_hash": row.creating_arguments_hash,
    }


def _sync_get_blob(engine: Engine, blob_id: str, session_id: str | None = None) -> BlobToolRecord | None:
    """Synchronous blob lookup for use in the tool executor thread."""
    with engine.connect() as conn:
        query = select(blobs_table).where(blobs_table.c.id == blob_id)
        if session_id is not None:
            query = query.where(blobs_table.c.session_id == session_id)
        row = conn.execute(query).first()
        if row is None:
            return None
        return _blob_row_to_tool_dict(row)


@trust_boundary(
    tier=3,
    source="LLM composer tool-call blob_id argument",
    source_param="blob_id",
    suppresses=("R5",),
    invariant="returns a repairable error message for non-string or non-UUID blob_id and None for canonical input; never raises on blob_id",
    non_raising=True,
)
def _blob_id_uuid_validation_error(blob_id: Any) -> str | None:
    """Return a repairable boundary error when ``blob_id`` is not canonical."""
    if not isinstance(blob_id, str):
        return f"blob_id must be a UUID string, got {type(blob_id).__name__}."
    try:
        UUID(blob_id)
    except ValueError:
        return (
            "blob_id is not a valid UUID. Use list_blobs or "
            "list_composer_blobs to select an uploaded blob, ask the user to "
            "upload the source file, or use create_blob for inline content "
            "before calling this tool."
        )
    return None


def _sync_get_blob_by_storage_path(
    engine: Engine,
    storage_path: str,
    session_id: str,
) -> BlobToolRecord | None:
    """Look up a blob by its canonical storage_path within a session.

    Used by ``handle_step_1_source`` (steps.py) to detect whether a path
    supplied via the guided SchemaForm resolves to an already-uploaded blob.
    When it does, the blob_id (= blob["id"]) can be injected as ``blob_ref``
    into ``SourceResolved.options`` so that the recipe slot resolvers in
    ``recipe_match.py`` have access to the UUID they need.

    Returns None if no blob row matches the path, which is the correct
    representation for path-based sources that are not blob-backed.
    """
    with engine.connect() as conn:
        query = select(blobs_table).where(blobs_table.c.session_id == session_id).where(blobs_table.c.storage_path == storage_path)
        row = conn.execute(query).first()
        if row is None:
            return None
        return _blob_row_to_tool_dict(row)


def _sync_get_blob_by_id(
    engine: Engine,
    blob_id: str,
    session_id: str,
) -> BlobToolRecord | None:
    """Look up a blob by its UUID within a session (authoritative DB query).

    The inverse of :func:`_sync_get_blob_by_storage_path`: used by
    ``handle_step_1_source`` (steps.py) to resolve a ``blob:<ref>`` path sentinel
    — emitted by ``build_step_1_schema_form_turn_from_resolved`` to keep the
    absolute storage_path off the wire — back to the blob's real ``storage_path``
    before the source is committed. Session-scoped so a blob ref cannot resolve
    across sessions (project/tenant isolation). Returns None if no row matches.
    """
    with engine.connect() as conn:
        query = select(blobs_table).where(blobs_table.c.session_id == session_id).where(blobs_table.c.id == blob_id)
        row = conn.execute(query).first()
        if row is None:
            return None
        return _blob_row_to_tool_dict(row)


def _sync_list_blobs(engine: Engine, session_id: str) -> list[dict[str, Any]]:
    """Synchronous blob listing for use in the tool executor thread."""
    with engine.connect() as conn:
        rows = conn.execute(
            select(blobs_table).where(blobs_table.c.session_id == session_id).order_by(blobs_table.c.created_at.desc()).limit(50)
        ).fetchall()
        return [
            {
                "id": blob["id"],
                "filename": blob["filename"],
                "mime_type": blob["mime_type"],
                "size_bytes": blob["size_bytes"],
                "created_by": blob["created_by"],
                "status": blob["status"],
            }
            for blob in (_blob_row_to_tool_dict(row) for row in rows)
        ]


def _sync_list_ready_blob_inline_descriptors(engine: Engine, session_id: str) -> list[dict[str, Any]]:
    """Return H4 visibility descriptors for ready session blobs."""
    with engine.connect() as conn:
        rows = conn.execute(
            select(blobs_table)
            .where(blobs_table.c.session_id == session_id)
            .where(blobs_table.c.status == "ready")
            .order_by(blobs_table.c.created_at.desc())
            .limit(50)
        ).fetchall()

    descriptors: list[dict[str, Any]] = []
    for row in rows:
        blob = _blob_row_to_tool_dict(row)
        if blob["content_hash"] is None:
            raise AuditIntegrityError(f"Ready blob '{blob['id']}' has null content_hash; cannot list for inline_content authoring")
        descriptors.append(
            {
                "blob_id": blob["id"],
                "mime_type": blob["mime_type"],
                "size_bytes": blob["size_bytes"],
                "content_hash": blob["content_hash"],
                "filename": blob["filename"],
            }
        )
    return descriptors


def _handle_list_blobs(
    arguments: dict[str, Any],
    state: CompositionState,
    context: ToolContext,
) -> ToolResult:
    session_engine = context.session_engine
    session_id = context.session_id
    if session_engine is None or session_id is None:
        return _failure_result(state, "Blob tools require session context.")
    blobs = _sync_list_blobs(session_engine, session_id)
    return _discovery_result(state, blobs)


_LIST_BLOBS_DECLARATION = ToolDeclaration(
    name="list_blobs",
    handler=_handle_list_blobs,
    kind=ToolKind.BLOB_DISCOVERY,
    description="List uploaded/created files (blobs) in this session with metadata.",
    json_schema={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
)


def _handle_list_composer_blobs(
    arguments: dict[str, Any],
    state: CompositionState,
    context: ToolContext,
) -> ToolResult:
    """List blobs using the ADR-025 composer-LLM visibility shape.

    The LLM sees only metadata needed to author a pinned inline-content
    marker. Bytes, previews, storage paths, and free-text descriptions stay
    out of the response surface.
    """
    del arguments
    session_engine = context.session_engine
    session_id = context.session_id
    if session_engine is None or session_id is None:
        return _failure_result(state, "Blob tools require session context.")
    return _discovery_result(state, {"blobs": _sync_list_ready_blob_inline_descriptors(session_engine, session_id)})


_LIST_COMPOSER_BLOBS_DECLARATION = ToolDeclaration(
    name="list_composer_blobs",
    handler=_handle_list_composer_blobs,
    kind=ToolKind.BLOB_DISCOVERY,
    description=(
        "List ready blobs available for audited inline-content authoring. "
        "Returns only blob_id, mime_type, size_bytes, content_hash, and filename; never content bytes."
    ),
    json_schema={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
)


def _handle_get_blob_metadata(
    arguments: dict[str, Any],
    state: CompositionState,
    context: ToolContext,
) -> ToolResult:
    session_engine = context.session_engine
    session_id = context.session_id
    if session_engine is None or session_id is None:
        return _failure_result(state, "Blob tools require session context.")
    blob_id_error = _blob_id_uuid_validation_error(arguments["blob_id"])
    if blob_id_error is not None:
        return _failure_result(state, blob_id_error)
    blob = _sync_get_blob(session_engine, arguments["blob_id"], session_id)
    if blob is None:
        return _failure_result(state, "Blob not found for this session.")
    safe_blob = {
        "id": blob["id"],
        "filename": blob["filename"],
        "mime_type": blob["mime_type"],
        "size_bytes": blob["size_bytes"],
        "content_hash": blob["content_hash"],
        "status": blob["status"],
    }
    return _discovery_result(state, safe_blob)


_GET_BLOB_METADATA_DECLARATION = ToolDeclaration(
    name="get_blob_metadata",
    handler=_handle_get_blob_metadata,
    kind=ToolKind.BLOB_DISCOVERY,
    description="Get metadata for a specific blob (file) by ID.",
    json_schema={
        "type": "object",
        "properties": {
            "blob_id": {"type": "string", "description": "Blob ID."},
        },
        "required": ["blob_id"],
        "additionalProperties": False,
    },
)


def _set_nested_option(container: dict[str, Any], keys: list[str], value: Any) -> dict[str, Any]:
    if not keys:
        raise ValueError("field_path must include at least one .options.<field> segment")
    if len(keys) == 1:
        container[keys[0]] = value
        return container
    head = keys[0]
    if head in container:
        child = container[head]
        if not isinstance(child, Mapping):
            raise ValueError(f"field_path segment {head!r} already exists and is not an object")
        nested = dict(deep_thaw(child))
    else:
        nested = {}
    container[head] = _set_nested_option(nested, keys[1:], value)
    return container


def _apply_inline_blob_marker(state: CompositionState, field_path: str, marker: dict[str, Any]) -> CompositionState:
    prefix, separator, rest = field_path.partition(".options.")
    if separator == "":
        raise ValueError("field_path must include '.options.'")
    keys = rest.split(".")

    if prefix == "source":
        source_name = "source"
    elif prefix.startswith("source:"):
        source_name = prefix.removeprefix("source:")
        if not source_name:
            raise ValueError("source:<name> field_path must include a source name")
    else:
        source_name = None

    if source_name is not None:
        source = state.sources[source_name] if source_name in state.sources else None
        if source is None:
            if source_name == "source":
                raise ValueError("Cannot wire source ref: no source has been set")
            raise ValueError(f"Source {source_name!r} not found in composition state")
        # Symmetric with the node arm below: never let a wire write land inside a
        # source's interpretation_requirements. Source review metadata
        # (INVENTED_SOURCE) may only be staged as a pending composer requirement
        # and resolved by resolve_interpretation_event — a wired ref here would
        # corrupt that structure outside the review boundary.
        if keys[0] == INTERPRETATION_REQUIREMENTS_KEY:
            raise ValueError(
                "wire_blob_inline_ref cannot write source interpretation_requirements; "
                "review metadata may only be staged as pending composer input and "
                "resolved by resolve_interpretation_event."
            )
        patched_options = _set_nested_option(dict(deep_thaw(source.options)), keys, marker)
        return state.with_named_source(source_name, replace(source, options=patched_options))

    if prefix.startswith("node:"):
        node_id = prefix.removeprefix("node:")
        new_nodes = []
        found = False
        for node in state.nodes:
            if node.id == node_id:
                if node.plugin == "llm" and keys[0] == INTERPRETATION_REQUIREMENTS_KEY:
                    raise ValueError(
                        "wire_blob_inline_ref cannot write LLM interpretation_requirements; "
                        "review metadata may only be staged as pending composer input and "
                        "resolved by resolve_interpretation_event."
                    )
                runtime_owned_error = _runtime_owned_llm_option_error(
                    node.plugin,
                    {keys[0]: marker},
                    tool_name="wire_blob_inline_ref",
                )
                if runtime_owned_error is not None:
                    raise ValueError(runtime_owned_error)
                patched_options = _set_nested_option(dict(deep_thaw(node.options)), keys, marker)
                new_nodes.append(replace(node, options=patched_options))
                found = True
            else:
                new_nodes.append(node)
        if not found:
            raise ValueError(f"Node {node_id!r} not found in composition state")
        return replace(state, nodes=tuple(new_nodes), version=state.version + 1)

    if prefix.startswith("output:"):
        output_name = prefix.removeprefix("output:")
        new_outputs = []
        found = False
        for output in state.outputs:
            if output.name == output_name:
                patched_options = _set_nested_option(dict(deep_thaw(output.options)), keys, marker)
                new_outputs.append(replace(output, options=patched_options))
                found = True
            else:
                new_outputs.append(output)
        if not found:
            raise ValueError(f"Output {output_name!r} not found in composition state")
        return replace(state, outputs=tuple(new_outputs), version=state.version + 1)

    raise ValueError("field_path must start with source.options, source:<name>.options, node:<id>.options, or output:<name>.options")


def _affected_component_for_inline_field_path(field_path: str) -> tuple[str, ...]:
    prefix, _, _rest = field_path.partition(".options.")
    if prefix == "source":
        return ("source",)
    if prefix.startswith("source:"):
        return (prefix.removeprefix("source:"),)
    if prefix.startswith("node:"):
        return (prefix.removeprefix("node:"),)
    if prefix.startswith("output:"):
        return (prefix.removeprefix("output:"),)
    return ()


def _execute_wire_blob_inline_ref(
    arguments: dict[str, Any],
    state: CompositionState,
    context: ToolContext,
) -> ToolResult:
    """Author a widened blob_ref inline-content marker into composition state."""
    session_engine = context.session_engine
    session_id = context.session_id
    if session_engine is None or session_id is None:
        return _failure_result(state, "Blob tools require session context.")

    field_path = arguments["field_path"]
    blob_id_error = _blob_id_uuid_validation_error(arguments["blob_id"])
    if blob_id_error is not None:
        return _failure_result(state, blob_id_error)
    blob_id = UUID(arguments["blob_id"])

    # Tier-3 LLM tool argument. Absent ``encoding`` means utf-8 by the
    # published tool-schema contract (json_schema declares default "utf-8").
    # The ``isinstance(..., str)`` guard is load-bearing and must precede the
    # membership test: the LLM may emit a JSON array/object (Python
    # list/dict), and ``unhashable_value not in ALLOWED_CONTENT_ENCODINGS``
    # would raise TypeError out of the dispatcher rather than returning the
    # explicit failure result. The str narrowing also satisfies the
    # ContentEncoding cast below.
    encoding_value = arguments["encoding"] if "encoding" in arguments else "utf-8"
    if type(encoding_value) is not str:
        return _failure_result(state, f"encoding must be a string, got {type(encoding_value).__name__}")
    if encoding_value not in ALLOWED_CONTENT_ENCODINGS:
        return _failure_result(state, f"encoding must be one of {sorted(ALLOWED_CONTENT_ENCODINGS)}, got {encoding_value!r}")
    encoding = cast(ContentEncoding, encoding_value)

    blob = _sync_get_blob(session_engine, str(blob_id), session_id)
    if blob is None:
        return _failure_result(state, f"Blob '{blob_id}' not found.")
    if blob["status"] != "ready":
        return _failure_result(state, f"Blob '{blob_id}' is not ready (status: {blob['status']}).")
    pinned_hash = blob["content_hash"]
    if pinned_hash is None:
        raise AuditIntegrityError(f"Ready blob '{blob_id}' has null content_hash; cannot author inline_content ref")

    # Optional Tier-3 LLM tool argument; its absence honestly means "no
    # override supplied" (None), so the missing key is recorded as None
    # rather than fabricated into a value.
    sha256_override = arguments["sha256_override"] if "sha256_override" in arguments else None
    if sha256_override is not None and sha256_override != pinned_hash:
        return _failure_result(state, "sha256 override disagrees with authoritative blob content_hash; composer pins from blob metadata")

    try:
        ref = BlobInlineRef(
            field_path=field_path,
            blob_id=blob_id,
            sha256=pinned_hash,
            encoding=encoding,
        )
    except ValueError as exc:
        return _failure_result(state, f"Invalid field_path for inline blob ref: {exc}")

    marker: dict[str, Any] = {
        "blob_ref": str(blob_id),
        "mode": "inline_content",
        "sha256": pinned_hash,
    }
    if encoding != "utf-8":
        marker["encoding"] = encoding

    try:
        new_state = _apply_inline_blob_marker(state, ref.field_path, marker)
    except ValueError as exc:
        return _failure_result(state, str(exc))
    return _mutation_result(new_state, _affected_component_for_inline_field_path(ref.field_path), data={"field_path": ref.field_path})


_WIRE_BLOB_INLINE_REF_DECLARATION = ToolDeclaration(
    name="wire_blob_inline_ref",
    handler=_execute_wire_blob_inline_ref,
    kind=ToolKind.BLOB_MUTATION,
    description=(
        "Author a widened blob_ref inline_content marker at a canonical field_path. "
        "Composer pins sha256 from blob metadata; callers must not pass content bytes."
    ),
    json_schema={
        "type": "object",
        "properties": {
            "field_path": {
                "type": "string",
                "description": (
                    "Canonical path: source.options.<field>, source:<name>.options.<field>, "
                    "node:<node_id>.options.<field>, or output:<name>.options.<field>."
                ),
            },
            "blob_id": {"type": "string", "format": "uuid", "description": "Ready blob ID to wire as inline content."},
            "encoding": {
                "type": "string",
                "enum": sorted(ALLOWED_CONTENT_ENCODINGS),
                "default": "utf-8",
                "description": "Text decoder used at runtime. Defaults to utf-8.",
            },
        },
        "required": ["field_path", "blob_id"],
        "additionalProperties": False,
    },
)


_ALLOWED_BLOB_MIME_TYPES: frozenset[str] = frozenset(
    {
        "text/plain",
        "application/json",
        "text/csv",
        "application/x-jsonlines",
        "application/jsonl",
        "text/jsonl",
    }
)

_BLOB_QUOTA_BYTES: int = 500 * 1024 * 1024


def _resolve_blob_quota_bytes(max_blob_storage_per_session_bytes: int | None) -> int:
    return _BLOB_QUOTA_BYTES if max_blob_storage_per_session_bytes is None else max_blob_storage_per_session_bytes


@dataclass(frozen=True, slots=True)
class _PreparedBlobCreate:
    """Validated blob-create payload ready for filesystem/DB persistence.

    Provenance fields
    -----------------
    ``creation_modality`` declares how the content was produced; mirror
    enum is :class:`elspeth.contracts.enums.CreationModality`.  The five
    ``creating_*`` fields carry LLM-provenance and are populated only for
    LLM-authored modalities — the all-or-nothing invariant is enforced at
    the DB layer by ``ck_blobs_creating_llm_provenance_nullability`` in
    ``web/sessions/models.py``.  ``created_from_message_id`` binds the
    blob to the user chat message that triggered its creation; the
    composite FK on ``(created_from_message_id, session_id)`` rejects
    cross-session lineage.
    """

    blob_id: str
    filename: str
    mime_type: str
    content_bytes: bytes
    content_hash: str
    storage_path: Path
    description: Any | None
    creation_modality: CreationModality
    created_from_message_id: str | None
    creating_model_identifier: str | None
    creating_model_version: str | None
    creating_provider: str | None
    creating_composer_skill_hash: str | None
    creating_arguments_hash: str | None


@dataclass(frozen=True, slots=True)
class _BlobCreationProvenance:
    creation_modality: CreationModality
    creating_model_identifier: str | None
    creating_model_version: str | None
    creating_provider: str | None
    creating_composer_skill_hash: str | None
    creating_arguments_hash: str | None


def _verbatim_blob_creation_provenance() -> _BlobCreationProvenance:
    return _BlobCreationProvenance(
        creation_modality=CreationModality.VERBATIM,
        creating_model_identifier=None,
        creating_model_version=None,
        creating_provider=None,
        creating_composer_skill_hash=None,
        creating_arguments_hash=None,
    )


def _blob_provenance_message_id(user_message_id: str | None) -> str | None:
    return _blob_provenance_string(user_message_id)


def _blob_provenance_string(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized if normalized else None


def _blob_creation_provenance(content: str, context: ToolContext) -> _BlobCreationProvenance:
    """Classify composer-created blob content and return DB provenance fields."""
    user_message_id = _blob_provenance_message_id(context.user_message_id)
    if user_message_id is not None and context.user_message_content is not None and content and content in context.user_message_content:
        return _verbatim_blob_creation_provenance()

    required = {
        "user_message_id": user_message_id,
        "composer_model_identifier": _blob_provenance_string(context.composer_model_identifier),
        "composer_model_version": _blob_provenance_string(context.composer_model_version),
        "composer_provider": _blob_provenance_string(context.composer_provider),
        "composer_skill_hash": _blob_provenance_string(context.composer_skill_hash),
        "tool_arguments_hash": _blob_provenance_string(context.tool_arguments_hash),
    }
    missing = tuple(name for name, value in required.items() if value is None)
    if missing:
        raise AuditIntegrityError(f"LLM-authored blob provenance requires complete composer context; missing: {', '.join(missing)}")

    return _BlobCreationProvenance(
        creation_modality=CreationModality.LLM_GENERATED,
        creating_model_identifier=required["composer_model_identifier"],
        creating_model_version=required["composer_model_version"],
        creating_provider=required["composer_provider"],
        creating_composer_skill_hash=required["composer_skill_hash"],
        creating_arguments_hash=required["tool_arguments_hash"],
    )


def _state_source_blob_refs(state: CompositionState) -> frozenset[str]:
    """Blob refs bound to any pipeline source root."""
    refs: set[str] = set()
    for source_name, source in state.sources.items():
        if "blob_ref" not in source.options:
            continue
        blob_ref = source.options["blob_ref"]
        if not isinstance(blob_ref, str):
            # The canonical writer sets blob_ref exclusively from
            # authoritative blob metadata as blob["id"] (a str) in
            # sources.py::_resolve_source_blob, and every caller-injection
            # path is rejected (_reject_manual_source_blob_ref, the
            # patch_source_options blob_ref guard). A present-but-non-str
            # blob_ref therefore cannot arise from any valid authoring path:
            # it is a corruption of the audited CompositionState. Silently
            # treating it as "not bound" would let _execute_update_blob mutate
            # a blob that is in fact bound to a pipeline source, defeating the
            # binding guard — so escalate rather than suppress.
            raise AuditIntegrityError(
                f"Source '{source_name}' has a non-str blob_ref ({type(blob_ref).__name__}); CompositionState integrity anomaly"
            )
        refs.add(blob_ref)
    return frozenset(refs)


def _blob_storage_path(data_dir: str, session_id: str, blob_id: str, filename: str) -> Path:
    """Compute blob storage path matching BlobServiceImpl layout.

    Pattern: {data_dir}/blobs/{session_id}/{blob_id}_{filename}
    """
    return Path(data_dir).resolve() / "blobs" / session_id / f"{blob_id}_{filename}"


def _check_blob_quota(
    conn: Any,
    session_id: str,
    additional_bytes: int,
    *,
    quota_bytes: int | None = None,
    session_locked: bool = False,
) -> str | None:
    """Check if adding bytes would exceed the session blob quota.

    Returns an error message if quota exceeded, None if OK.
    Runs inside an existing transaction for TOCTOU safety.
    """
    if not session_locked:
        _lock_session_for_blob_quota(conn, session_id)
    current_total = conn.execute(
        select(func.coalesce(func.sum(blobs_table.c.size_bytes), 0)).where(blobs_table.c.session_id == session_id)
    ).scalar()
    current_total = int(current_total)
    resolved_quota = _resolve_blob_quota_bytes(quota_bytes)
    if current_total + additional_bytes > resolved_quota:
        return f"Session blob quota exceeded: {current_total + additional_bytes} bytes would exceed {resolved_quota} byte limit."
    return None


@trust_boundary(
    tier=3,
    source="LLM-supplied create_blob-style tool arguments (filename / mime_type / content / optional description)",
    source_param="arguments",
    suppresses=("R1",),
    invariant="raises ToolArgumentError on a disallowed MIME type, unsanitizable filename, or non-UTF-8-encodable content; never coerces malformed arguments",
    test_ref="tests/integration/web/composer/test_inline_source_provenance.py::test_non_utf8_content_raises_tool_argument_error",
    test_fingerprint="0ba34e12e1e4291965b7a438789c3b877f8a9f1a2add72e9c8d1fe51628f3ab3",
)
def _prepare_blob_create(
    arguments: Mapping[str, Any],
    *,
    data_dir: str,
    session_id: str,
    creation_modality: CreationModality,
    created_from_message_id: str | None,
    creating_model_identifier: str | None = None,
    creating_model_version: str | None = None,
    creating_provider: str | None = None,
    creating_composer_skill_hash: str | None = None,
    creating_arguments_hash: str | None = None,
) -> _PreparedBlobCreate:
    """Validate a create_blob-style payload and allocate its storage path.

    Type guarantees on entry
    ------------------------
    Every reachable caller validates ``arguments`` via a Pydantic model
    BEFORE invoking this helper:

      * :func:`_execute_create_blob` — :class:`CreateBlobArgumentsModel`
        (``filename: str``, ``mime_type: str``, ``content: str`` +
        ``extra="forbid"``).
      * :func:`_execute_set_pipeline` inline-blob path — passes
        ``validated.source.inline_blob.model_dump()`` (via
        :class:`_InlineBlobModel`; same string-typed required fields
        + ``extra="forbid"``).

    The three ``isinstance(..., str)`` guards that previously sat at the
    top of this function are therefore unreachable — Pydantic rejects any
    non-string value with a structured :class:`pydantic.ValidationError`
    re-raised by the caller as :class:`ToolArgumentError` before this
    helper is invoked.  They are removed in the same commit that promotes
    ``set_pipeline`` so the dead-code surface does not linger past the
    wave that makes it dead (CLAUDE.md "No Legacy Code Policy").

    Semantic checks below this point (MIME allowlist, filename
    sanitisation, UTF-8 encodability) ARE NOT type checks — they enforce
    content-validity rules Pydantic cannot express — and remain.

    Provenance kwargs
    -----------------
    All callers MUST supply ``creation_modality`` and
    ``created_from_message_id``.  The five ``creating_*`` kwargs default
    to ``None`` and MUST be left as ``None`` for ``CreationModality.VERBATIM``;
    the three LLM-authored modalities require all five.  The DB-side
    CHECK ``ck_blobs_creating_llm_provenance_nullability`` rejects any
    other combination.  We do not duplicate the biconditional in Python
    — the constraint IS the validation, per the offensive-programming
    discipline in CLAUDE.md ("The CHECK constraint is the validation").
    """
    filename = arguments["filename"]
    mime_type = arguments["mime_type"]
    content = arguments["content"]

    if is_llm_authored_creation_modality(creation_modality) and created_from_message_id is None:
        raise AuditIntegrityError(
            "LLM-authored blob creation_modality requires created_from_message_id so the audit trail can walk back to the triggering chat message"
        )

    if mime_type not in _ALLOWED_BLOB_MIME_TYPES:
        # Tier-3 boundary: the LLM-supplied mime_type is not in the
        # operator-controlled allowlist. ToolArgumentError keeps the
        # leak-prevention discipline (no value field) — only the
        # allowlist itself appears in the LLM echo, never the rejected
        # value. Composer exception-channel discipline (CEC1) requires
        # ToolArgumentError here, not bare ValueError.
        allowed = ", ".join(sorted(_ALLOWED_BLOB_MIME_TYPES))
        raise ToolArgumentError(
            argument="mime_type",
            expected=f"one of: {allowed}",
            actual_type="str",
        )

    try:
        safe_filename = sanitize_filename(filename)
    except ValueError as exc:
        # Tier-3 boundary: filename failed sanitization (path traversal,
        # empty after strip, etc.). The underlying ValueError message
        # may echo the offending filename, so we wrap with
        # ToolArgumentError (no value field) and preserve the original
        # cause on __cause__ for auditors. CEC1 channel discipline.
        raise ToolArgumentError(
            argument="filename",
            expected="a sanitizable filename (no path separators, non-empty after stripping)",
            actual_type="str",
        ) from exc

    # UTF-8 encode guard: a Python ``str`` that contains
    # an unpaired surrogate code point (e.g. ``"\udc80"``) is a valid
    # ``str`` but is NOT encodable to UTF-8 — the underlying file write
    # would raise UnicodeEncodeError downstream and leave the audit layer
    # holding a half-written blob row.  Wrap as ToolArgumentError here
    # so the compose loop's ARG_ERROR routing handles it the same way as
    # disallowed MIME types and unsanitizable filenames (CEC1 channel).
    try:
        content_bytes = content.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ToolArgumentError(
            argument="content",
            expected="valid UTF-8 text",
            actual_type="str (contained non-encodable character, e.g. surrogate)",
        ) from exc
    file_hash = content_hash(content_bytes)
    blob_id = str(uuid4())
    return _PreparedBlobCreate(
        blob_id=blob_id,
        filename=safe_filename,
        mime_type=mime_type,
        content_bytes=content_bytes,
        content_hash=file_hash,
        storage_path=_blob_storage_path(data_dir, session_id, blob_id, safe_filename),
        description=arguments.get("description"),
        creation_modality=creation_modality,
        created_from_message_id=created_from_message_id,
        creating_model_identifier=creating_model_identifier,
        creating_model_version=creating_model_version,
        creating_provider=creating_provider,
        creating_composer_skill_hash=creating_composer_skill_hash,
        creating_arguments_hash=creating_arguments_hash,
    )


def _persist_prepared_blob_create(
    prepared: _PreparedBlobCreate,
    *,
    session_engine: Engine,
    session_id: str,
    max_blob_storage_per_session_bytes: int | None = None,
) -> str | None:
    """Persist a prepared blob create payload, returning a quota error if any."""
    prepared.storage_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.storage_path.write_bytes(prepared.content_bytes)

    now = datetime.now(UTC)
    try:
        with session_engine.begin() as conn:
            quota_error = _check_blob_quota(
                conn,
                session_id,
                len(prepared.content_bytes),
                quota_bytes=max_blob_storage_per_session_bytes,
            )
            if quota_error is not None:
                prepared.storage_path.unlink(missing_ok=True)
                return quota_error

            conn.execute(
                blobs_table.insert().values(
                    id=prepared.blob_id,
                    session_id=session_id,
                    filename=prepared.filename,
                    mime_type=prepared.mime_type,
                    size_bytes=len(prepared.content_bytes),
                    content_hash=prepared.content_hash,
                    storage_path=str(prepared.storage_path),
                    created_at=now,
                    created_by="assistant",
                    source_description=prepared.description,
                    status="ready",
                    # Inline-blob provenance. The
                    # DB-side CHECK ck_blobs_creating_llm_provenance_nullability
                    # rejects any combination where the modality and the
                    # five creating_* fields disagree on LLM authorship.
                    creation_modality=prepared.creation_modality.value,
                    created_from_message_id=prepared.created_from_message_id,
                    creating_model_identifier=prepared.creating_model_identifier,
                    creating_model_version=prepared.creating_model_version,
                    creating_provider=prepared.creating_provider,
                    creating_composer_skill_hash=prepared.creating_composer_skill_hash,
                    creating_arguments_hash=prepared.creating_arguments_hash,
                )
            )
    except Exception:
        prepared.storage_path.unlink(missing_ok=True)
        raise
    return None


def _blob_create_payload(prepared: _PreparedBlobCreate) -> BlobCreatePayload:
    """Return the LLM/audit-safe create_blob result payload."""
    return {
        "blob_id": prepared.blob_id,
        "filename": prepared.filename,
        "mime_type": prepared.mime_type,
        "size_bytes": len(prepared.content_bytes),
        "content_hash": prepared.content_hash,
    }


def _execute_create_blob(
    arguments: dict[str, Any],
    state: CompositionState,
    context: ToolContext,
) -> ToolResult:
    """Create a new blob (file) in the session from inline content.

    Uses the same storage layout and safety functions as BlobServiceImpl:
    sanitize_filename() for path traversal defence, content_hash() for
    SHA-256, per-session subdirectory, and atomic quota enforcement.

    Tier-3 boundary: ``arguments`` is an LLM-supplied dict.  Validated
    via :class:`CreateBlobArgumentsModel` (the single source of truth for
    the argument schema — supersedes the deleted
    ``_TOOL_REQUIRED_PATHS["create_blob"]`` entry in ``service.py``,
    rev-3 N7 / rev-4 M1).  On :class:`pydantic.ValidationError` we
    re-raise as :class:`ToolArgumentError` so the compose loop's
    ARG_ERROR routing at ``service.py:2480`` receives the right
    exception class.

    The validated ``model_dump()`` is then fed to ``_prepare_blob_create``
    which still performs the MIME-type allowlist check and
    :func:`sanitize_filename` traversal-defence — those are semantic
    Tier-3 checks (value-based) that Pydantic's type validation cannot
    express.
    """
    session_engine = context.session_engine
    session_id = context.session_id
    if session_engine is None or session_id is None:
        return _failure_result(state, "Blob tools require session context.")
    if context.data_dir is None:
        return _failure_result(state, "Blob tools require data_dir for storage.")

    try:
        validated = CreateBlobArgumentsModel.model_validate(arguments)
    except PydanticValidationError as exc:
        raise ToolArgumentError(
            argument="create_blob arguments",
            expected="object conforming to CreateBlobArgumentsModel",
            actual_type=type(exc).__name__,
        ) from exc

    # _prepare_blob_create still raises ToolArgumentError on semantic
    # Tier-3 violations (disallowed MIME type, un-sanitizable filename).
    # The Pydantic model catches type/shape violations; _prepare_blob_create
    # catches value-domain violations.  Both route via ToolArgumentError
    # to ARG_ERROR (CEC1 channel discipline).
    provenance = _blob_creation_provenance(validated.content, context)
    prepared = _prepare_blob_create(
        validated.model_dump(),
        data_dir=context.data_dir,
        session_id=session_id,
        creation_modality=provenance.creation_modality,
        created_from_message_id=context.user_message_id,
        creating_model_identifier=provenance.creating_model_identifier,
        creating_model_version=provenance.creating_model_version,
        creating_provider=provenance.creating_provider,
        creating_composer_skill_hash=provenance.creating_composer_skill_hash,
        creating_arguments_hash=provenance.creating_arguments_hash,
    )

    quota_error = _persist_prepared_blob_create(
        prepared,
        session_engine=session_engine,
        session_id=session_id,
        max_blob_storage_per_session_bytes=context.max_blob_storage_per_session_bytes,
    )
    if quota_error is not None:
        return _failure_result(state, quota_error)

    return _discovery_result(state, _blob_create_payload(prepared))


_CREATE_BLOB_DECLARATION = ToolDeclaration(
    name="create_blob",
    handler=_execute_create_blob,
    kind=ToolKind.BLOB_MUTATION,
    description=(
        "Create a new file (blob) from inline content. "
        "Use this to create seed input files (URLs, JSON, CSV snippets) "
        "mid-conversation without requiring manual upload."
    ),
    json_schema={
        "type": "object",
        "properties": {
            "filename": {
                "type": "string",
                "description": "Filename for the blob (e.g. 'urls.csv', 'seed.json').",
            },
            "mime_type": {
                "type": "string",
                "enum": [
                    "text/plain",
                    "application/json",
                    "text/csv",
                    "application/x-jsonlines",
                    "application/jsonl",
                    "text/jsonl",
                ],
                "description": "MIME type of the content.",
            },
            "content": {
                "type": "string",
                "description": "The file content as a string.",
            },
            "description": {
                "type": "string",
                "description": "Optional description of the file's purpose.",
            },
        },
        "required": ["filename", "mime_type", "content"],
        "additionalProperties": False,
    },
    blob_store_only=True,
)


# Per-session mutex guarding blob-file/DB consistency.
#
# ``_execute_update_blob`` reads the prior file content, writes new
# content, then opens a DB transaction that updates the size/hash
# metadata.  Two concurrent callers on the same session+blob can
# otherwise interleave these steps so that:
#
#   1. Thread A reads ``old_A`` from storage_path.
#   2. Thread A writes ``new_A``.
#   3. Thread B reads ``new_A`` (believing it to be ``old_B``).
#   4. Thread B writes ``new_B`` and commits the DB row with ``new_B``'s
#      size/hash.
#   5. Thread A's DB transaction fails.
#   6. Thread A's rollback writes ``old_A`` back to storage_path —
#      clobbering B's committed content.  File = ``old_A``, DB row =
#      ``new_B`` metadata: silent file/DB divergence with no signal.
#
# The composer tool layer is the only writer with this
# read→write→commit shape.  ``BlobServiceImpl.create_blob`` allocates a
# unique storage_path per blob, so it cannot hit this race; only the
# update path shares a storage_path between sequential writers.
#
# Serialising per-session (rather than per-blob) is deliberate: composer
# blob operations are low-frequency and a human typically interacts with
# one session at a time, so contention is benign.  Per-blob locking
# would require bookkeeping (reference counting, stale-lock GC) without
# a meaningful throughput win.
#
# The registry is a plain dict protected by a registry mutex.  A
# ``WeakValueDictionary`` cannot hold ``threading.Lock`` because the
# lock primitive does not support weak references.  Stale entries
# accumulate at roughly one entry per unique session_id observed during
# process lifetime (~150 bytes each) — negligible for the expected
# deployment (hundreds of sessions per server process).  If this ever
# becomes a concern, ``clear_session_blob_lock(session_id)`` below is
# the single-site cleanup hook; today there is no caller because
# session teardown is not yet observable from this module.
#
# PROCESS-LOCAL CORRECTNESS PRECONDITION:
# This registry holds Python ``threading.Lock`` objects — in-process
# mutexes with zero cross-process visibility.  The I4 blob-file/DB
# rollback race is serialised correctly ONLY because the web app
# refuses to start in multi-worker mode: see the startup guard in
# ``create_app`` (web/app.py) that raises ``RuntimeError`` on
# ``--workers > 1`` / ``-w > 1`` / ``--workers=N``.  If that guard is
# ever relaxed, every per-session lock becomes silently per-worker
# and two workers handling the same session can interleave
# blob-file writes and DB rollbacks.  The fix at that point is not
# to widen this registry but to move the lock into a cross-process
# coordination primitive (advisory DB lock / file lock / Redis) —
# changing this dict from process-local is a design-level decision
# that needs to be made alongside the multi-worker relaxation, not
# after it.
_SESSION_BLOB_LOCKS: dict[str, threading.Lock] = {}

_SESSION_BLOB_LOCKS_REGISTRY_MUTEX = threading.Lock()


def _session_blob_lock(session_id: str) -> threading.Lock:
    """Return the per-session mutex guarding blob-file/DB consistency.

    Double-checked locking: the fast path skips the registry mutex when
    the lock already exists; the registry mutex serialises the
    get-or-create race on first access so two concurrent callers on the
    same session_id cannot each install a different lock instance.
    """
    if session_id in _SESSION_BLOB_LOCKS:
        return _SESSION_BLOB_LOCKS[session_id]
    with _SESSION_BLOB_LOCKS_REGISTRY_MUTEX:
        if session_id not in _SESSION_BLOB_LOCKS:
            _SESSION_BLOB_LOCKS[session_id] = threading.Lock()
        return _SESSION_BLOB_LOCKS[session_id]


class _BlobQuotaExceededInTxn(Exception):
    """Internal sentinel raised inside the blob-update DB transaction.

    The quota check in ``_execute_update_blob`` must fire AFTER the file
    has been overwritten (so the size delta reflects the newly-written
    bytes) and INSIDE the DB transaction (so the delta uses the current
    row's size_bytes rather than a stale pre-transaction snapshot).
    When the quota is exceeded, the transaction must roll back AND the
    file must be restored from the ``old_content`` snapshot — the same
    rollback-write-with-add_note discipline the DB-failure path applies.

    Raising a distinct sentinel lets the outer ``except`` clauses model
    this cleanly:

    * ``except _BlobQuotaExceededInTxn`` handles the quota-exceeded
      flow: attempt the rollback write, attach add_note on rollback
      failure, then (if rollback succeeded) return the failure result.
    * ``except Exception as primary_exc`` handles DB-layer failures
      identically but re-raises ``primary_exc`` rather than returning a
      ToolResult.

    The two clauses share the rollback-with-add_note structure so the
    divergence-on-rollback-failure diagnostic is produced identically
    for both paths.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.user_message = message


class _BlobUpdateBlockedByActiveRun(Exception):
    """Internal sentinel raised inside the blob-update DB transaction.

    The active-run guard fires INSIDE ``session_engine.begin()`` so it
    shares SQLite's writer lock with concurrent run-creation attempts
    (see ``_execute_locked``) — any new run row that would reference
    this blob serialises behind the update transaction's guard check.
    When the guard trips, we must (a) roll the DB transaction back so
    no partial mutation leaks out, and (b) surface a tool-failure
    result rather than an exception so the compose loop treats the
    rejection as recoverable.

    Raising a distinct sentinel lets the outer handler distinguish
    three exit paths cleanly:

    * ``except _BlobUpdateBlockedByActiveRun`` — returns
      ``_failure_result`` (caller retries after the active run
      completes).
    * ``except _BlobQuotaExceededInTxn`` — returns a quota-specific
      ``_failure_result``.
    * ``except Exception`` — DB-layer or ``os.replace`` fault;
      re-raises after attaching rollback diagnostics on divergence.

    Keeping this separate from ``_BlobQuotaExceededInTxn`` is deliberate:
    the two conditions reach the same rollback-on-divergence handler
    but produce different user-facing failure messages.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.user_message = message


def _execute_update_blob(
    arguments: dict[str, Any],
    state: CompositionState,
    context: ToolContext,
) -> ToolResult:
    """Update the content of an existing blob.

    Tier-3 boundary: ``arguments`` is an LLM-supplied dict.  Validated
    via :class:`UpdateBlobArgumentsModel` (the single source of truth
    for the argument schema — supersedes the deleted
    ``_TOOL_REQUIRED_PATHS["update_blob"]`` entry in ``service.py``,
    rev-3 N7 / rev-4 M1).  On :class:`pydantic.ValidationError` we
    re-raise as :class:`ToolArgumentError` so the compose loop's
    ARG_ERROR routing at ``service.py:2480`` receives the right
    exception class.

    Validation precedence (file/lock safety).  ``model_validate`` MUST
    run BEFORE :func:`_session_blob_lock` is acquired and BEFORE any
    filesystem read/write.  The prior in-handler ``isinstance(content,
    str)`` guard documented this requirement at length — the same
    discipline still applies, now expressed structurally: Pydantic
    rejects a non-str ``content`` (or a missing ``blob_id``) before the
    handler reaches the tempfile/replace critical section, so the
    rollback-on-divergence path (which would otherwise issue an
    unnecessary filesystem write over an unmodified file) is never
    entered on a pure argument-validation failure.  ``_execute_create_blob``'s
    cleanup is ``unlink(missing_ok=True)`` (a genuine no-op); ``_execute_update_blob``'s
    is ``write_bytes(old_content)`` (a real filesystem mutation) — hence
    the validation MUST precede lock acquisition here, not merely
    precede the begin-transaction block.
    """
    session_engine = context.session_engine
    session_id = context.session_id
    if session_engine is None or session_id is None:
        return _failure_result(state, "Blob tools require session context.")

    try:
        validated = UpdateBlobArgumentsModel.model_validate(arguments)
    except PydanticValidationError as exc:
        raise ToolArgumentError(
            argument="update_blob arguments",
            expected="object conforming to UpdateBlobArgumentsModel",
            actual_type=type(exc).__name__,
        ) from exc

    blob_id = validated.blob_id
    blob_id_error = _blob_id_uuid_validation_error(blob_id)
    if blob_id_error is not None:
        return _failure_result(state, blob_id_error)
    content = validated.content
    if blob_id in _state_source_blob_refs(state):
        return _failure_result(
            state,
            f"Blob '{blob_id}' is currently bound as a pipeline source; create a new blob and rebind the source instead.",
        )
    provenance = _blob_creation_provenance(content, context)
    provenance_message_id = _blob_provenance_message_id(context.user_message_id)

    # Serialise the read→write→commit critical section across concurrent
    # composer-tool callers on this session.  See ``_session_blob_lock``'s
    # module-level docstring for the rollback-clobber race this closes
    # (I4).  The lock MUST be acquired BEFORE ``_sync_get_blob`` — a lock
    # scoped any tighter (e.g. only around the file write) would still
    # permit the interleave described in that docstring.
    with _session_blob_lock(session_id):
        blob = _sync_get_blob(session_engine, blob_id, session_id)
        if blob is None:
            return _failure_result(state, f"Blob '{blob_id}' not found.")

        storage_path = Path(blob["storage_path"])
        try:
            content_bytes = content.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ToolArgumentError(
                argument="update_blob content",
                expected="valid UTF-8 text",
                actual_type=type(exc).__name__,
            ) from exc
        file_hash = content_hash(content_bytes)
        new_size = len(content_bytes)

        # Snapshot the prior bytes BEFORE any filesystem mutation so the
        # post-replace divergence rollback (commit-failure window) can
        # restore them.  read_bytes() precedes tempfile creation so a
        # read-side OSError cannot orphan a tempfile.
        old_content = storage_path.read_bytes()

        # Write the NEW content to a sibling tempfile; ``os.replace``
        # swaps it in atomically only after the active-run guard, quota
        # check, and DB UPDATE have all succeeded.  Writing to a tempfile
        # (rather than overwriting storage_path up front as the pre-fix
        # code did) closes two audit-corruption windows:
        #
        # * Path-based sources reading the backing file mid-update would
        #   observe the new bytes against the stale DB content_hash —
        #   silent Tier-1 audit corruption.
        # * blob_ref sources recomputing the hash mid-update would raise
        #   a false-positive BlobIntegrityError because the on-disk
        #   bytes no longer match the stored hash.
        #
        # ``tempfile.mkstemp`` in ``storage_path.parent`` guarantees a
        # same-filesystem swap (required for POSIX ``os.replace``
        # atomicity).  The ``dot-prefix + .tmp`` suffix keeps stray
        # tempfiles (if any survive a kill) out of directory listings
        # that assume blob files are exactly ``{blob_id}_*`` — the
        # composer listing logic filters on that prefix.
        tmp_fd, tmp_name = tempfile.mkstemp(
            dir=storage_path.parent,
            prefix=f".{storage_path.name}.",
            suffix=".tmp",
        )
        tmp_path = Path(tmp_name)
        replaced = False
        try:
            with os.fdopen(tmp_fd, "wb") as tmp_file:
                tmp_file.write(content_bytes)

            try:
                with session_engine.begin() as conn:
                    # Active-run guard (two checks — mirror of the
                    # pattern in ``_execute_delete_blob``).  Lives
                    # INSIDE the transaction so SQLite's writer lock
                    # serialises it against concurrent run inserts —
                    # ``_execute_locked`` cannot slip a new run row
                    # past this guard because its INSERT would block on
                    # our transaction's lock.
                    #
                    # 1. Explicit link: ``blob_run_links`` already
                    #    points at an active run.
                    active_link = conn.execute(
                        select(blob_run_links_table)
                        .join(runs_table, blob_run_links_table.c.run_id == runs_table.c.id)
                        .where(blob_run_links_table.c.blob_id == blob_id)
                        .where(runs_table.c.status.in_(["pending", "running"]))
                    ).first()
                    if active_link is not None:
                        raise _BlobUpdateBlockedByActiveRun(
                            f"Blob '{blob_id}' is linked to active run '{active_link.run_id}' and cannot be updated."
                        )

                    # 2. Pre-link window: ``_execute_locked`` creates
                    #    the run record before ``link_blob_to_run``
                    #    inserts the link row.  During that gap the
                    #    explicit-link check sees nothing, but the
                    #    backing file is about to be read.  Scan the active
                    #    run's canonical pipeline dict for a ``blob_ref``
                    #    match OR a ``path``/``file`` that matches
                    #    ``storage_path``.
                    active_run = conn.execute(
                        select(*_ACTIVE_RUN_COMPOSITION_COLUMNS)
                        .join(
                            composition_states_table,
                            runs_table.c.state_id == composition_states_table.c.id,
                        )
                        .where(runs_table.c.session_id == session_id)
                        .where(runs_table.c.status.in_(["pending", "running"]))
                    ).first()
                    if active_run is not None and _composition_references_blob(
                        _active_run_pipeline_dict(active_run),
                        blob_id,
                        str(storage_path),
                    ):
                        raise _BlobUpdateBlockedByActiveRun(
                            f"Blob '{blob_id}' cannot be updated while active run '{active_run.run_id}' references it."
                        )

                    # Atomic quota check. The session row lock serializes
                    # same-session writers before ``size_bytes`` is re-read,
                    # so the delta reflects the current DB row rather than a
                    # pre-transaction snapshot (stale under writers that
                    # bypass the composer session lock — e.g.
                    # ``BlobServiceImpl`` paths that share the same
                    # session_engine).
                    _lock_session_for_blob_quota(conn, session_id)
                    current_size: int = conn.execute(
                        select(blobs_table.c.size_bytes).where(
                            blobs_table.c.id == blob_id,
                            blobs_table.c.session_id == session_id,
                        )
                    ).scalar_one()
                    size_delta = new_size - current_size
                    if size_delta > 0:
                        quota_error = _check_blob_quota(
                            conn,
                            session_id,
                            size_delta,
                            quota_bytes=context.max_blob_storage_per_session_bytes,
                            session_locked=True,
                        )
                        if quota_error is not None:
                            # Raising inside the ``with`` rolls the DB
                            # transaction back before the outer handler
                            # runs.  ``os.replace`` has not executed,
                            # so storage_path is still the prior bytes
                            # and no rollback write is required.
                            raise _BlobQuotaExceededInTxn(quota_error)

                    update_values = {
                        "size_bytes": new_size,
                        "content_hash": file_hash,
                    }
                    if provenance is not None:
                        update_values.update(
                            creation_modality=provenance.creation_modality.value,
                            created_from_message_id=provenance_message_id,
                            creating_model_identifier=provenance.creating_model_identifier,
                            creating_model_version=provenance.creating_model_version,
                            creating_provider=provenance.creating_provider,
                            creating_composer_skill_hash=provenance.creating_composer_skill_hash,
                            creating_arguments_hash=provenance.creating_arguments_hash,
                        )

                    conn.execute(
                        update(blobs_table)
                        .where(
                            blobs_table.c.id == blob_id,
                            blobs_table.c.session_id == session_id,
                        )
                        .values(**update_values)
                    )

                    # Atomic file swap — the final mutation before the
                    # with-block commit.  If ``os.replace`` raises,
                    # control exits the with-block via exception and
                    # the DB transaction rolls back — neither the file
                    # nor the DB row changes.  On success, control
                    # returns to the with-block which then commits;
                    # file and DB land in sync on the happy path.
                    #
                    # The residual divergence window is narrow and
                    # handled by the ``except Exception`` arm below:
                    # (os.replace succeeded) ∧ (commit subsequently
                    # failed).
                    os.replace(tmp_path, storage_path)
                    replaced = True
            except _BlobUpdateBlockedByActiveRun as blocked:
                # Guard rejected the update BEFORE ``os.replace`` ran;
                # DB transaction has rolled back, tempfile awaits
                # cleanup in the outer finally, storage_path is
                # unchanged.  Surface as tool-failure so the compose
                # loop treats the rejection as recoverable.
                return _failure_result(state, blocked.user_message)
            except _BlobQuotaExceededInTxn as quota_exc:
                # Quota raised BEFORE ``os.replace`` ran; storage_path
                # is unchanged.  If for any reason ``replaced`` is True
                # here (defensive — current ordering raises before
                # replace), restore old_content with add_note
                # discipline mirroring the DB-failure path so
                # divergence is surfaced, not silenced.
                if replaced:
                    try:
                        storage_path.write_bytes(old_content)
                    except OSError as rollback_exc:
                        quota_exc.add_note(
                            f"Rollback failed: could not restore prior content of {storage_path} "
                            f"({type(rollback_exc).__name__}: {rollback_exc}). "
                            f"Storage file and DB metadata for blob_id={blob_id!r} may now be "
                            f"inconsistent — the file may contain the new (uncommitted) bytes "
                            f"while the DB row retains the prior size_bytes/content_hash. "
                            f"Manual reconciliation required."
                        )
                        raise RuntimeError(
                            f"Blob quota rollback diverged for {blob_id!r}: "
                            f"{quota_exc.user_message}  Rollback write_bytes raised "
                            f"{type(rollback_exc).__name__}: {rollback_exc}. "
                            f"storage_path {storage_path!s} contains the uncommitted "
                            f"new content while the DB row retains the prior "
                            f"size_bytes/content_hash.  Manual reconciliation required."
                        ) from rollback_exc
                return _failure_result(state, quota_exc.user_message)
            except Exception as primary_exc:
                # DB-layer fault (commit OSError, UPDATE I/O error,
                # SQLAlchemy error) or ``os.replace`` fault.  If
                # ``replaced`` is True, ``os.replace`` has already
                # swapped the new bytes in and storage_path now
                # diverges from the (un-committed or about-to-fail) DB
                # row — restore from old_content.  Narrow the
                # rollback-error handler to OSError per
                # offensive-programming policy: programmer bugs
                # (TypeError, AttributeError, AssertionError) must
                # propagate so a broken rollback isn't silently
                # downgraded to a note.  Catching ``Exception`` (not
                # ``BaseException``) preserves KeyboardInterrupt /
                # SystemExit — asserted by
                # ``test_blob_rollback_does_not_catch_keyboard_interrupt``.
                if replaced:
                    try:
                        storage_path.write_bytes(old_content)
                    except OSError as rollback_exc:
                        primary_exc.add_note(
                            f"Rollback failed: could not restore prior content of {storage_path} "
                            f"({type(rollback_exc).__name__}: {rollback_exc}). "
                            f"Storage file and DB metadata for blob_id={blob_id!r} may now be "
                            f"inconsistent — the file may contain the new (uncommitted) bytes "
                            f"while the DB row retains the prior size_bytes/content_hash. "
                            f"Manual reconciliation required."
                        )
                raise
        finally:
            # Unconditional tempfile cleanup.  On the happy path
            # ``os.replace`` moves the inode and ``tmp_path`` vanishes
            # (unlink becomes a no-op via missing_ok).  On every
            # failure path the tempfile still exists and must be
            # removed to prevent inode exhaustion and leakage of
            # uncommitted content to any directory listing.
            tmp_path.unlink(missing_ok=True)

        return _discovery_result(
            state,
            {
                "blob_id": blob_id,
                "filename": blob["filename"],
                "mime_type": blob["mime_type"],
                "size_bytes": len(content_bytes),
                "content_hash": file_hash,
            },
        )


_UPDATE_BLOB_DECLARATION = ToolDeclaration(
    name="update_blob",
    handler=_execute_update_blob,
    kind=ToolKind.BLOB_MUTATION,
    description="Update the content of an existing blob (file). Overwrites the file content while preserving metadata.",
    json_schema={
        "type": "object",
        "properties": {
            "blob_id": {
                "type": "string",
                "description": "ID of the blob to update.",
            },
            "content": {
                "type": "string",
                "description": "New file content.",
            },
        },
        "required": ["blob_id", "content"],
        "additionalProperties": False,
    },
    blob_store_only=True,
)


def _execute_delete_blob(
    arguments: dict[str, Any],
    state: CompositionState,
    context: ToolContext,
) -> ToolResult:
    """Delete a blob and its storage file."""
    session_engine = context.session_engine
    session_id = context.session_id
    if session_engine is None or session_id is None:
        return _failure_result(state, "Blob tools require session context.")

    blob_id = arguments["blob_id"]
    blob_id_error = _blob_id_uuid_validation_error(blob_id)
    if blob_id_error is not None:
        return _failure_result(state, blob_id_error)

    blob = _sync_get_blob(session_engine, blob_id, session_id)
    if blob is None:
        return _failure_result(state, f"Blob '{blob_id}' not found.")

    storage_path = Path(blob["storage_path"])
    tombstone_path: Path | None = None

    try:
        with session_engine.begin() as conn:
            # Active-run guard (two checks):
            #
            # 1. Explicit link: blob_run_links already points at an active run.
            active_link = conn.execute(
                select(blob_run_links_table)
                .join(runs_table, blob_run_links_table.c.run_id == runs_table.c.id)
                .where(blob_run_links_table.c.blob_id == blob_id)
                .where(runs_table.c.status.in_(["pending", "running"]))
            ).first()
            if active_link is not None:
                return _failure_result(
                    state,
                    f"Blob '{blob_id}' is linked to active run '{active_link.run_id}' and cannot be deleted.",
                )

            # 2. Pre-link window: _execute_locked() creates the run record before
            #    link_blob_to_run() inserts the blob_run_links row.  During that
            #    gap the explicit-link check above sees nothing, but the backing
            #    file is about to be needed.
            #
            #    Scoped to THIS blob: join runs → composition_states and check
            #    whether the active run's canonical pipeline dict references
            #    this blob via blob_ref OR via a path/file matching this
            #    blob's storage_path.
            #    Runs whose source doesn't touch this blob must not block
            #    unrelated blob deletions.
            active_run = conn.execute(
                select(*_ACTIVE_RUN_COMPOSITION_COLUMNS)
                .join(
                    composition_states_table,
                    runs_table.c.state_id == composition_states_table.c.id,
                )
                .where(runs_table.c.session_id == session_id)
                .where(runs_table.c.status.in_(["pending", "running"]))
            ).first()
            if active_run is not None and _composition_references_blob(
                _active_run_pipeline_dict(active_run),
                blob_id,
                blob["storage_path"],
            ):
                return _failure_result(
                    state,
                    f"Blob '{blob_id}' cannot be deleted while active run '{active_run.run_id}' references it.",
                )

            # Move the file to a tombstone path before the DB delete so a
            # later SQL/commit failure can restore it atomically. This avoids
            # leaving a live blobs row pointing at missing bytes.
            if storage_path.exists():
                tombstone_path = storage_path.with_name(f".{storage_path.name}.delete-{uuid4().hex}")
                os.replace(storage_path, tombstone_path)

            # Delete record — include session_id filter for defence in depth
            conn.execute(
                delete(blobs_table).where(
                    blobs_table.c.id == blob_id,
                    blobs_table.c.session_id == session_id,
                )
            )
    except Exception as primary_exc:
        if tombstone_path is not None and tombstone_path.exists():
            try:
                os.replace(tombstone_path, storage_path)
            except OSError as rollback_exc:
                primary_exc.add_note(
                    f"Rollback failed: could not restore deleted blob file {storage_path} from tombstone "
                    f"{tombstone_path} ({type(rollback_exc).__name__}: {rollback_exc}). "
                    f"Blob row and storage may now diverge; manual reconciliation required."
                )
        raise

    if tombstone_path is not None and tombstone_path.exists():
        try:
            tombstone_path.unlink()
        except OSError as cleanup_exc:
            raise RuntimeError(
                f"Blob '{blob_id}' metadata was deleted but tombstone cleanup failed for {tombstone_path}: "
                f"{type(cleanup_exc).__name__}: {cleanup_exc}"
            ) from cleanup_exc

    return _discovery_result(state, {"blob_id": blob_id, "deleted": True})


_DELETE_BLOB_DECLARATION = ToolDeclaration(
    name="delete_blob",
    handler=_execute_delete_blob,
    kind=ToolKind.BLOB_MUTATION,
    description="Delete a blob (file) and its storage.",
    json_schema={
        "type": "object",
        "properties": {
            "blob_id": {
                "type": "string",
                "description": "ID of the blob to delete.",
            },
        },
        "required": ["blob_id"],
        "additionalProperties": False,
    },
    blob_store_only=True,
)


def _verify_blob_content_integrity(blob: BlobToolRecord, data: bytes) -> None:
    """Verify on-disk blob bytes match the stored content_hash.

    Tier-1 invariant: a ``ready`` blob's stored ``content_hash`` is
    enforced non-NULL by the ``ck_blobs_ready_hash`` CHECK constraint
    at write time. Reading NULL here is therefore a DB-integrity
    anomaly (someone bypassed the constraint, the row was tampered
    with, or the constraint is missing in this database). A SHA-256
    mismatch between recomputed bytes and stored hash is filesystem
    corruption, tampering, or a write-path bug.

    Both conditions ESCALATE via ``AuditIntegrityError`` /
    ``BlobIntegrityError`` rather than degrading to a soft result;
    silently passing through unverified bytes would let the audit
    trail confidently record decisions made on garbage.
    """
    _verify_blob_content_hash(blob, content_hash(data))


def _verify_blob_content_hash(blob: BlobToolRecord, actual_hash: str) -> None:
    """Verify a precomputed SHA-256 digest against a blob row."""
    blob_id = blob["id"]
    stored_hash = blob["content_hash"]
    if stored_hash is None:
        raise AuditIntegrityError(f"Tier 1: ready blob {blob_id} has NULL content_hash — DB integrity anomaly, cannot verify")
    if not hmac.compare_digest(actual_hash, stored_hash):
        raise BlobIntegrityError(blob_id, expected=stored_hash, actual=actual_hash)


def _execute_get_blob_content(
    arguments: dict[str, Any],
    state: CompositionState,
    context: ToolContext,
) -> ToolResult:
    """Retrieve the content of a blob for inspection.

    Mirrors the three Tier-1 guards enforced by
    ``BlobServiceImpl.read_blob_content`` so the composer read path and
    the HTTP read path apply the same invariants:

    1. **Lifecycle guard** — only ``ready`` blobs have finalised,
       trustworthy content.  ``pending`` blobs may be partial writes;
       ``error`` blobs belong to failed runs whose output is not
       authoritative.  Returned as a ``_failure_result`` so the
       compose loop can surface a helpful message to the LLM.
    2. **Integrity verification** — recompute SHA-256 of the on-disk
       bytes and compare (``hmac.compare_digest`` — constant-time) to
       the stored ``content_hash``.  A mismatch is a Tier-1 anomaly
       (our hash, our file) indicating filesystem corruption,
       tampering, or a write-path bug; it must ESCALATE via
       ``BlobIntegrityError``, not degrade to a tool-failure result.
       Implemented by ``_verify_blob_content_integrity`` (shared with
       ``_execute_inspect_source`` and ``compute_proof_diagnostics``).
    3. **Decode safety** — the MIME allowlist admits encodings other
       than UTF-8 (``text/csv`` is frequently latin-1 in the wild).
       ``UnicodeDecodeError`` is converted to a ``_failure_result``
       so the tool dispatcher is not crashed by admissible-but-
       undecodable content.

    The canonical path — ``BlobServiceImpl.read_blob_content`` — is
    async and engine-bound, so the guards are mirrored inline rather
    than shared via a common helper.  Any drift between this function
    and ``BlobServiceImpl.read_blob_content`` is caught by
    ``TestGetBlobContentGuards`` at CI time.
    """
    session_engine = context.session_engine
    session_id = context.session_id
    if session_engine is None or session_id is None:
        return _failure_result(state, "Blob tools require session context.")

    blob_id = arguments["blob_id"]
    blob_id_error = _blob_id_uuid_validation_error(blob_id)
    if blob_id_error is not None:
        return _failure_result(state, blob_id_error)
    blob = _sync_get_blob(session_engine, blob_id, session_id)
    if blob is None:
        return _failure_result(state, f"Blob '{blob_id}' not found.")

    # Guard 1 — lifecycle.  Pending/error blobs are not readable.
    blob_status = blob["status"]
    if blob_status != "ready":
        return _failure_result(
            state,
            f"Blob '{blob_id}' is not readable — status is '{blob_status}', expected 'ready'.",
        )

    storage_path = Path(blob["storage_path"])
    if not storage_path.exists():
        return _failure_result(state, f"Blob storage file missing for '{blob_id}'.")

    data = storage_path.read_bytes()

    # Guard 2 — integrity.  Shared helper: NULL stored_hash escalates
    # via AuditIntegrityError, mismatch via BlobIntegrityError.
    _verify_blob_content_integrity(blob, data)

    # Guard 3 — decode safety.  Non-UTF-8 bytes are a Tier-3 external
    # input condition (the operator supplied content in an encoding we
    # cannot losslessly round-trip to the LLM); surface as
    # tool-failure so the compose loop treats it as recoverable rather
    # than raising an unhandled exception out of the dispatcher.
    try:
        content = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        return _failure_result(
            state,
            f"Blob '{blob_id}' is not valid UTF-8 text ({exc.reason} at byte offset {exc.start}).",
        )

    # Truncate very large content to avoid overwhelming the LLM context
    max_chars = 50_000
    truncated = len(content) > max_chars
    if truncated:
        content = content[:max_chars]

    return _discovery_result(
        state,
        {
            "blob_id": blob_id,
            "filename": blob["filename"],
            "mime_type": blob["mime_type"],
            "content": content,
            "truncated": truncated,
            "size_bytes": blob["size_bytes"],
        },
    )


_GET_BLOB_CONTENT_DECLARATION = ToolDeclaration(
    name="get_blob_content",
    handler=_execute_get_blob_content,
    kind=ToolKind.BLOB_DISCOVERY,
    description="Retrieve the content of a blob (file) for inspection. Large files are truncated to 50,000 characters.",
    json_schema={
        "type": "object",
        "properties": {
            "blob_id": {
                "type": "string",
                "description": "ID of the blob to read.",
            },
        },
        "required": ["blob_id"],
        "additionalProperties": False,
    },
)


# ``_BLOB_STORE_ONLY_MUTATION_TOOL_NAMES`` and the matching predicate
# ``is_blob_store_only_mutation_tool`` are declared in
# ``elspeth.web.composer.tools.discovery``. The dispatcher carries the full
# ``ToolContext`` (including ``max_blob_storage_per_session_bytes`` and
# ``user_message_id``) to every handler, so there is no per-tool kwarg-shape
# gate to maintain at the declaration site.


TOOLS_IN_MODULE: tuple[ToolDeclaration, ...] = (
    _LIST_BLOBS_DECLARATION,
    _LIST_COMPOSER_BLOBS_DECLARATION,
    _GET_BLOB_METADATA_DECLARATION,
    _GET_BLOB_CONTENT_DECLARATION,
    _CREATE_BLOB_DECLARATION,
    _UPDATE_BLOB_DECLARATION,
    _DELETE_BLOB_DECLARATION,
    _WIRE_BLOB_INLINE_REF_DECLARATION,
)
"""Every tool declared in this module, in stable order.

``_dispatch.py`` aggregates this tuple from every plane to build the
registered-tool universe. Tests that import this module directly see the
same TOOLS_IN_MODULE that production sees; the aggregation logic lives at
the consumer site, not in a module-level side effect."""
