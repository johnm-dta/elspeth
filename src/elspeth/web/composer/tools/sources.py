"""Composer sources plane — source-spec tool handlers and source-from-blob bridge."""

from __future__ import annotations

import csv
import io
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, TypedDict
from uuid import UUID

from pydantic import ValidationError as PydanticValidationError
from sqlalchemy import Engine, select

from elspeth.contracts.composer_interpretation import InterpretationKind
from elspeth.contracts.enums import CreationModality, is_llm_authored_creation_modality
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.freeze import freeze_fields
from elspeth.web.catalog.protocol import CatalogService
from elspeth.web.composer.protocol import ToolArgumentError
from elspeth.web.composer.redaction import (
    PatchSourceOptionsArgumentsModel,
    SetSourceArgumentsModel,
    SetSourceFromBlobArgumentsModel,
)
from elspeth.web.composer.source_inspection import (
    facts_to_dict,
    inspect_blob_content,
)
from elspeth.web.composer.state import (
    CompositionState,
    SourceSpec,
)
from elspeth.web.composer.tools._common import (
    _DEFAULT_SOURCE_VALIDATION_FAILURE,
    _SOURCE_VALIDATION_FAILURE_DESCRIPTION,
    ToolContext,
    ToolResult,
    _apply_merge_patch,
    _attach_post_call_hints,
    _credential_wiring_contract_failure,
    _discovery_result,
    _failure_result,
    _mutation_result,
    _options_with_pending_requirement,
    _pending_interpretation_requirement,
    _prevalidate_source,
    _validate_plugin_name,
    _validate_source_path,
    _vf_destination_note,
)
from elspeth.web.composer.tools.blobs import (
    BlobToolRecord,
    _blob_row_to_tool_dict,
    _PreparedBlobCreate,
    _sync_get_blob,
    _verify_blob_content_integrity,
)
from elspeth.web.composer.tools.declarations import (
    ToolDeclaration,
    ToolKind,
)
from elspeth.web.interpretation_state import SOURCE_AUTHORING_KEY, SourceAuthoringMetadata
from elspeth.web.sessions.models import blobs_table


def _handle_list_sources(
    arguments: dict[str, Any],
    state: CompositionState,
    context: ToolContext,
) -> ToolResult:
    return _discovery_result(state, context.catalog.list_sources())


_LIST_SOURCES_DECLARATION = ToolDeclaration(
    name="list_sources",
    handler=_handle_list_sources,
    kind=ToolKind.DISCOVERY,
    description="List available source plugins with name and summary.",
    json_schema={"type": "object", "properties": {}, "required": []},
    cacheable=True,
)


def _handle_set_source(
    arguments: dict[str, Any],
    state: CompositionState,
    context: ToolContext,
) -> ToolResult:
    result = _execute_set_source(arguments, state, context)
    if not result.success:
        return result
    source_name = SetSourceArgumentsModel.model_validate(arguments).source_name
    if source_name not in result.updated_state.sources:
        return result
    source = result.updated_state.sources[source_name]
    return _attach_post_call_hints(
        result,
        context.catalog,
        plugin_type="source",
        tool_name="set_source",
        plugin_name=source.plugin,
        config_snapshot=source.options,
    )


_SET_SOURCE_DECLARATION = ToolDeclaration(
    name="set_source",
    handler=_handle_set_source,
    kind=ToolKind.MUTATION,
    description="Set or replace a named pipeline source.",
    json_schema={
        "type": "object",
        "properties": {
            "source_name": {
                "type": "string",
                "description": "Stable source root name. Defaults to 'source' for legacy single-source pipelines.",
            },
            "plugin": {"type": "string", "description": "Source plugin name."},
            "on_success": {
                "type": "string",
                "description": (
                    "Connection-name string this source PUBLISHES. Some downstream consumer "
                    "(transform 'input' or output 'sink_name') MUST equal this value for wiring "
                    "to resolve. The runtime matches strings, not graph topology — pick any "
                    "name unique within the pipeline; it does not need to be the downstream "
                    "node's id."
                ),
                "examples": ["raw_url_rows", "csv_rows", "fetched_text"],
            },
            "options": {"type": "object", "description": "Plugin-specific config."},
            "on_validation_failure": {
                "type": "string",
                "description": _SOURCE_VALIDATION_FAILURE_DESCRIPTION,
            },
        },
        "required": ["plugin", "on_success", "options", "on_validation_failure"],
    },
    augments_on_failure=True,
)


_MIME_TO_SOURCE: dict[str, tuple[str, dict[str, str]]] = {
    "text/csv": ("csv", {}),
    "application/json": ("json", {}),
    "application/x-jsonlines": ("json", {"format": "jsonl"}),
    "application/jsonl": ("json", {"format": "jsonl"}),
    "text/jsonl": ("json", {"format": "jsonl"}),
    "text/plain": ("text", {}),
}


class SourceBlobPayload(TypedDict):
    """LLM/audit-safe source blob metadata for set_pipeline/set_source_from_blob."""

    blob_id: str
    filename: str
    mime_type: str
    size_bytes: int
    content_hash: str | None


@dataclass(frozen=True, slots=True)
class _ResolvedSourceBlob:
    plugin: str
    options: Mapping[str, Any]
    payload: SourceBlobPayload
    creation_modality: CreationModality

    def __post_init__(self) -> None:
        # ``options`` is the resolved-source pipeline-options mapping; it
        # may carry nested dicts/lists from the composer YAML and is
        # mutable through the attribute reference without a freeze guard,
        # defeating ``frozen=True``. ``payload`` is a SourceBlobPayload
        # TypedDict whose declared fields are all scalars (str / int /
        # str | None) — the dict itself is a container so we deep-freeze
        # both for symmetry rather than relying on caller discipline to
        # keep payload scalar-only forever.
        freeze_fields(self, "options", "payload")


def _source_blob_payload(blob: BlobToolRecord) -> SourceBlobPayload:
    """Return source-blob metadata without leaking storage_path."""
    return {
        "blob_id": blob["id"],
        "filename": blob["filename"],
        "mime_type": blob["mime_type"],
        "size_bytes": blob["size_bytes"],
        "content_hash": blob["content_hash"],
    }


def _source_authoring_options(
    creation_modality: CreationModality,
    content_hash_value: str | None,
) -> dict[str, SourceAuthoringMetadata]:
    """Return source authoring metadata for LLM-authored blob-backed sources."""
    if not is_llm_authored_creation_modality(creation_modality):
        return {}
    if content_hash_value is None:
        raise AuditIntegrityError("LLM-authored blob-backed source requires non-null content_hash for source authoring metadata")
    return {
        SOURCE_AUTHORING_KEY: {
            "modality": creation_modality.value,
            "content_hash": content_hash_value,
            "review_event_id": None,
            "resolved_kind": None,
        }
    }


def _source_blob_review_user_term(*, mime_type: str, content: str) -> str:
    """Choose the stable Class 2 review term for an LLM-authored source blob."""
    if mime_type != "text/csv":
        return "inline_source_data"
    rows = csv.reader(io.StringIO(content))
    header = next(rows, ())
    if len(header) == 1 and header[0].strip().lower() == "url":
        return "inline_source_url_list"
    return "inline_source_data"


def _options_with_source_blob_review(
    options: Mapping[str, Any],
    *,
    mime_type: str,
    content: str,
) -> Mapping[str, Any]:
    """Ensure LLM-authored blob-backed sources carry a Class 2 review gate."""
    if SOURCE_AUTHORING_KEY not in options:
        return options
    user_term = _source_blob_review_user_term(mime_type=mime_type, content=content)
    requirement = _pending_interpretation_requirement(
        requirement_id=f"source_review:{user_term}",
        kind=InterpretationKind.INVENTED_SOURCE,
        user_term=user_term,
        draft=content,
    )
    return _options_with_pending_requirement(
        options,
        requirement=requirement,
        replace_kind=InterpretationKind.INVENTED_SOURCE,
    )


def _manual_source_authoring_error(*, tool_name: str) -> str:
    """Return the source-options error for caller-supplied source_authoring."""
    return (
        f"{tool_name} must not be called with '{SOURCE_AUTHORING_KEY}' in source.options. "
        "source_authoring is reserved for ELSPETH's blob provenance writer and is stamped only when "
        "a source is bound through set_source_from_blob, source.blob_id, or source.inline_blob."
    )


def _reject_manual_source_authoring(
    options: Mapping[str, Any],
    *,
    tool_name: str = "set_source_from_blob",
) -> str | None:
    """Reject caller-supplied source_authoring outside provenance stamping."""
    if SOURCE_AUTHORING_KEY not in options:
        return None
    return _manual_source_authoring_error(tool_name=tool_name)


def _source_component_id(source_name: str) -> str:
    """Return the legacy/default or named source component identifier."""
    return "source" if source_name == "source" else f"source:{source_name}"


def _resolve_source_blob(
    *,
    blob_id: str,
    explicit_plugin: str | None,
    caller_options: Mapping[str, Any],
    on_validation_failure: str,
    state: CompositionState,
    catalog: CatalogService,
    session_engine: Engine | None,
    session_id: str | None,
    tool_name: str = "set_source_from_blob",
) -> _ResolvedSourceBlob | ToolResult:
    """Resolve an existing ready blob into authoritative source options."""
    if session_engine is None or session_id is None:
        return _failure_result(state, "Blob tools require session context.")
    manual_authoring_error = _reject_manual_source_authoring(caller_options, tool_name=tool_name)
    if manual_authoring_error is not None:
        return _failure_result(state, manual_authoring_error)
    blob = _sync_get_blob(session_engine, blob_id, session_id)
    if blob is None:
        return _failure_result(state, f"Blob '{blob_id}' not found.")

    if blob["status"] != "ready":
        return _failure_result(state, f"Blob is not ready (status: {blob['status']}).")

    mime_extra: dict[str, str] = {}
    if explicit_plugin:
        plugin = explicit_plugin
    elif blob["mime_type"] not in _MIME_TO_SOURCE:
        return _failure_result(
            state,
            f"Cannot infer source plugin for MIME type '{blob['mime_type']}'. Please specify the 'plugin' parameter explicitly.",
        )
    else:
        plugin, mime_extra = _MIME_TO_SOURCE[blob["mime_type"]]

    try:
        catalog.get_schema("source", plugin)
    except (ValueError, KeyError) as exc:
        return _failure_result(state, f"Unknown source plugin '{plugin}': {exc}")

    creation_modality = CreationModality(blob["creation_modality"])
    merged_options: Mapping[str, Any] = {
        **caller_options,
        **mime_extra,
        "path": blob["storage_path"],
        "blob_ref": blob["id"],
        "mode": "bind_source",
        **_source_authoring_options(creation_modality, blob["content_hash"]),
    }
    if SOURCE_AUTHORING_KEY in merged_options:
        storage_path = Path(blob["storage_path"])
        if not storage_path.exists():
            return _failure_result(state, f"Blob storage file missing for '{blob_id}'.")
        data = storage_path.read_bytes()
        _verify_blob_content_integrity(blob, data)
        try:
            content = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            return _failure_result(
                state,
                f"Blob '{blob_id}' is not valid UTF-8 text ({exc.reason} at byte offset {exc.start}).",
            )
        merged_options = _options_with_source_blob_review(
            merged_options,
            mime_type=blob["mime_type"],
            content=content,
        )
    prevalidation_error = _prevalidate_source(plugin, merged_options, on_validation_failure)
    if prevalidation_error is not None:
        return _failure_result(state, prevalidation_error)

    return _ResolvedSourceBlob(
        plugin=plugin,
        options=merged_options,
        payload=_source_blob_payload(blob),
        creation_modality=creation_modality,
    )


def _manual_source_blob_ref_error(*, tool_name: str, inline_blob_supported: bool = False) -> str:
    """Return the source-options error for tools that reject manual blob_ref."""
    if inline_blob_supported:
        bind_path = "set_source_from_blob, source.blob_id, or source.inline_blob"
    else:
        bind_path = "set_source_from_blob"
    return (
        f"Use {bind_path} to bind a blob to the source. "
        f"{tool_name} must not be called with 'blob_ref' in source.options "
        "because it cannot enforce that 'path' equals the blob's canonical storage_path."
    )


def _reject_manual_source_blob_ref(
    options: Mapping[str, Any],
    *,
    tool_name: str,
    inline_blob_supported: bool = False,
) -> str | None:
    """Reject caller-supplied blob_ref outside authoritative blob-binding tools."""
    if "blob_ref" not in options:
        return None
    return _manual_source_blob_ref_error(tool_name=tool_name, inline_blob_supported=inline_blob_supported)


def _execute_set_source(
    args: dict[str, Any],
    state: CompositionState,
    context: ToolContext,
) -> ToolResult:
    """Set or replace the pipeline source.

    Tier-3 boundary: ``args`` is an LLM-supplied dict.  Validated via the
    Pydantic redaction-bearing model :class:`SetSourceArgumentsModel` (the
    single source of truth for the argument schema — supersedes the
    deleted ``_TOOL_REQUIRED_PATHS["set_source"]`` entry in ``service.py``,
    rev-3 N7 / rev-4 M1).

    On :class:`pydantic.ValidationError` the handler re-raises as
    :class:`ToolArgumentError` so the compose loop's ARG_ERROR routing at
    ``service.py:2480`` receives the right exception class.  A bare
    ``ValidationError`` would escape into the catch-all
    (``ComposerPluginCrashError`` → HTTP 500) — wrong disposition for
    Tier-3 input.  Pattern: ``tools.py:2668, 2761, 2767, 2773, 2787, 2801``.
    """
    try:
        validated = SetSourceArgumentsModel.model_validate(args)
    except PydanticValidationError as exc:
        raise ToolArgumentError(
            argument="set_source arguments",
            expected="object conforming to SetSourceArgumentsModel",
            actual_type=type(exc).__name__,
        ) from exc

    plugin = validated.plugin
    options = validated.options
    source_name = validated.source_name

    # Validate plugin exists in catalog
    plugin_error = _validate_plugin_name(context.catalog, "source", plugin)
    if plugin_error is not None:
        return _failure_result(state, plugin_error)

    # Reject manual blob_ref injection.  The canonical write path for a
    # blob-backed source is set_source_from_blob, which forces the path to
    # the blob's authoritative storage_path.  set_source with a hand-crafted
    # blob_ref + path lets the caller persist a path that disagrees with the
    # blob's canonical storage_path, breaking runtime resolution and
    # composer/runtime agreement.  See elspeth-07089fbaa3.
    manual_blob_ref_error = _reject_manual_source_blob_ref(options, tool_name="set_source")
    if manual_blob_ref_error is not None:
        return _failure_result(state, manual_blob_ref_error)
    manual_authoring_error = _reject_manual_source_authoring(options, tool_name="set_source")
    if manual_authoring_error is not None:
        return _failure_result(state, manual_authoring_error)
    credential_error = _credential_wiring_contract_failure(
        state,
        component_id=_source_component_id(source_name),
        component_type="source",
        options=options,
    )
    if credential_error is not None:
        return credential_error

    # S2: Validate source path allowlist
    path_error = _validate_source_path(options, context.data_dir)
    if path_error is not None:
        return _failure_result(state, path_error)

    on_vf = validated.on_validation_failure
    prevalidation_error = _prevalidate_source(plugin, options, on_vf)
    if prevalidation_error is not None:
        return _failure_result(state, prevalidation_error)

    source = SourceSpec(
        plugin=plugin,
        on_success=validated.on_success,
        options=options,
        on_validation_failure=on_vf,
    )
    new_state = state.with_named_source(source_name, source)
    affected = (_source_component_id(source_name),)
    return _mutation_result(new_state, affected, data=_vf_destination_note(new_state, on_vf))


def _execute_set_source_from_blob(
    arguments: dict[str, Any],
    state: CompositionState,
    context: ToolContext,
) -> ToolResult:
    """Bind the pipeline source to an existing blob.

    Tier-3 boundary: ``arguments`` is an LLM-supplied dict.  Validated
    via :class:`SetSourceFromBlobArgumentsModel` (the single source of
    truth for the argument schema — supersedes the deleted
    ``_TOOL_REQUIRED_PATHS["set_source_from_blob"]`` entry in
    ``service.py``, rev-3 N7 / rev-4 M1).  On
    :class:`pydantic.ValidationError` we re-raise as
    :class:`ToolArgumentError` so the compose loop's ARG_ERROR routing
    at ``service.py:2480`` receives the right exception class.

    The prior in-handler ``isinstance(caller_options, dict)`` guard at
    this site is superseded by the Pydantic model's ``options: dict[str,
    Any]`` validation: a non-dict (or missing-required-fields) input now
    raises a structured ValidationError that the handler re-raises as
    ToolArgumentError before any blob-lookup work is done.

    Optional-field semantics (mirrors the JSON schema's `required`):
      * ``options`` defaults to ``{}`` (matches the prior
        ``arguments.get("options", {})``).
      * ``plugin`` and ``on_validation_failure`` remain ``str | None``
        so the handler can distinguish operator-omitted from
        operator-specified.  ``on_validation_failure`` None falls back
        to ``_DEFAULT_SOURCE_VALIDATION_FAILURE`` ("discard") at the
        seam below, matching the prior ``arguments.get(...)`` default.
    """
    try:
        validated = SetSourceFromBlobArgumentsModel.model_validate(arguments)
    except PydanticValidationError as exc:
        raise ToolArgumentError(
            argument="set_source_from_blob arguments",
            expected="object conforming to SetSourceFromBlobArgumentsModel",
            actual_type=type(exc).__name__,
        ) from exc

    on_vf = validated.on_validation_failure if validated.on_validation_failure is not None else _DEFAULT_SOURCE_VALIDATION_FAILURE
    resolved = _resolve_source_blob(
        blob_id=validated.blob_id,
        explicit_plugin=validated.plugin,
        caller_options=validated.options,
        on_validation_failure=on_vf,
        state=state,
        catalog=context.catalog,
        session_engine=context.session_engine,
        session_id=context.session_id,
        tool_name="set_source_from_blob",
    )
    if isinstance(resolved, ToolResult):
        return resolved

    source = SourceSpec(
        plugin=resolved.plugin,
        on_success=validated.on_success,
        options=resolved.options,
        on_validation_failure=on_vf,
    )
    new_state = state.with_source(source)
    data = _vf_destination_note(new_state, on_vf) or {}
    return _mutation_result(new_state, ("source",), data={**data, "source_blob": resolved.payload})


_SET_SOURCE_FROM_BLOB_DECLARATION = ToolDeclaration(
    name="set_source_from_blob",
    handler=_execute_set_source_from_blob,
    kind=ToolKind.BLOB_MUTATION,
    description=(
        "Wire a blob as the pipeline source. Resolves the blob's storage path "
        "internally and infers the source plugin from its MIME type. "
        "Use 'options' for plugin-specific config (e.g., 'column' and 'schema' for text sources)."
    ),
    json_schema={
        "type": "object",
        "properties": {
            "blob_id": {"type": "string", "description": "Blob ID to use as source."},
            "plugin": {
                "type": "string",
                "description": "Source plugin override (e.g. 'csv'). Inferred from MIME type if omitted.",
            },
            "on_success": {
                "type": "string",
                "description": (
                    "Connection-name string the source PUBLISHES. Some downstream consumer "
                    "(node 'input' or output 'sink_name') MUST equal this value. Despite the "
                    "field name, this is NOT a node id — connections match by string, not by "
                    "topology."
                ),
                "examples": ["raw_url_rows", "csv_rows", "fetched_text"],
            },
            "on_validation_failure": {
                "type": "string",
                "description": _SOURCE_VALIDATION_FAILURE_DESCRIPTION,
                "default": _DEFAULT_SOURCE_VALIDATION_FAILURE,
            },
            "options": {
                "type": "object",
                "description": (
                    "Plugin-specific config (merged with blob path). Required fields vary by plugin: "
                    "text sources need 'column' (output field name) and 'schema' (e.g., {mode: 'observed'})."
                ),
            },
        },
        "required": ["blob_id", "on_success"],
    },
    blob_store_only=False,
    augments_on_failure=True,
)


def _first_nonempty_csv_row(content: str) -> tuple[str, ...] | None:
    """Return the first non-empty CSV row, if any."""
    for row in csv.reader(io.StringIO(content)):
        if any(cell.strip() for cell in row):
            return tuple(row)
    return None


def _is_header_only_csv(content: str) -> tuple[str, ...] | None:
    """Return the sole CSV row when content is header-only, otherwise None."""
    nonempty_rows = [tuple(row) for row in csv.reader(io.StringIO(content)) if any(cell.strip() for cell in row)]
    if len(nonempty_rows) != 1:
        return None
    return nonempty_rows[0]


def _header_only_inline_csv_conflict(
    prepared: _PreparedBlobCreate,
    *,
    session_engine: Engine,
    session_id: str,
) -> str | None:
    """Reject schema-only CSV blobs when a matching uploaded CSV is ready."""
    if prepared.mime_type != "text/csv":
        return None
    header = _is_header_only_csv(prepared.content_bytes.decode("utf-8"))
    if header is None:
        return None

    with session_engine.connect() as conn:
        rows = conn.execute(
            select(blobs_table).where(
                blobs_table.c.session_id == session_id,
                blobs_table.c.mime_type == "text/csv",
                blobs_table.c.status == "ready",
                blobs_table.c.created_by == "user",
            )
        ).fetchall()

    matches: list[BlobToolRecord] = []
    for row in rows:
        blob = _blob_row_to_tool_dict(row)
        try:
            candidate_header = _first_nonempty_csv_row(Path(blob["storage_path"]).read_text(encoding="utf-8"))
        except OSError as exc:
            raise AuditIntegrityError(
                f"Ready uploaded blob '{blob['id']}' storage_path could not be read during set_pipeline inline CSV custody check"
            ) from exc
        if candidate_header == header and blob["size_bytes"] > len(prepared.content_bytes):
            matches.append(blob)

    if not matches:
        return None

    choices = ", ".join(f"{blob['filename']} ({blob['id']}, {blob['size_bytes']} bytes)" for blob in matches)
    return (
        "Refusing header-only inline CSV for set_pipeline because ready uploaded CSV blob(s) "
        f"with matching headers already exist in this session: {choices}. "
        "Bind the uploaded file with source.blob_id or call list_blobs then set_source_from_blob."
    )


def _execute_inspect_source(
    arguments: dict[str, Any],
    state: CompositionState,
    context: ToolContext,
) -> ToolResult:
    """Inspect a blob-backed source and return bounded structural facts.

    Mirrors the lifecycle and integrity guards of ``_execute_get_blob_content``
    (only ``ready`` blobs are readable; SHA-256 verified; UnicodeDecodeError
    surfaced as tool-failure) but returns ``SourceInspectionFacts`` rather
    than raw content. Reads at most 8 KiB and parses at most 100 rows.

    Never returns raw row content — only summary facts (headers, inferred
    types, URL candidates, warnings, redacted identity).
    """
    if context.session_engine is None or context.session_id is None:
        return _failure_result(state, "Blob tools require session context.")

    blob_id = arguments["blob_id"]
    blob = _sync_get_blob(context.session_engine, blob_id, context.session_id)
    if blob is None:
        return _failure_result(state, f"Blob '{blob_id}' not found.")

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

    _verify_blob_content_integrity(blob, data)

    blob_id_warning: str | None = None
    try:
        blob_uuid = UUID(blob_id)
    except ValueError:
        blob_uuid = None
        truncated = blob_id if len(blob_id) <= 64 else blob_id[:64] + "..."
        blob_id_warning = (
            f"blob_id_not_uuid: matched blob_id {truncated!r} is not a parseable "
            "UUID — redacted_identity will omit blob_id and surface "
            "content_hash_prefix only"
        )

    facts = inspect_blob_content(
        content=data,
        filename=blob["filename"],
        mime_type=blob["mime_type"],
        blob_id=blob_uuid,
        content_hash=blob["content_hash"],
    )
    if blob_id_warning is not None:
        facts = replace(facts, warnings=(blob_id_warning, *facts.warnings))
    return _discovery_result(state, facts_to_dict(facts))


_INSPECT_SOURCE_DECLARATION = ToolDeclaration(
    name="inspect_source",
    handler=_execute_inspect_source,
    kind=ToolKind.BLOB_DISCOVERY,
    description=(
        "Return bounded structural facts about a blob-backed source: source kind, observed "
        "headers, sample row count, inferred scalar types per column, URL candidates, and "
        "warnings. Reads at most 8 KiB of the blob and parses at most 100 rows. Use this "
        "before declaring a fixed CSV/JSON schema — observed headers and inferred types "
        "tell you which fields the source actually contains and what numeric coercion is "
        "needed before any gate or value_transform numeric op. Never returns raw row "
        "content; only summary facts."
    ),
    json_schema={
        "type": "object",
        "properties": {
            "blob_id": {
                "type": "string",
                "description": "ID of the blob to inspect.",
            },
        },
        "required": ["blob_id"],
    },
)


def _execute_patch_source_options(
    args: dict[str, Any],
    state: CompositionState,
    context: ToolContext,
) -> ToolResult:
    """Apply a merge-patch to the current source options.

    Tier-3 boundary: ``args`` is an LLM-supplied dict.  Validated via the
    Pydantic redaction-bearing model :class:`PatchSourceOptionsArgumentsModel`
    (the single source of truth for the argument schema — supersedes the
    deleted ``_TOOL_REQUIRED_PATHS["patch_source_options"]`` entry in
    ``service.py``, rev-3 N7 / rev-4 M1).

    On :class:`pydantic.ValidationError` the handler re-raises as
    :class:`ToolArgumentError` so the compose loop's ARG_ERROR routing at
    ``service.py:2480`` receives the right exception class.  A bare
    ``ValidationError`` would escape into the catch-all
    (``ComposerPluginCrashError`` → HTTP 500) — wrong disposition for
    Tier-3 input.
    """
    try:
        validated = PatchSourceOptionsArgumentsModel.model_validate(args)
    except PydanticValidationError as exc:
        raise ToolArgumentError(
            argument="patch_source_options arguments",
            expected="object conforming to PatchSourceOptionsArgumentsModel",
            actual_type=type(exc).__name__,
        ) from exc
    source_name = validated.source_name
    if source_name not in state.sources:
        return _failure_result(state, f"No source named '{source_name}' configured to patch.")
    current_source = state.sources[source_name]
    patch = validated.patch

    manual_authoring_error = _reject_manual_source_authoring(patch, tool_name="patch_source_options")
    if manual_authoring_error is not None:
        return _failure_result(state, manual_authoring_error)

    # Lock the (path, blob_ref) pair on blob-backed sources.  Once
    # set_source_from_blob has bound a source to a blob, the path is the
    # blob's canonical storage_path and is not patchable: any divergence
    # breaks runtime path resolution and composer/runtime agreement.
    # Replace the binding via a fresh set_source_from_blob (or
    # clear_source) instead of patching it.  See elspeth-07089fbaa3.
    if "blob_ref" in current_source.options:
        forbidden_keys = {"path", "blob_ref"} & patch.keys()
        if forbidden_keys:
            return _failure_result(
                state,
                "Cannot patch "
                f"{sorted(forbidden_keys)} on a blob-backed source. "
                "The 'path' is bound to the referenced blob's canonical "
                "storage_path. Re-bind via set_source_from_blob (or call "
                "clear_source first) to change the underlying blob.",
            )

    new_options = _apply_merge_patch(current_source.options, patch)
    credential_error = _credential_wiring_contract_failure(
        state,
        component_id=_source_component_id(source_name),
        component_type="source",
        options=new_options,
    )
    if credential_error is not None:
        return credential_error

    # S2: Validate patched source paths against allowlist
    path_error = _validate_source_path(new_options, context.data_dir)
    if path_error is not None:
        return _failure_result(state, path_error)

    # Pre-validate patched options against config model
    prevalidation_error = _prevalidate_source(
        current_source.plugin,
        new_options,
        current_source.on_validation_failure,
    )
    if prevalidation_error is not None:
        return _failure_result(state, prevalidation_error)

    new_source = replace(current_source, options=new_options)
    new_state = state.with_named_source(source_name, new_source)
    return _mutation_result(new_state, (_source_component_id(source_name),))


def _handle_patch_source_options(
    arguments: dict[str, Any],
    state: CompositionState,
    context: ToolContext,
) -> ToolResult:
    result = _execute_patch_source_options(arguments, state, context)
    if not result.success:
        return result
    source_name = PatchSourceOptionsArgumentsModel.model_validate(arguments).source_name
    if source_name not in result.updated_state.sources:
        return result
    source = result.updated_state.sources[source_name]
    return _attach_post_call_hints(
        result,
        context.catalog,
        plugin_type="source",
        tool_name="patch_source_options",
        plugin_name=source.plugin,
        config_snapshot=source.options,
    )


_PATCH_SOURCE_OPTIONS_DECLARATION = ToolDeclaration(
    name="patch_source_options",
    handler=_handle_patch_source_options,
    kind=ToolKind.MUTATION,
    description="Apply a shallow merge-patch to a named source's options. "
    "Keys in the patch overwrite existing keys. "
    "Keys set to null are deleted. Missing keys are unchanged.",
    json_schema={
        "type": "object",
        "properties": {
            "source_name": {
                "type": "string",
                "description": "Source root name to patch. Defaults to 'source'.",
            },
            "patch": {
                "type": "object",
                "description": "Merge-patch to apply to source options.",
            },
        },
        "required": ["patch"],
    },
    augments_on_failure=True,
)


def _execute_clear_source(
    args: dict[str, Any],
    state: CompositionState,
    context: ToolContext,
) -> ToolResult:
    """Remove the pipeline source."""
    del context  # unused; signature uniformity with the other handlers.
    if state.source is None:
        return _failure_result(state, "No source configured to clear.")
    new_state = state.without_source()
    return _mutation_result(new_state, ("source",))


def _handle_clear_source(
    arguments: dict[str, Any],
    state: CompositionState,
    context: ToolContext,
) -> ToolResult:
    return _execute_clear_source(arguments, state, context)


_CLEAR_SOURCE_DECLARATION = ToolDeclaration(
    name="clear_source",
    handler=_handle_clear_source,
    kind=ToolKind.MUTATION,
    description="Remove the source from the pipeline composition state.",
    json_schema={"type": "object", "properties": {}, "required": []},
)


TOOLS_IN_MODULE: tuple[ToolDeclaration, ...] = (
    _LIST_SOURCES_DECLARATION,
    _SET_SOURCE_DECLARATION,
    _PATCH_SOURCE_OPTIONS_DECLARATION,
    _CLEAR_SOURCE_DECLARATION,
    _SET_SOURCE_FROM_BLOB_DECLARATION,
    _INSPECT_SOURCE_DECLARATION,
)
"""Every tool declared in this module, in stable order.

``_dispatch.py`` aggregates this tuple alongside every other plane's
TOOLS_IN_MODULE to build the registered-tool universe."""
