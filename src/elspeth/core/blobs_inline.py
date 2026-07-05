"""Inline blob content resolver helpers.

Layer: L1 (core). Imports L0 contracts only.

This module mirrors the secret resolver's tree-walk shape while keeping
blob content resolution split into sync discovery, async fetch, and sync
substitution phases. This first slice implements pure discovery only.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
from collections.abc import Callable, Mapping
from typing import Any, Final, cast
from uuid import UUID

from elspeth.contracts.blobs import (
    AllowedMimeType,
    BlobContentMissingError,
    BlobIntegrityError,
    BlobNotFoundError,
    BlobRecord,
    BlobServiceProtocol,
    BlobStateError,
)
from elspeth.contracts.blobs_inline import (
    BlobContentResolutionError,
    BlobInlineRef,
    BlobInlineValidationViolation,
    ResolvedBlobContent,
    WidenedBlobRefShape,
    is_widened_blob_ref,
)

_NODE_COLLECTION_KEYS: Final = ("transforms", "gates", "aggregations", "coalesce")
_OUTPUT_COLLECTION_KEYS: Final = ("outputs", "sinks")
_VALIDATION_INLINE_CONTENT_PLACEHOLDER: Final = "validated blob-backed inline content placeholder"
BLOB_INLINE_PER_REF_BYTE_CAP: Final = 256 * 1024
BLOB_INLINE_AGGREGATE_BYTE_CAP: Final = 1024 * 1024


def _discover_blob_content_refs(config: dict[str, Any]) -> list[BlobInlineRef]:
    """Return inline-content refs from a YAML-shaped config dict.

    The returned field paths use the canonical audit form:
    ``source.options.<field>`` for legacy singular-source dicts,
    ``source:<name>.options.<field>`` for plural sources,
    ``node:<name>.options.<field>``, or ``output:<sink>.options.<field>``.
    """
    refs: list[BlobInlineRef] = []
    malformed: list[tuple[str, str]] = []

    _walk_source(config, refs, malformed)
    _walk_sources(config, refs, malformed)
    _walk_nodes(config, refs, malformed)
    _walk_outputs(config, refs, malformed)

    if malformed:
        raise BlobContentResolutionError(malformed=malformed)
    return refs


async def _validate_blob_content_refs(
    blob_service: BlobServiceProtocol,
    config: dict[str, Any],
    *,
    session_id: UUID,
    per_ref_byte_cap: int | None = None,
    aggregate_byte_cap: int | None = None,
) -> list[BlobInlineValidationViolation]:
    """Return validate-path violations without raising recoverable errors."""
    try:
        refs = _discover_blob_content_refs(config)
    except BlobContentResolutionError as exc:
        return _malformed_validation_violations(exc)

    violations: list[BlobInlineValidationViolation] = []
    aggregate_bytes = 0
    for ref in refs:
        lookup_result = await _get_blob_record_for_validation(blob_service, ref, session_id=session_id)
        if type(lookup_result) is BlobInlineValidationViolation:
            violations.append(lookup_result)
            continue
        record = cast(BlobRecord, lookup_result)
        record_violations, counts_toward_aggregate = _record_validation_violations(
            ref,
            record,
            per_ref_byte_cap=per_ref_byte_cap,
        )
        violations.extend(record_violations)
        if counts_toward_aggregate:
            aggregate_bytes += record.size_bytes

    _append_aggregate_size_violation(violations, aggregate_bytes, aggregate_byte_cap)
    return violations


def _validate_blob_content_refs_sync(
    blob_get_metadata: Callable[[UUID], BlobRecord | None],
    config: dict[str, Any],
    *,
    per_ref_byte_cap: int | None = None,
    aggregate_byte_cap: int | None = None,
) -> list[BlobInlineValidationViolation]:
    """Sync validate-path sibling for callers that cannot await metadata lookup."""
    try:
        refs = _discover_blob_content_refs(config)
    except BlobContentResolutionError as exc:
        return _malformed_validation_violations(exc)

    violations: list[BlobInlineValidationViolation] = []
    aggregate_bytes = 0
    for ref in refs:
        record = blob_get_metadata(ref.blob_id)
        if record is None:
            violations.append(_missing_validation_violation(ref))
            continue
        record_violations, counts_toward_aggregate = _record_validation_violations(
            ref,
            record,
            per_ref_byte_cap=per_ref_byte_cap,
        )
        violations.extend(record_violations)
        if counts_toward_aggregate:
            aggregate_bytes += record.size_bytes

    _append_aggregate_size_violation(violations, aggregate_bytes, aggregate_byte_cap)
    return violations


def _substitute_blob_content_refs_for_validation(config: dict[str, Any]) -> dict[str, Any]:
    """Replace validated inline-content markers with parseable placeholder text.

    Validate-time metadata checks intentionally do not read blob bytes; runtime
    remains the only path that links blobs to a run, fetches bytes, verifies the
    pinned hash against content, and records the audit rows. Once metadata has
    proven a marker points at a ready blob within caps, plugin construction
    still needs a field-shaped value rather than the deferred marker dict.
    """
    refs = _discover_blob_content_refs(config)
    for ref in refs:
        _substitute_at_path(config, ref.field_path, _VALIDATION_INLINE_CONTENT_PLACEHOLDER)
    return config


async def _get_blob_record_for_validation(
    blob_service: BlobServiceProtocol,
    ref: BlobInlineRef,
    *,
    session_id: UUID,
) -> BlobRecord | BlobInlineValidationViolation:
    try:
        record = await blob_service.get_blob(ref.blob_id)
    except BlobNotFoundError:
        return _missing_validation_violation(ref)
    except BlobStateError as exc:
        return BlobInlineValidationViolation(
            category="not_ready",
            field_path=ref.field_path,
            detail=str(exc),
        )
    if record.session_id != session_id:
        return _missing_validation_violation(ref)
    return record


def _malformed_validation_violations(exc: BlobContentResolutionError) -> list[BlobInlineValidationViolation]:
    return [
        BlobInlineValidationViolation(category="malformed", field_path=field_path, detail=reason) for field_path, reason in exc.malformed
    ]


def _missing_validation_violation(ref: BlobInlineRef) -> BlobInlineValidationViolation:
    return BlobInlineValidationViolation(
        category="missing",
        field_path=ref.field_path,
        detail=f"blob {ref.blob_id} not found",
    )


def _record_validation_violations(
    ref: BlobInlineRef,
    record: BlobRecord,
    *,
    per_ref_byte_cap: int | None,
) -> tuple[list[BlobInlineValidationViolation], bool]:
    if record.status != "ready":
        return [
            BlobInlineValidationViolation(
                category="not_ready",
                field_path=ref.field_path,
                detail=f"blob {ref.blob_id} status is {record.status!r}",
            )
        ], False
    if record.content_hash != ref.sha256:
        return [
            BlobInlineValidationViolation(
                category="hash_mismatch",
                field_path=ref.field_path,
                detail=f"composer-pinned hash {ref.sha256[:16]}... != blob hash {_short_hash(record.content_hash)}",
            )
        ], False

    violations: list[BlobInlineValidationViolation] = []
    if per_ref_byte_cap is not None and record.size_bytes > per_ref_byte_cap:
        violations.append(
            BlobInlineValidationViolation(
                category="oversized",
                field_path=ref.field_path,
                detail=f"{record.size_bytes} bytes exceeds per-ref cap {per_ref_byte_cap}",
            )
        )
    return violations, True


def _enforce_blob_content_ref_metadata(
    refs: list[BlobInlineRef],
    records_by_blob_id: Mapping[UUID, BlobRecord],
    *,
    per_ref_byte_cap: int | None = None,
    aggregate_byte_cap: int | None = None,
) -> None:
    """Fail closed on metadata-only inline-content violations before byte reads."""
    missing: list[str] = []
    oversized: list[tuple[str, int, int]] = []
    not_ready: list[tuple[str, str]] = []

    aggregate_bytes = 0
    for ref in refs:
        if ref.blob_id not in records_by_blob_id:
            missing.append(ref.field_path)
            continue
        record = records_by_blob_id[ref.blob_id]
        if record.status != "ready":
            not_ready.append((ref.field_path, f"blob {ref.blob_id} status is {record.status!r}"))
            continue
        if record.content_hash != ref.sha256:
            raise BlobIntegrityError(
                str(ref.blob_id),
                expected=ref.sha256,
                actual=record.content_hash or "<missing>",
            )
        if per_ref_byte_cap is not None and record.size_bytes > per_ref_byte_cap:
            oversized.append((ref.field_path, record.size_bytes, per_ref_byte_cap))
        aggregate_bytes += record.size_bytes

    if aggregate_byte_cap is not None and aggregate_bytes > aggregate_byte_cap:
        oversized.append(("(aggregate)", aggregate_bytes, aggregate_byte_cap))

    if missing or oversized or not_ready:
        raise BlobContentResolutionError(
            missing=missing,
            oversized=oversized,
            not_ready=not_ready,
        )


def _short_hash(value: str | None) -> str:
    if value is None:
        return "<missing>"
    return f"{value[:16]}..."


def _append_aggregate_size_violation(
    violations: list[BlobInlineValidationViolation],
    aggregate_bytes: int,
    aggregate_byte_cap: int | None,
) -> None:
    if aggregate_byte_cap is None or aggregate_bytes <= aggregate_byte_cap:
        return
    violations.append(
        BlobInlineValidationViolation(
            category="oversized",
            field_path="(aggregate)",
            detail=f"total resolved bytes {aggregate_bytes} exceeds aggregate cap {aggregate_byte_cap}",
        )
    )


async def _fetch_blob_contents(
    blob_service: BlobServiceProtocol,
    refs: list[BlobInlineRef],
) -> dict[BlobInlineRef, bytes]:
    """Fetch content bytes for discovered refs, deduped by blob id."""
    unique_blob_ids = _unique_blob_ids(refs)
    results = await asyncio.gather(
        *(blob_service.read_blob_content(blob_id) for blob_id in unique_blob_ids),
        return_exceptions=True,
    )
    refs_by_blob = _refs_by_blob_id(refs)
    bytes_by_blob: dict[UUID, bytes] = {}
    missing: list[str] = []
    not_ready: list[tuple[str, str]] = []

    for blob_id, result in zip(unique_blob_ids, results, strict=True):
        result_type = type(result)
        if result_type is BlobIntegrityError or result_type is BlobContentMissingError:
            raise cast(BlobIntegrityError | BlobContentMissingError, result)
        if result_type is BlobNotFoundError:
            for ref in refs_by_blob[blob_id]:
                missing.append(ref.field_path)
            continue
        if result_type is BlobStateError:
            for ref in refs_by_blob[blob_id]:
                not_ready.append((ref.field_path, str(result)))
            continue
        if type(result) is not bytes:
            raise cast(BaseException, result)
        bytes_by_blob[blob_id] = result

    if missing or not_ready:
        raise BlobContentResolutionError(missing=missing, not_ready=not_ready)

    return {ref: bytes_by_blob[ref.blob_id] for ref in refs}


def _unique_blob_ids(refs: list[BlobInlineRef]) -> list[UUID]:
    unique: list[UUID] = []
    seen: set[UUID] = set()
    for ref in refs:
        if ref.blob_id in seen:
            continue
        seen.add(ref.blob_id)
        unique.append(ref.blob_id)
    return unique


def _refs_by_blob_id(refs: list[BlobInlineRef]) -> dict[UUID, list[BlobInlineRef]]:
    refs_by_blob: dict[UUID, list[BlobInlineRef]] = {}
    for ref in refs:
        if ref.blob_id not in refs_by_blob:
            refs_by_blob[ref.blob_id] = []
        refs_by_blob[ref.blob_id].append(ref)
    return refs_by_blob


def _substitute_blob_content_refs(
    config: dict[str, Any],
    fetched: dict[BlobInlineRef, bytes],
    *,
    refs: list[BlobInlineRef],
    blob_metadata: dict[UUID, tuple[AllowedMimeType, int]],
) -> tuple[dict[str, Any], list[ResolvedBlobContent]]:
    """Replace inline-content markers with decoded strings and audit rows."""
    decoded_by_ref: dict[BlobInlineRef, str] = {}
    audit: list[ResolvedBlobContent] = []
    undecodable: list[tuple[str, str]] = []

    for ref in refs:
        content = fetched[ref]
        actual_hash = hashlib.sha256(content).hexdigest()
        if not hmac.compare_digest(actual_hash, ref.sha256):
            raise BlobIntegrityError(str(ref.blob_id), expected=ref.sha256, actual=actual_hash)
        decoded, decode_error = _decode_blob_content(ref, content)
        if decode_error is not None:
            undecodable.append(decode_error)
            continue
        if decoded is None:
            raise RuntimeError("Blob inline decode helper returned neither decoded content nor an error")

        decoded_by_ref[ref] = decoded
        mime_type, byte_length = blob_metadata[ref.blob_id]
        audit.append(
            ResolvedBlobContent(
                field_path=ref.field_path,
                blob_id=ref.blob_id,
                content_hash=actual_hash,
                byte_length=byte_length,
                mime_type=mime_type,
                encoding=ref.encoding,
            )
        )

    if undecodable:
        raise BlobContentResolutionError(undecodable=undecodable)

    for ref in refs:
        _substitute_at_path(config, ref.field_path, decoded_by_ref[ref])

    return config, audit


def _decode_blob_content(ref: BlobInlineRef, content: bytes) -> tuple[str, None] | tuple[None, tuple[str, str]]:
    try:
        return content.decode(ref.encoding), None
    except UnicodeDecodeError:
        return None, (ref.field_path, ref.encoding)


def _substitute_at_path(config: dict[str, Any], field_path: str, value: str) -> None:
    if ".options." not in field_path:
        raise ValueError(f"Unrecognised blob inline field_path: {field_path!r}")
    prefix, rest = field_path.split(".options.", 1)
    keys = rest.split(".")
    if prefix == "source":
        container = _source_options(config, None)
    elif prefix.startswith("source:"):
        container = _source_options(config, prefix[len("source:") :])
    elif prefix.startswith("node:"):
        container = _node_options(config, prefix[len("node:") :])
    elif prefix.startswith("output:"):
        container = _output_options(config, prefix[len("output:") :])
    else:
        raise ValueError(f"Unrecognised blob inline field_path prefix: {prefix!r}")
    _assign_nested_option(container, keys, value)


def _source_options(config: dict[str, Any], source_name: str | None) -> dict[str, Any]:
    if source_name is None:
        source = config["source"]
        error_prefix = "source"
    else:
        sources = config["sources"]
        if type(sources) is not dict:
            raise TypeError("sources must be a mapping")
        source = cast(dict[object, object], sources)[source_name]
        error_prefix = f"source:{source_name}"
    if type(source) is not dict:
        raise TypeError(f"{error_prefix} must be a mapping")
    source_dict = cast(dict[str, object], source)
    options = source_dict["options"]
    if type(options) is not dict:
        raise TypeError(f"{error_prefix}.options must be a mapping")
    return cast(dict[str, Any], options)


def _node_options(config: dict[str, Any], node_name: str) -> dict[str, Any]:
    for collection_key in _NODE_COLLECTION_KEYS:
        if collection_key not in config:
            continue
        nodes = config[collection_key]
        if type(nodes) is not list:
            continue
        for node in cast(list[object], nodes):
            if type(node) is not dict:
                continue
            node_dict = cast(dict[str, object], node)
            if "name" not in node_dict or node_dict["name"] != node_name:
                continue
            if "options" not in node_dict:
                raise KeyError(f"Node {node_name!r} has no options mapping")
            options = node_dict["options"]
            if type(options) is not dict:
                raise TypeError(f"Node {node_name!r}.options must be a mapping")
            return cast(dict[str, Any], options)
    raise KeyError(f"Node {node_name!r} not found")


def _output_options(config: dict[str, Any], output_name: str) -> dict[str, Any]:
    for collection_key in _OUTPUT_COLLECTION_KEYS:
        if collection_key not in config:
            continue
        outputs = config[collection_key]
        if type(outputs) is not dict:
            continue
        outputs_dict = cast(dict[str, object], outputs)
        if output_name not in outputs_dict:
            continue
        output = outputs_dict[output_name]
        if type(output) is not dict:
            raise TypeError(f"Output {output_name!r} must be a mapping")
        output_dict = cast(dict[str, object], output)
        if "options" not in output_dict:
            raise KeyError(f"Output {output_name!r} has no options mapping")
        options = output_dict["options"]
        if type(options) is not dict:
            raise TypeError(f"Output {output_name!r}.options must be a mapping")
        return cast(dict[str, Any], options)
    raise KeyError(f"Output {output_name!r} not found")


def _assign_nested_option(container: dict[str, Any], keys: list[str], value: str) -> None:
    current = container
    for key in keys[:-1]:
        child = current[key]
        if type(child) is not dict:
            raise TypeError(f"Inline blob path segment {key!r} must point to a mapping")
        current = cast(dict[str, Any], child)
    current[keys[-1]] = value


def _walk_source(
    config: dict[str, Any],
    refs: list[BlobInlineRef],
    malformed: list[tuple[str, str]],
) -> None:
    if "source" not in config:
        return
    source = config["source"]
    if type(source) is not dict:
        return
    source_dict = cast(dict[str, object], source)
    if "options" not in source_dict:
        return
    options = source_dict["options"]
    if type(options) is not dict:
        return
    _walk_options(cast(dict[str, object], options), "source.options", refs, malformed)


def _walk_sources(
    config: dict[str, Any],
    refs: list[BlobInlineRef],
    malformed: list[tuple[str, str]],
) -> None:
    if "sources" not in config:
        return
    sources = config["sources"]
    if type(sources) is not dict:
        return
    for source_name, source in cast(dict[object, object], sources).items():
        if type(source_name) is not str or type(source) is not dict:
            continue
        source_dict = cast(dict[str, object], source)
        if "options" not in source_dict:
            continue
        options = source_dict["options"]
        if type(options) is not dict:
            continue
        _walk_options(cast(dict[str, object], options), f"source:{source_name}.options", refs, malformed)


def _walk_nodes(
    config: dict[str, Any],
    refs: list[BlobInlineRef],
    malformed: list[tuple[str, str]],
) -> None:
    for collection_key in _NODE_COLLECTION_KEYS:
        if collection_key not in config:
            continue
        nodes = config[collection_key]
        if type(nodes) is not list:
            continue
        for node in nodes:
            if type(node) is not dict:
                continue
            node_dict = cast(dict[str, object], node)
            if "name" not in node_dict or "options" not in node_dict:
                continue
            name = node_dict["name"]
            options = node_dict["options"]
            if type(name) is not str or type(options) is not dict:
                continue
            _walk_options(cast(dict[str, object], options), f"node:{name}.options", refs, malformed)


def _walk_outputs(
    config: dict[str, Any],
    refs: list[BlobInlineRef],
    malformed: list[tuple[str, str]],
) -> None:
    for collection_key in _OUTPUT_COLLECTION_KEYS:
        if collection_key not in config:
            continue
        outputs = config[collection_key]
        if type(outputs) is not dict:
            continue
        for output_name, output in cast(dict[object, object], outputs).items():
            if type(output_name) is not str or type(output) is not dict:
                continue
            output_dict = cast(dict[str, object], output)
            if "options" not in output_dict:
                continue
            options = output_dict["options"]
            if type(options) is not dict:
                continue
            _walk_options(cast(dict[str, object], options), f"output:{output_name}.options", refs, malformed)


def _walk_options(
    value: object,
    field_path: str,
    refs: list[BlobInlineRef],
    malformed: list[tuple[str, str]],
) -> None:
    if type(value) is dict:
        mapping = cast(dict[object, object], value)
        if _is_source_options_path(field_path) and _is_source_options_bind_source(mapping):
            for key, child in mapping.items():
                if type(key) is str:
                    _walk_options(child, f"{field_path}.{key}", refs, malformed)
            return
        if "blob_ref" in mapping:
            _collect_or_reject_marker(mapping, field_path, refs, malformed)
            return
        for key, child in mapping.items():
            if type(key) is str:
                _walk_options(child, f"{field_path}.{key}", refs, malformed)
        return

    if type(value) is list:
        for child in cast(list[object], value):
            if type(child) is dict and "blob_ref" in cast(dict[object, object], child):
                malformed.append((field_path, "inline blob refs inside lists are not supported"))
                return


def _is_source_options_path(field_path: str) -> bool:
    return field_path == "source.options" or (field_path.startswith("source:") and field_path.endswith(".options"))


def _collect_or_reject_marker(
    marker: dict[object, object],
    field_path: str,
    refs: list[BlobInlineRef],
    malformed: list[tuple[str, str]],
) -> None:
    shape = is_widened_blob_ref(marker)
    if shape is None:
        malformed.append((field_path, _malformed_reason(marker)))
        return
    if shape.mode != "inline_content":
        return
    refs.append(_ref_from_shape(field_path, shape))


def _is_source_options_bind_source(mapping: dict[object, object]) -> bool:
    if "blob_ref" not in mapping or "mode" not in mapping:
        return False
    mode = mapping["mode"]
    if mode != "bind_source":
        return False
    if "sha256" in mapping or "encoding" in mapping:
        return False
    return "path" not in mapping or type(mapping["path"]) is str


def _ref_from_shape(field_path: str, shape: WidenedBlobRefShape) -> BlobInlineRef:
    if shape.sha256 is None:
        raise RuntimeError("inline_content marker passed recognition without sha256")
    return BlobInlineRef(
        field_path=field_path,
        blob_id=shape.blob_id,
        sha256=shape.sha256,
        encoding=shape.encoding,
    )


def _malformed_reason(marker: dict[object, object]) -> str:
    if "mode" not in marker:
        return "missing mode"
    mode = marker["mode"]
    if type(mode) is not str:
        return "mode must be string"
    if mode not in {"bind_source", "inline_content"}:
        return f"unknown mode {mode!r}"
    if "encoding" in marker and type(marker["encoding"]) is not str:
        return "encoding must be string"
    if mode == "inline_content" and "sha256" not in marker:
        return "inline_content requires sha256"
    if mode == "bind_source" and ("sha256" in marker or "encoding" in marker):
        return "bind_source cannot carry sha256 or encoding"
    return "invalid blob_ref marker"
