"""Inline blob content resolver helpers.

Layer: L1 (core). Imports L0 contracts only.

This module mirrors the secret resolver's tree-walk shape while keeping
blob content resolution split into sync discovery, async fetch, and sync
substitution phases. This first slice implements pure discovery only.
"""

from __future__ import annotations

from typing import Any, Final, cast

from elspeth.contracts.blobs_inline import (
    BlobContentResolutionError,
    BlobInlineRef,
    WidenedBlobRefShape,
    is_widened_blob_ref,
)

_NODE_COLLECTION_KEYS: Final = ("transforms", "gates", "aggregations", "coalesce")
_OUTPUT_COLLECTION_KEYS: Final = ("outputs", "sinks")


def _discover_blob_content_refs(config: dict[str, Any]) -> list[BlobInlineRef]:
    """Return inline-content refs from a YAML-shaped config dict.

    The returned field paths use the canonical audit form:
    ``source.options.<field>``, ``node:<name>.options.<field>``, or
    ``output:<sink>.options.<field>``.
    """
    refs: list[BlobInlineRef] = []
    malformed: list[tuple[str, str]] = []

    _walk_source(config, refs, malformed)
    _walk_nodes(config, refs, malformed)
    _walk_outputs(config, refs, malformed)

    if malformed:
        raise BlobContentResolutionError(malformed=malformed)
    return refs


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
        if field_path == "source.options" and _is_source_options_bind_source(mapping):
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
    if mode not in {"bind_source", "inline_content"}:
        return f"unknown mode {mode!r}"
    if mode == "inline_content" and "sha256" not in marker:
        return "inline_content requires sha256"
    if mode == "bind_source" and ("sha256" in marker or "encoding" in marker):
        return "bind_source cannot carry sha256 or encoding"
    return "invalid blob_ref marker"
