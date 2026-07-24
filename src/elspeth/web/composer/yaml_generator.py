"""Deterministic YAML generator -- CompositionState to ELSPETH pipeline YAML.

Pure function. Same CompositionState always produces byte-identical YAML.
Uses yaml.dump() with sort_keys=True for determinism.

Layer: L3 (application).

Trust model: state_dict comes from CompositionState.to_dict() — our own
serialization of our own frozen dataclasses. Fields are either always
present (direct access) or conditionally present based on node_type
(check with ``in``). Never use .get() — a missing field is a bug in
to_dict(), not an expected absence.

Web-specific metadata keys (e.g., blob_ref for file provenance tracking)
are filtered from options before YAML generation. These are UI-layer
concerns that should not leak into engine configuration. Plugin configs
use Pydantic with extra="forbid" — unknown keys cause validation failure.

Public export/share/MCP YAML has one extra scrub: blob-bound source storage
paths are omitted. HTTP export returns ``source_blob_ids`` so imports can
re-bind an uploaded blob in the destination session; other public surfaces
must not expose a server storage path as a path-only replay target. Runtime
execution keeps the path because the engine still needs the local file path
after blob ownership checks pass.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import yaml

from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.trust_boundary import trust_boundary
from elspeth.web.composer.guided_blob_refs import (
    GUIDED_REVIEWED_BLOB_PATH_KEYS,
    validate_guided_reviewed_blob_binding,
    validate_guided_reviewed_blob_ref,
    validate_guided_reviewed_blob_source_mapping,
)
from elspeth.web.composer.state import COMPOSER_NODE_TYPES, CompositionState, queue_node_contract_error
from elspeth.web.interpretation_state import AUTHORING_METADATA_OPTION_KEYS
from elspeth.web.paths import SOURCE_LOCAL_PATH_OPTION_KEYS

# Web-specific metadata keys that should NOT appear in engine YAML.
# These are UI-layer concerns for provenance tracking, not plugin config.
# Plugin configs use Pydantic with extra="forbid" — unknown keys cause errors.
_WEB_ONLY_OPTION_KEYS = frozenset({"blob_ref"}) | AUTHORING_METADATA_OPTION_KEYS


@trust_boundary(
    tier=3,
    source="web-authored source options mapping (untrusted blob_ref value)",
    source_param="options",
    suppresses=("R1",),
    invariant="returns True only when blob_ref is present and non-null; absent keys yield False, never raise",
    non_raising=True,
)
def _has_blob_binding(options: dict[str, Any]) -> bool:
    return options.get("blob_ref") is not None


@trust_boundary(
    tier=3,
    source="web-authored source options mapping (untrusted mode value)",
    source_param="options",
    suppresses=("R1",),
    invariant="returns True only for the exact 'bind_source' mode string; absent or mistyped mode yields False, never raises",
    non_raising=True,
)
def _has_bind_source_mode(options: dict[str, Any]) -> bool:
    return options.get("mode") == "bind_source"


def _strip_web_metadata(options: dict[str, Any], *, omit_blob_bound_source_paths: bool = False) -> dict[str, Any]:
    """Remove web-specific metadata keys from options dict.

    Returns a shallow copy with web-only keys removed.
    """
    stripped = {k: v for k, v in options.items() if k not in _WEB_ONLY_OPTION_KEYS}
    if _has_blob_binding(options) and _has_bind_source_mode(options):
        # The guard above proves ``mode`` is present in ``options``; ``mode`` is
        # not in _WEB_ONLY_OPTION_KEYS, so it survives into ``stripped`` — pop
        # it directly without a default (a missing key here would be a bug).
        stripped.pop("mode")
    if omit_blob_bound_source_paths and _has_blob_binding(options):
        for key in SOURCE_LOCAL_PATH_OPTION_KEYS:
            stripped.pop(key, None)
    return stripped


def _source_entry(source: dict[str, Any], *, omit_blob_bound_source_paths: bool) -> dict[str, Any]:
    """Convert a serialized SourceSpec dict into runtime YAML shape."""
    source_options = _strip_web_metadata(
        dict(source["options"]),
        omit_blob_bound_source_paths=omit_blob_bound_source_paths,
    )
    source_options["on_validation_failure"] = source["on_validation_failure"]
    return {
        "plugin": source["plugin"],
        "on_success": source["on_success"],
        "options": source_options,
    }


def _generate_pipeline_dict(state: CompositionState, *, omit_blob_bound_source_paths: bool) -> dict[str, Any]:
    """Convert a CompositionState to ELSPETH's canonical pipeline dict.

    Maps CompositionState fields to the YAML structure expected by
    ELSPETH's load_settings() parser. This is the canonical analysis form
    for code that needs to walk a composition state using runtime/YAML
    section names without serializing to text first.

    Calls state.to_dict() to unwrap all frozen containers
    (MappingProxyType -> dict, tuple -> list) before building the dict.

    Args:
        state: The pipeline composition state to convert.

    Returns:
        Plain dict representing the pipeline configuration.
    """
    # Unwrap frozen containers to plain Python types (R4).
    # to_dict() recursively converts MappingProxyType -> dict,
    # tuple -> list. Without this, yaml.dump() raises RepresenterError.
    state_dict = state.to_dict()

    doc: dict[str, Any] = {}

    for node in state_dict["nodes"]:
        node_type = node["node_type"]
        if node_type not in COMPOSER_NODE_TYPES:
            raise ValueError(f"Unknown node_type '{node_type}' for node '{node['id']}'.")

    sources = state_dict["sources"]
    if sources:
        doc["sources"] = {
            name: _source_entry(source, omit_blob_bound_source_paths=omit_blob_bound_source_paths) for name, source in sources.items()
        }

    # Queues — structural pass-through fan-in points (elspeth-a5b86149d4).
    # Emitted after sources and before executable node lists so the YAML reads
    # source -> queues -> transforms -> ... Queue nodes are in COMPOSER_NODE_TYPES
    # but belong to none of the executable node lists below, so without this
    # block a queue node would be silently dropped from the export. Defend the
    # canonical shape here via the single source of truth rather than trusting
    # internal state blindly.
    queues = [node for node in state.nodes if node.node_type == "queue"]
    if queues:
        queues_doc: dict[str, Any] = {}
        for queue in queues:
            contract_error = queue_node_contract_error(queue)
            if contract_error is not None:
                raise ValueError(contract_error)
            queue_entry: dict[str, Any] = {}
            description = queue.options.get("description")
            if isinstance(description, str):
                queue_entry["description"] = description
            queues_doc[queue.id] = queue_entry
        doc["queues"] = queues_doc

    # Transforms — filter nodes by type, access always-present fields directly.
    transforms = [n for n in state_dict["nodes"] if n["node_type"] == "transform"]
    if transforms:
        doc["transforms"] = []
        for t in transforms:
            if t["on_error"] is None:
                raise ValueError(
                    f"Transform '{t['id']}' has on_error=None — "
                    f"upsert_node must default this at the mutation boundary, "
                    f"not leave it for the YAML generator to fabricate"
                )
            entry: dict[str, Any] = {
                "name": t["id"],
                "plugin": t["plugin"],
                "input": t["input"],
                "on_success": t["on_success"],
                "on_error": t["on_error"],
            }
            if t["options"]:
                entry["options"] = _strip_web_metadata(dict(t["options"]))
            doc["transforms"].append(entry)

    # Gates — condition and routes are conditionally present (only on gates).
    # to_dict() emits them when not None. Since we filtered to gates,
    # they must be present — access directly.
    gates = [n for n in state_dict["nodes"] if n["node_type"] == "gate"]
    if gates:
        doc["gates"] = []
        for g in gates:
            entry = {
                "name": g["id"],
                "input": g["input"],
                "condition": g["condition"],
                "routes": g["routes"],
            }
            # fork_to is conditionally present — only on fork gates
            if "fork_to" in g:
                entry["fork_to"] = g["fork_to"]
            doc["gates"].append(entry)

    # Aggregations
    aggregations = [n for n in state_dict["nodes"] if n["node_type"] == "aggregation"]
    if aggregations:
        doc["aggregations"] = []
        for a in aggregations:
            if a["on_error"] is None:
                raise ValueError(
                    f"Aggregation '{a['id']}' has on_error=None — "
                    f"upsert_node must default this at the mutation boundary, "
                    f"not leave it for the YAML generator to fabricate"
                )
            entry = {
                "name": a["id"],
                "plugin": a["plugin"],
                "input": a["input"],
                "on_success": a["on_success"],
                "on_error": a["on_error"],
            }
            # trigger, output_mode, expected_output_count are conditionally
            # emitted by to_dict() (only when non-None).  Use "in" checks to
            # match the to_dict() contract — a missing key is not an error
            # here; the engine treats absence as end-of-source-only flush.
            if "trigger" in a:
                entry["trigger"] = a["trigger"]
            if "output_mode" in a:
                entry["output_mode"] = a["output_mode"]
            if "expected_output_count" in a:
                entry["expected_output_count"] = a["expected_output_count"]
            if a["options"]:
                entry["options"] = _strip_web_metadata(dict(a["options"]))
            doc["aggregations"].append(entry)

    # Coalesce — branches, policy, merge are conditionally present.
    # Since we filtered to coalesces, they must be present.
    coalesces = [n for n in state_dict["nodes"] if n["node_type"] == "coalesce"]
    if coalesces:
        doc["coalesce"] = []
        for c in coalesces:
            entry = {
                "name": c["id"],
                "branches": c["branches"],
                "policy": c["policy"],
                "merge": c["merge"],
            }
            if c["on_success"] is not None:
                entry["on_success"] = c["on_success"]
            doc["coalesce"].append(entry)

    # Sinks — always-present fields, direct access.
    if state_dict["outputs"]:
        doc["sinks"] = {}
        for output in state_dict["outputs"]:
            sink_entry: dict[str, Any] = {
                "plugin": output["plugin"],
                "on_write_failure": output["on_write_failure"],
            }
            if output["options"]:
                sink_entry["options"] = _strip_web_metadata(dict(output["options"]))
            doc["sinks"][output["name"]] = sink_entry

    # landscape key is intentionally omitted -- URL comes from
    # WebSettings.get_landscape_url() at execution time (security fix S1).
    return doc


def generate_pipeline_dict(state: CompositionState) -> dict[str, Any]:
    """Convert a CompositionState to the runtime pipeline dict."""
    return _generate_pipeline_dict(state, omit_blob_bound_source_paths=False)


def reattach_guided_blob_refs_for_public_export(state: CompositionState) -> CompositionState:
    """Reconstitute guided blob refs before public YAML generation.

    Guided mode can commit sources with only their storage ``path`` while each
    authoritative ``blob_ref`` survives in schema-8 ``reviewed_sources``.
    Public YAML stripping keys off ``blob_ref``, so reattach each binding to a
    working copy. Private reviewed paths must exactly match the live source;
    public ``blob:<uuid>`` sentinels must match the retained ref and source name,
    after which the HTTP export boundary verifies live blob custody and the exact
    private storage path before returning the sidecar.
    """
    guided = state.guided_session
    if guided is None or not guided.reviewed_sources:
        return state

    reviewed_bindings: list[tuple[str, frozenset[str], str]] = []
    sentinel_bindings: list[tuple[str, str]] = []
    reviewed_names: set[str] = set()
    for snapshot in guided.reviewed_sources.values():
        source_name = snapshot.name
        if source_name in reviewed_names:
            raise AuditIntegrityError("guided reviewed source names must be unique")
        reviewed_names.add(source_name)
        snapshot_options = snapshot.options
        if "blob_ref" not in snapshot_options:
            continue
        blob_ref, blob_backed_paths = validate_guided_reviewed_blob_binding(snapshot_options)
        sentinel_paths = frozenset(path for path in blob_backed_paths if path.startswith("blob:"))
        if sentinel_paths:
            if sentinel_paths != blob_backed_paths:
                raise AuditIntegrityError("guided reviewed blob source mixes public sentinels and private paths")
            sentinel_ids = {validate_guided_reviewed_blob_ref(sentinel.removeprefix("blob:")) for sentinel in sentinel_paths}
            if sentinel_ids != {blob_ref}:
                raise AuditIntegrityError("guided reviewed blob sentinel and blob_ref differ")
            sentinel_bindings.append((source_name, blob_ref))
        else:
            reviewed_bindings.append((source_name, blob_backed_paths, blob_ref))

    if not reviewed_bindings and not sentinel_bindings:
        return state
    validate_guided_reviewed_blob_source_mapping(
        [(name, paths) for name, paths, _blob_ref in reviewed_bindings],
        {name: source.options for name, source in state.sources.items()},
    )
    all_reviewed_paths = frozenset(path for _name, paths, _blob_ref in reviewed_bindings for path in paths)
    reattached = dict(state.sources)
    changed = False
    for source_name, source in state.sources.items():
        live_reviewed_paths = {
            value for key in SOURCE_LOCAL_PATH_OPTION_KEYS if type(value := source.options.get(key)) is str and value in all_reviewed_paths
        }
        if not live_reviewed_paths:
            continue
        candidates = [
            (paths, blob_ref)
            for reviewed_name, paths, blob_ref in reviewed_bindings
            if reviewed_name == source_name and live_reviewed_paths <= paths
        ]
        if len(candidates) != 1:
            raise AuditIntegrityError("guided blob source mapping is inconsistent")
        _reviewed_paths, blob_ref = candidates[0]
        options = source.options
        if "blob_ref" in options:
            if options["blob_ref"] != blob_ref:
                raise AuditIntegrityError("guided blob source mapping is inconsistent")
            continue
        merged = dict(options)
        merged["blob_ref"] = blob_ref
        reattached[source_name] = replace(source, options=merged)
        changed = True

    for source_name, blob_ref in sentinel_bindings:
        sentinel_source = state.sources.get(source_name)
        if sentinel_source is None:
            raise AuditIntegrityError("guided blob source mapping is inconsistent")
        live_carriers = [sentinel_source.options[key] for key in GUIDED_REVIEWED_BLOB_PATH_KEYS if key in sentinel_source.options]
        if not live_carriers or any(type(value) is not str or not value or "\x00" in value for value in live_carriers):
            raise AuditIntegrityError("guided blob source mapping is inconsistent")
        options = sentinel_source.options
        if "blob_ref" in options:
            if options["blob_ref"] != blob_ref:
                raise AuditIntegrityError("guided blob source mapping is inconsistent")
            continue
        merged = dict(options)
        merged["blob_ref"] = blob_ref
        reattached[source_name] = replace(sentinel_source, options=merged)
        changed = True

    return replace(state, sources=reattached) if changed else state


def generate_public_pipeline_dict(state: CompositionState) -> dict[str, Any]:
    """Convert a CompositionState to public export/share/MCP pipeline dict."""
    export_state = reattach_guided_blob_refs_for_public_export(state)
    return _generate_pipeline_dict(export_state, omit_blob_bound_source_paths=True)


def generate_yaml(state: CompositionState) -> str:
    """Convert a CompositionState to deterministic ELSPETH pipeline YAML.

    The output is deterministic: same state produces byte-identical YAML.
    YAML serialization is a thin wrapper around ``generate_pipeline_dict()``
    so there is only one mapping from composer state to runtime/YAML shape.

    Args:
        state: The pipeline composition state to serialize.

    Returns:
        YAML string representing the pipeline configuration.
    """
    doc = generate_pipeline_dict(state)

    # sort_keys=False preserves insertion order: source → transforms →
    # gates → aggregations → coalesce → sinks (the natural pipeline flow).
    return yaml.dump(doc, default_flow_style=False, sort_keys=False)


def generate_public_yaml(state: CompositionState) -> str:
    """Convert a CompositionState to deterministic public export/share/MCP YAML."""
    doc = generate_public_pipeline_dict(state)
    return yaml.dump(doc, default_flow_style=False, sort_keys=False)
