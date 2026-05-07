"""Deterministic YAML generator -- CompositionState to ELSPETH pipeline YAML.

Pure function. Same CompositionState always produces byte-identical YAML.
Uses yaml.dump() with sort_keys=True for determinism.

Layer: L3 (application). Imports from L3 (web/composer/state) only.

Trust model: state_dict comes from CompositionState.to_dict() — our own
serialization of our own frozen dataclasses. Fields are either always
present (direct access) or conditionally present based on node_type
(check with ``in``). Never use .get() — a missing field is a bug in
to_dict(), not an expected absence.

Web-specific metadata keys (e.g., blob_ref for file provenance tracking)
are filtered from options before YAML generation. These are UI-layer
concerns that should not leak into engine configuration. Plugin configs
use Pydantic with extra="forbid" — unknown keys cause validation failure.
"""

from __future__ import annotations

from typing import Any

import yaml

from elspeth.web.composer.state import CompositionState

# Web-specific metadata keys that should NOT appear in engine YAML.
# These are UI-layer concerns for provenance tracking, not plugin config.
# Plugin configs use Pydantic with extra="forbid" — unknown keys cause errors.
_WEB_ONLY_OPTION_KEYS = frozenset({"blob_ref"})


def _strip_web_metadata(options: dict[str, Any]) -> dict[str, Any]:
    """Remove web-specific metadata keys from options dict.

    Returns a shallow copy with web-only keys removed.
    """
    return {k: v for k, v in options.items() if k not in _WEB_ONLY_OPTION_KEYS}


def generate_yaml(state: CompositionState) -> str:
    """Convert a CompositionState to ELSPETH pipeline YAML.

    The output is deterministic: same state produces byte-identical YAML.
    Maps CompositionState fields to the YAML structure expected by
    ELSPETH's load_settings() parser.

    Calls state.to_dict() to unwrap all frozen containers
    (MappingProxyType -> dict, tuple -> list) before passing to
    yaml.dump(). This avoids RepresenterError from PyYAML on frozen
    types. See spec R4 and AC #15.

    Args:
        state: The pipeline composition state to serialize.

    Returns:
        YAML string representing the pipeline configuration.
    """
    # Unwrap frozen containers to plain Python types (R4).
    # to_dict() recursively converts MappingProxyType -> dict,
    # tuple -> list. Without this, yaml.dump() raises RepresenterError.
    state_dict = state.to_dict()

    doc: dict[str, Any] = {}

    # Source — state_dict["source"] is always present (None or dict).
    source = state_dict["source"]
    if source is not None:
        # Strip web-specific metadata (e.g., blob_ref) before generating engine YAML.
        source_options = _strip_web_metadata(dict(source["options"]))
        source_options["on_validation_failure"] = source["on_validation_failure"]
        doc["source"] = {
            "plugin": source["plugin"],
            "on_success": source["on_success"],
            "options": source_options,
        }

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
                entry["options"] = t["options"]
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
                entry["options"] = a["options"]
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
                sink_entry["options"] = output["options"]
            doc["sinks"][output["name"]] = sink_entry

    # landscape key is intentionally omitted -- URL comes from
    # WebSettings.get_landscape_url() at execution time (security fix S1).

    # sort_keys=False preserves insertion order: source → transforms →
    # gates → aggregations → coalesce → sinks (the natural pipeline flow).
    return yaml.dump(doc, default_flow_style=False, sort_keys=False)
