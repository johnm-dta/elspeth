"""Runtime YAML to composer-state import helpers.

This is the reverse of the subset emitted by ``yaml_generator`` for hard-mode
replay. It reconstructs the declarative routing fields needed by validate and
execute; editor-only UI edges are intentionally left empty.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

import yaml

from elspeth.contracts.trust_boundary import trust_boundary
from elspeth.web.composer.state import (
    CompositionState,
    NodeSpec,
    NodeType,
    OutputSpec,
    PipelineMetadata,
    SourceSpec,
)

MAX_RUNTIME_YAML_IMPORT_CHARS = 262_144
_UNSUPPORTED_COALESCE_FIELDS = frozenset(
    {
        "union_collision_policy",
        "timeout_seconds",
        "quorum_count",
        "select_branch",
    }
)
# Recognised top-level pipeline sections. A parsed document that is a
# mapping but shares none of these keys is not a pipeline export at all
# (see the guard in composition_state_from_runtime_yaml).
_PIPELINE_SECTION_KEYS = frozenset({"source", "sources", "transforms", "gates", "aggregations", "coalesce", "sinks"})


class RuntimeYamlImportError(ValueError):
    """Raised when runtime YAML cannot be represented as composer state."""


def _reject_yaml_aliases(pipeline_yaml: str) -> None:
    """Reject YAML anchors/aliases before safe construction.

    Neither the eval-replay harness nor the composer's own export
    (:func:`elspeth.web.composer.yaml_generator.generate_public_yaml`) ever
    emits anchors — each field is written out in full on every dump. PyYAML's
    anchor/alias machinery lets an alias reuse an already-composed node by
    object identity rather than re-parsing it, so a document well under
    ``MAX_RUNTIME_YAML_IMPORT_CHARS`` can compose into a logical structure
    many orders of magnitude larger once something downstream (``dict()``
    copies, ``state.to_dict()``, JSON persistence) walks every reference
    instead of sharing it — a "billion laughs"-style amplification the
    character cap does not bound. Rejecting aliases outright removes the
    amplification vector for user-pasted YAML with no loss of legitimate
    functionality.
    """
    for event in yaml.parse(pipeline_yaml):
        if isinstance(event, yaml.events.AliasEvent):
            raise yaml.composer.ComposerError(
                None,
                None,
                "aliases are not permitted in pipeline YAML imports",
                event.start_mark,
            )


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return cast(Mapping[str, Any], value)
    raise RuntimeYamlImportError(f"{path} must be a mapping, got {type(value).__name__}")


def _optional_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    return _require_mapping(value, path)


def _require_sequence(value: Any, path: str) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    raise RuntimeYamlImportError(f"{path} must be a list, got {type(value).__name__}")


@trust_boundary(
    tier=3,
    source="web-authored pipeline YAML entry mapping (untrusted import payload)",
    source_param="entry",
    suppresses=("R1", "R5"),
    invariant="raises RuntimeYamlImportError unless the key holds a non-empty string",
    test_ref="tests/unit/web/composer/test_yaml_importer.py::test_require_str_rejects_non_string_value",
    test_fingerprint="94159e1bc6a7e13e0eeb7e72912b0a3fa036bf3091793e61c1893ab75b61722f",
)
def _require_str(entry: Mapping[str, Any], key: str, path: str) -> str:
    value = entry.get(key)
    if isinstance(value, str) and value:
        return value
    raise RuntimeYamlImportError(f"{path}.{key} must be a non-empty string")


def _optional_str(entry: Mapping[str, Any], key: str) -> str | None:
    value = entry.get(key)
    if value is None:
        return None
    if isinstance(value, str):
        return value
    raise RuntimeYamlImportError(f"{key} must be a string when provided")


def _route_label(value: Any) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str) and value:
        return value
    raise RuntimeYamlImportError("route labels must be non-empty strings")


def _string_mapping(value: Any, path: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw_key, raw_value in _require_mapping(value, path).items():
        if not isinstance(raw_value, str) or not raw_value:
            raise RuntimeYamlImportError(f"{path}.{raw_key} must be a non-empty string")
        result[_route_label(raw_key)] = raw_value
    return result


def _string_tuple(value: Any, path: str) -> tuple[str, ...]:
    items = []
    for index, item in enumerate(_require_sequence(value, path)):
        if not isinstance(item, str) or not item:
            raise RuntimeYamlImportError(f"{path}[{index}] must be a non-empty string")
        items.append(item)
    return tuple(items)


@trust_boundary(
    tier=3,
    source="web-authored pipeline YAML source entry (untrusted import payload)",
    source_param="entry",
    suppresses=("R1", "R5"),
    invariant=(
        "raises RuntimeYamlImportError on non-mapping entries, missing plugin/on_success, "
        "or inline blob_ref; a missing or malformed on_validation_failure falls back to 'discard'"
    ),
    test_ref="tests/unit/web/composer/test_yaml_importer.py::test_source_from_runtime_entry_rejects_non_mapping_entry",
    test_fingerprint="464ba44200f35d34d3eed43f62f4a50595385f53a9879dde0a9bd1e898dc8bae",
)
def _source_from_runtime_entry(source_name: str, entry: Any) -> SourceSpec:
    source = _require_mapping(entry, f"sources.{source_name}")
    options = dict(_optional_mapping(source.get("options"), f"sources.{source_name}.options"))
    if "blob_ref" in options:
        raise RuntimeYamlImportError(f"sources.{source_name}.options.blob_ref must be supplied via source_blob_ids")
    on_validation_failure = source.get("on_validation_failure")
    option_on_validation_failure = options.pop("on_validation_failure", None)
    if on_validation_failure is None:
        on_validation_failure = option_on_validation_failure
    if not isinstance(on_validation_failure, str) or not on_validation_failure:
        on_validation_failure = "discard"
    return SourceSpec(
        plugin=_require_str(source, "plugin", f"sources.{source_name}"),
        on_success=_require_str(source, "on_success", f"sources.{source_name}"),
        options=options,
        on_validation_failure=on_validation_failure,
    )


@trust_boundary(
    tier=3,
    source="web-authored pipeline YAML node section (untrusted import payload)",
    source_param="section",
    suppresses=("R1", "R5"),
    invariant=(
        "raises RuntimeYamlImportError on non-sequence sections, non-mapping entries, and "
        "missing or mistyped required node fields; optional fields (routes, fork_to, "
        "branches, trigger) are read permissively and validated when present"
    ),
    test_ref="tests/unit/web/composer/test_yaml_importer.py::test_nodes_from_runtime_list_rejects_non_sequence_section",
    test_fingerprint="7c2c5adda9b620f4628b42f0ece20f0b6c403e3fad0890a22ba38dacc86b0a3d",
)
def _nodes_from_runtime_list(section: Any, section_name: str, node_type: NodeType) -> list[NodeSpec]:
    if section is None:
        return []
    nodes: list[NodeSpec] = []
    for index, raw_entry in enumerate(_require_sequence(section, section_name)):
        path = f"{section_name}[{index}]"
        entry = _require_mapping(raw_entry, path)
        if node_type == "gate":
            routes = entry.get("routes")
            fork_to = entry.get("fork_to")
            nodes.append(
                NodeSpec(
                    id=_require_str(entry, "name", path),
                    node_type=node_type,
                    plugin=None,
                    input=_require_str(entry, "input", path),
                    on_success=None,
                    on_error=None,
                    options={},
                    condition=_require_str(entry, "condition", path),
                    routes=_string_mapping(routes, f"{path}.routes") if routes is not None else None,
                    fork_to=_string_tuple(fork_to, f"{path}.fork_to") if fork_to is not None else None,
                    branches=None,
                    policy=None,
                    merge=None,
                )
            )
            continue
        if node_type == "coalesce":
            unsupported = sorted(_UNSUPPORTED_COALESCE_FIELDS & entry.keys())
            if unsupported:
                raise RuntimeYamlImportError(
                    f"{path} uses unsupported coalesce field(s) that composer state cannot preserve: {unsupported}"
                )
            branches = entry.get("branches")
            branch_spec: tuple[str, ...] | dict[str, str] | None
            if isinstance(branches, Mapping):
                branch_spec = _string_mapping(branches, f"{path}.branches")
            elif branches is None:
                branch_spec = None
            else:
                branch_spec = _string_tuple(branches, f"{path}.branches")
            nodes.append(
                NodeSpec(
                    id=_require_str(entry, "name", path),
                    node_type=node_type,
                    plugin=None,
                    input=str(entry.get("input") or ""),
                    on_success=_optional_str(entry, "on_success"),
                    on_error=None,
                    options={},
                    condition=None,
                    routes=None,
                    fork_to=None,
                    branches=branch_spec,
                    policy=_require_str(entry, "policy", path),
                    merge=_require_str(entry, "merge", path),
                )
            )
            continue
        options = dict(_optional_mapping(entry.get("options"), f"{path}.options"))
        expected_output_count = entry.get("expected_output_count")
        if expected_output_count is not None and (not isinstance(expected_output_count, int) or isinstance(expected_output_count, bool)):
            raise RuntimeYamlImportError(f"{path}.expected_output_count must be an integer when provided")
        nodes.append(
            NodeSpec(
                id=_require_str(entry, "name", path),
                node_type=node_type,
                plugin=_require_str(entry, "plugin", path),
                input=_require_str(entry, "input", path),
                on_success=_optional_str(entry, "on_success") if node_type == "aggregation" else _require_str(entry, "on_success", path),
                on_error=_require_str(entry, "on_error", path),
                options=options,
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
                trigger=dict(_optional_mapping(entry.get("trigger"), f"{path}.trigger")) if entry.get("trigger") is not None else None,
                output_mode=_optional_str(entry, "output_mode"),
                expected_output_count=expected_output_count,
            )
        )
    return nodes


@trust_boundary(
    tier=3,
    source="web-authored pipeline YAML sinks mapping (untrusted import payload)",
    source_param="sinks",
    suppresses=("R1", "R5"),
    invariant=("raises RuntimeYamlImportError on non-mapping sinks, non-string sink names, and missing plugin/on_write_failure fields"),
    test_ref="tests/unit/web/composer/test_yaml_importer.py::test_outputs_from_runtime_sinks_rejects_non_mapping_sinks",
    test_fingerprint="b44ac151ec41c1923352de32ff62370b062c753adf1edd1a973c46aa3bef64a7",
)
def _outputs_from_runtime_sinks(sinks: Any) -> tuple[OutputSpec, ...]:
    if sinks is None:
        return ()
    outputs = []
    for sink_name, raw_entry in _require_mapping(sinks, "sinks").items():
        if not isinstance(sink_name, str) or not sink_name:
            raise RuntimeYamlImportError("sinks keys must be non-empty strings")
        entry = _require_mapping(raw_entry, f"sinks.{sink_name}")
        outputs.append(
            OutputSpec(
                name=sink_name,
                plugin=_require_str(entry, "plugin", f"sinks.{sink_name}"),
                options=dict(_optional_mapping(entry.get("options"), f"sinks.{sink_name}.options")),
                on_write_failure=_require_str(entry, "on_write_failure", f"sinks.{sink_name}"),
            )
        )
    return tuple(outputs)


@trust_boundary(
    tier=3,
    source="operator-supplied runtime pipeline YAML import request",
    source_param="pipeline_yaml",
    suppresses=("R1", "R5"),
    invariant="raises RuntimeYamlImportError on malformed or unsupported YAML; never coerces non-mapping roots",
    test_ref="tests/unit/web/composer/test_yaml_importer.py::test_composition_state_from_runtime_yaml_rejects_non_mapping_root",
    test_fingerprint="31da4d90277422f518daf165ce0e90aeeed106c3333a39447a848982aeedc795",
)
def composition_state_from_runtime_yaml(pipeline_yaml: str, *, version: int = 1) -> CompositionState:
    """Build a composer state from runtime pipeline YAML."""
    if len(pipeline_yaml) > MAX_RUNTIME_YAML_IMPORT_CHARS:
        raise RuntimeYamlImportError("pipeline YAML exceeds the 262144 character import limit")
    try:
        _reject_yaml_aliases(pipeline_yaml)
        parsed = yaml.safe_load(pipeline_yaml)
    except (yaml.YAMLError, RecursionError) as exc:
        # RecursionError: a deeply (but not textually large) nested mapping
        # chain -- e.g. ~500 levels of ``a:\n  a:\n    a:\n ...`` fits well
        # under the character cap above but exhausts CPython's default
        # recursion limit inside PyYAML's pure-Python composer/constructor.
        # ``yaml.safe_load`` uses PyYAML's safe constructors, so this is a
        # plain, catchable parser/constructor failure rather than arbitrary
        # object construction.
        raise RuntimeYamlImportError(f"YAML parse failed: {exc.__class__.__name__}") from exc
    doc = _require_mapping(parsed, "pipeline YAML")
    if not _PIPELINE_SECTION_KEYS & doc.keys():
        # A pasted document can be syntactically valid YAML *and* a mapping
        # (so it clears _require_mapping above) while describing nothing
        # pipeline-shaped at all -- a grocery list, an unrelated config file,
        # `{}`. Without this gate that silently imports as an empty
        # CompositionState (no sources, no nodes, no outputs), which
        # save_composition_state then persists as the session's new current
        # version -- a silent destructive replace of whatever composition
        # was there before, with no error to signal anything went wrong.
        raise RuntimeYamlImportError(
            "pipeline YAML must define at least one pipeline section (sources, transforms, gates, aggregations, coalesce, or sinks)"
        )

    raw_sources = doc.get("sources")
    if raw_sources is None and doc.get("source") is not None:
        raw_sources = {"source": doc["source"]}
    sources = {}
    for source_name, source_entry in _optional_mapping(raw_sources, "sources").items():
        if not isinstance(source_name, str) or not source_name:
            raise RuntimeYamlImportError("sources keys must be non-empty strings")
        sources[source_name] = _source_from_runtime_entry(source_name, source_entry)

    nodes = [
        *_nodes_from_runtime_list(doc.get("transforms"), "transforms", "transform"),
        *_nodes_from_runtime_list(doc.get("gates"), "gates", "gate"),
        *_nodes_from_runtime_list(doc.get("aggregations"), "aggregations", "aggregation"),
        *_nodes_from_runtime_list(doc.get("coalesce"), "coalesce", "coalesce"),
    ]

    metadata = PipelineMetadata()
    if isinstance(doc.get("metadata"), Mapping):
        raw_metadata = cast(Mapping[str, Any], doc["metadata"])
        metadata = PipelineMetadata(
            name=str(raw_metadata.get("name") or metadata.name),
            description=str(raw_metadata.get("description") or metadata.description),
        )

    return CompositionState(
        sources=sources,
        nodes=tuple(nodes),
        edges=(),
        outputs=_outputs_from_runtime_sinks(doc.get("sinks")),
        metadata=metadata,
        version=version,
    )
