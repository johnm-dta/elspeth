"""Canonical capability guidance and per-call planner integrity manifests."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final, cast

from elspeth.contracts.errors import AuditIntegrityError
from elspeth.core.canonical import canonical_json, stable_hash
from elspeth.web.composer.pipeline_proposal import PlannerSurface
from elspeth.web.composer.skills import load_skill_with_hash

PLANNER_DISCOVERY_TOOL_NAMES: Final[tuple[str, ...]] = (
    "diff_pipeline",
    "explain_validation_error",
    "get_audit_info",
    "get_expression_grammar",
    "get_pipeline_state",
    "get_plugin_assistance",
    "get_plugin_schema",
    "list_models",
    "list_recipes",
    "list_sinks",
    "list_sources",
    "list_transforms",
    "preview_pipeline",
    "get_blob_content",
    "get_blob_metadata",
    "inspect_source",
    "list_blobs",
    "list_composer_blobs",
    "list_secret_refs",
    "validate_secret_ref",
)
PLANNER_TERMINAL_TOOL_NAME: Final[str] = "emit_pipeline_proposal"
PLANNER_IMPLEMENTATION_ID: Final[str] = "elspeth.web.composer.pipeline_planner.plan_pipeline"

CAPABILITY_CORE_NODE_GUIDANCE: Final[Mapping[str, str]] = MappingProxyType(
    {
        "aggregation": "[capability-node:aggregation]",
        "coalesce": "[capability-node:coalesce]",
        "gate": "[capability-node:gate]",
        "queue": "[capability-node:queue]",
        "transform": "[capability-node:transform]",
    }
)

_FIELD_INVENTORY_START: Final[str] = "<!-- canonical-field-inventory:start -->"
_FIELD_INVENTORY_END: Final[str] = "<!-- canonical-field-inventory:end -->"
_FIELD_ROW_RE: Final[re.Pattern[str]] = re.compile(r"^\| ([a-z_]+) \| (`[a-z_]+`(?:, `[a-z_]+`)*) \|$")


def documented_capability_fields(text: str) -> Mapping[str, frozenset[str]]:
    """Parse and validate the public canonical-field inventory.

    The inventory is the human-visible authority.  Python derives its field
    mapping from those exact bytes, then the planner compares it with the
    terminal schema on every call.  There is deliberately no second list of
    field names in code that can drift in lockstep with the schema.
    """
    if type(text) is not str:
        raise TypeError("text must be an exact string")
    if text.count(_FIELD_INVENTORY_START) != 1 or text.count(_FIELD_INVENTORY_END) != 1:
        raise AuditIntegrityError("canonical capability field inventory anchors must occur exactly once")
    remainder = text.split(_FIELD_INVENTORY_START, 1)[1]
    block = remainder.split(_FIELD_INVENTORY_END, 1)[0]
    lines = block.strip().splitlines()
    if lines[:2] != ["| Family | Fields |", "| --- | --- |"]:
        raise AuditIntegrityError("canonical capability field inventory header is malformed")
    extracted: dict[str, frozenset[str]] = {}
    for line in lines[2:]:
        match = _FIELD_ROW_RE.fullmatch(line)
        if match is None:
            raise AuditIntegrityError("canonical capability field inventory row is malformed")
        family, rendered_fields = match.groups()
        if family in extracted:
            raise AuditIntegrityError("canonical capability field inventory repeats a family")
        extracted[family] = frozenset(item.removeprefix("`").removesuffix("`") for item in rendered_fields.split(", "))
    return MappingProxyType(extracted)


def _schema_mapping(value: object, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AuditIntegrityError(f"canonical capability schema node {path} must be an object")
    return cast(Mapping[str, Any], value)


def _property_names(schema: Mapping[str, Any], *, path: str) -> frozenset[str]:
    properties = _schema_mapping(schema.get("properties"), path=f"{path}.properties")
    if any(type(name) is not str for name in properties):
        raise AuditIntegrityError(f"canonical capability schema property names at {path} must be exact strings")
    return frozenset(properties)


def canonical_capability_fields(schema: Mapping[str, Any]) -> Mapping[str, frozenset[str]]:
    """Extract every authoring-structural field from the terminal schema."""
    root = _schema_mapping(schema, path="$pipeline")
    properties = _schema_mapping(root.get("properties"), path="$pipeline.properties")
    source = _schema_mapping(properties.get("source"), path="$pipeline.properties.source")
    sources = _schema_mapping(properties.get("sources"), path="$pipeline.properties.sources")
    named_source = _schema_mapping(sources.get("additionalProperties"), path="$pipeline.properties.sources.*")
    nodes = _schema_mapping(properties.get("nodes"), path="$pipeline.properties.nodes")
    node = _schema_mapping(nodes.get("items"), path="$pipeline.properties.nodes.items")
    node_properties = _schema_mapping(node.get("properties"), path="$pipeline.properties.nodes.items.properties")
    trigger = _schema_mapping(node_properties.get("trigger"), path="$pipeline.properties.nodes.items.properties.trigger")
    edges = _schema_mapping(properties.get("edges"), path="$pipeline.properties.edges")
    edge = _schema_mapping(edges.get("items"), path="$pipeline.properties.edges.items")
    outputs = _schema_mapping(properties.get("outputs"), path="$pipeline.properties.outputs")
    output = _schema_mapping(outputs.get("items"), path="$pipeline.properties.outputs.items")
    metadata = _schema_mapping(properties.get("metadata"), path="$pipeline.properties.metadata")
    source_properties = _schema_mapping(source.get("properties"), path="$pipeline.properties.source.properties")
    inline_blob = _schema_mapping(source_properties.get("inline_blob"), path="$pipeline.properties.source.properties.inline_blob")
    extracted = {
        "pipeline": frozenset(properties),
        "source": _property_names(source, path="$pipeline.source"),
        "named_source": _property_names(named_source, path="$pipeline.sources.*"),
        "inline_blob": _property_names(inline_blob, path="$pipeline.source.inline_blob"),
        "node": frozenset(node_properties),
        "trigger": _property_names(trigger, path="$pipeline.nodes[].trigger"),
        "edge": _property_names(edge, path="$pipeline.edges[]"),
        "output": _property_names(output, path="$pipeline.outputs[]"),
        "metadata": _property_names(metadata, path="$pipeline.metadata"),
    }
    return MappingProxyType(extracted)


def validate_capability_field_contract(schema: Mapping[str, Any], documented_text: str) -> None:
    """Fail when the visible field table and canonical schema do not agree."""
    if canonical_capability_fields(schema) != documented_capability_fields(documented_text):
        raise AuditIntegrityError("canonical pipeline schema and documented capability fields drifted")


def load_pipeline_capability_core() -> str:
    """Return the one static public capability core as exact UTF-8 text."""
    text, _digest = load_skill_with_hash("pipeline_capabilities")
    return text


# Derived from the public, visible inventory rather than duplicated in code.
CANONICAL_CAPABILITY_FIELDS: Final[Mapping[str, frozenset[str]]] = documented_capability_fields(load_pipeline_capability_core())


def render_with_pipeline_capabilities(interaction_skill: str) -> str:
    """Prepend the exact capability-core bytes to one interaction overlay."""
    if type(interaction_skill) is not str or not interaction_skill.strip():
        raise ValueError("interaction_skill must be a non-empty exact string")
    return f"{load_pipeline_capability_core()}\n{interaction_skill.rstrip()}\n"


@dataclass(frozen=True, slots=True)
class PlannerCapabilityManifest:
    """Hash-only audit facts derived from one exact outbound planner call."""

    surface: PlannerSurface
    profile: str
    planner_implementation_id: str
    capability_core_hash: str
    canonical_schema_hash: str
    effective_tool_hash: str
    rendered_prompt_hash: str

    def __post_init__(self) -> None:
        if type(self.surface) is not PlannerSurface:
            raise TypeError("surface must be an exact PlannerSurface")
        if self.profile not in {"ordinary", "tutorial"}:
            raise ValueError("profile must be 'ordinary' or 'tutorial'")
        if type(self.planner_implementation_id) is not str or not self.planner_implementation_id:
            raise ValueError("planner_implementation_id must be a non-empty exact string")
        for name in (
            "capability_core_hash",
            "canonical_schema_hash",
            "effective_tool_hash",
            "rendered_prompt_hash",
        ):
            value = getattr(self, name)
            if type(value) is not str or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _expected_profile(surface: PlannerSurface) -> str:
    return "tutorial" if surface is PlannerSurface.TUTORIAL_PROFILE else "ordinary"


def build_planner_capability_manifest(
    *,
    surface: PlannerSurface,
    profile: str,
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]],
    canonical_schema: Mapping[str, Any],
) -> PlannerCapabilityManifest:
    """Validate and hash the exact messages/tools used by one planner call."""
    if type(surface) is not PlannerSurface:
        raise TypeError("surface must be an exact PlannerSurface")
    if profile != _expected_profile(surface):
        raise AuditIntegrityError("planner surface/profile identity mismatch")
    canonical_json(messages)
    canonical_json(tools)
    core = load_pipeline_capability_core()
    system_contents = [message.get("content") for message in messages if message.get("role") == "system"]
    if not system_contents or type(system_contents[0]) is not str or not system_contents[0].startswith(core):
        raise AuditIntegrityError("planner capability core is missing from the first system message")
    if any(type(content) is not str for content in system_contents):
        raise AuditIntegrityError("planner system message content must be exact text")
    if sum(cast(str, content).count(core) for content in system_contents) != 1:
        raise AuditIntegrityError("planner capability core must occur exactly once")

    try:
        tool_names = tuple(cast(str, tool["function"]["name"]) for tool in tools)
    except (KeyError, TypeError) as exc:
        raise AuditIntegrityError("planner advertised tool definitions are malformed") from exc
    expected_names = (*PLANNER_DISCOVERY_TOOL_NAMES, PLANNER_TERMINAL_TOOL_NAME)
    if tool_names != expected_names:
        raise AuditIntegrityError("planner advertised tool identities or order drifted")
    terminal = _schema_mapping(tools[-1].get("function"), path="$tools[-1].function")
    parameters = _schema_mapping(terminal.get("parameters"), path="$tools[-1].function.parameters")
    properties = _schema_mapping(parameters.get("properties"), path="$tools[-1].function.parameters.properties")
    advertised_schema = _schema_mapping(properties.get("pipeline"), path="$tools[-1].function.parameters.properties.pipeline")
    if stable_hash(advertised_schema) != stable_hash(canonical_schema):
        raise AuditIntegrityError("planner terminal does not advertise the canonical pipeline schema")
    validate_capability_field_contract(canonical_schema, core)

    return PlannerCapabilityManifest(
        surface=surface,
        profile=profile,
        planner_implementation_id=PLANNER_IMPLEMENTATION_ID,
        capability_core_hash=hashlib.sha256(core.encode("utf-8")).hexdigest(),
        canonical_schema_hash=stable_hash(advertised_schema),
        effective_tool_hash=stable_hash(tools),
        rendered_prompt_hash=stable_hash(messages),
    )


__all__ = [
    "CANONICAL_CAPABILITY_FIELDS",
    "CAPABILITY_CORE_NODE_GUIDANCE",
    "PLANNER_DISCOVERY_TOOL_NAMES",
    "PLANNER_IMPLEMENTATION_ID",
    "PLANNER_TERMINAL_TOOL_NAME",
    "PlannerCapabilityManifest",
    "build_planner_capability_manifest",
    "canonical_capability_fields",
    "documented_capability_fields",
    "load_pipeline_capability_core",
    "render_with_pipeline_capabilities",
    "validate_capability_field_contract",
]
