"""Composition tools — discovery and mutation tools for the LLM composer.

Discovery tools delegate to CatalogService. Mutation tools modify
CompositionState and return ToolResult with validation.

Layer: L3 (application). Imports from L0 (contracts.freeze) and
L3 (web/composer/state, web/catalog/protocol).
"""

from __future__ import annotations

import ast
import asyncio
import csv
import hmac
import io
import json
import os
import re
import tempfile
import threading
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, TypedDict, cast
from uuid import UUID, uuid4

from opentelemetry import metrics
from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError
from sqlalchemy import Engine, delete, func, select, update

from elspeth.contracts.composer_interpretation import InterpretationEventRecord, InterpretationSource
from elspeth.contracts.enums import CreationModality, is_llm_authored_creation_modality
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.freeze import deep_thaw, freeze_fields
from elspeth.contracts.schema import get_aggregation_contract_options
from elspeth.core.config import TriggerConfig
from elspeth.core.secrets import collect_credential_field_violations, collect_disallowed_secret_ref_markers
from elspeth.web.blobs.protocol import BlobIntegrityError
from elspeth.web.blobs.service import (
    _ACTIVE_RUN_COMPOSITION_COLUMNS,
    _active_run_pipeline_dict,
    _composition_references_blob,
    _guard_blob_row_literals,
    content_hash,
    sanitize_filename,
)
from elspeth.web.catalog.protocol import CatalogService, PluginKind
from elspeth.web.composer.protocol import ToolArgumentError
from elspeth.web.composer.recipes import (
    RecipeValidationError,
    apply_recipe,
    list_recipes,
)
from elspeth.web.composer.redaction import (
    ApplyPipelineRecipeArgumentsModel,
    CreateBlobArgumentsModel,
    PatchNodeOptionsArgumentsModel,
    PatchOutputOptionsArgumentsModel,
    PatchSourceOptionsArgumentsModel,
    SetPipelineArgumentsModel,
    SetSourceArgumentsModel,
    SetSourceFromBlobArgumentsModel,
    UpdateBlobArgumentsModel,
    redact_source_storage_path,
)
from elspeth.web.composer.source_inspection import (
    derive_extra_column_risk,
    facts_to_dict,
    inspect_blob_content,
)
from elspeth.web.composer.state import (
    CoalesceBranches,
    CompositionState,
    EdgeSpec,
    EdgeType,
    NodeSpec,
    NodeType,
    OutputSpec,
    PipelineMetadata,
    SourceSpec,
    ValidationEntry,
    ValidationSummary,
    _batch_aware_placement_error,
    _batch_aware_required_input_fields_error,
    _coalesce_branch_connections,
    _coalesce_branch_names,
    _serialize_branches,
    _source_options_have_schema,
    _validate_gate_expression,
)
from elspeth.web.execution.schemas import ValidationResult
from elspeth.web.paths import allowed_sink_directories, allowed_source_directories, resolve_data_path
from elspeth.web.secrets.ref_policy import allowed_secret_ref_fields, allowed_secret_ref_fields_text
from elspeth.web.sessions.models import blob_run_links_table, blobs_table, composition_states_table, runs_table
from elspeth.web.validation import (
    INTERPRETATION_PLACEHOLDER_RE,
    _reject_credential_shaped_content,
    _validate_accepted_value_content,
)

if TYPE_CHECKING:
    pass

# Module-level OTel counter for authoring validation outcomes in preview_pipeline.
# Attributes: outcome (valid | invalid)
_AUTHORING_VALIDATION_COUNTER = metrics.get_meter(__name__).create_counter(
    "composer.authoring_validation.total",
    description="Total authoring (Stage 1) validation outcomes from preview_pipeline",
)

_FULL_STATE_COMPONENT_ALIASES: Final[tuple[str, ...]] = ("", "full", "all", "pipeline")
_FULL_STATE_COMPONENT_ALIAS_SET: Final[frozenset[str]] = frozenset(_FULL_STATE_COMPONENT_ALIASES)
_NODE_ROUTING_OPTION_PATCH_KEYS: Final[frozenset[str]] = frozenset({"input", "on_success", "on_error", "routes", "fork_to"})
_DEFAULT_SOURCE_VALIDATION_FAILURE: Final[str] = "discard"
_DATA_ERROR_KEY: Final[str] = "error"
_SOURCE_VALIDATION_FAILURE_DESCRIPTION: Final[str] = (
    "How to handle source validation failures. Use 'discard' to drop invalid rows without routing. "
    "Any other value, including 'quarantine', must match a configured output/sink name."
)


class _SemanticEdgeContractPayload(TypedDict):
    """Wire shape for a serialized SemanticEdgeContract.

    Mirrors composer_mcp.server._SemanticEdgeContractPayload and
    web.execution.schemas.SemanticEdgeContractResponse exactly so HTTP,
    MCP, and ToolResult surfaces stay identical modulo transport
    envelope. If a field changes here, change it in all three places.
    """

    from_id: str
    to_id: str
    consumer_plugin: str
    producer_plugin: str | None
    producer_field: str
    consumer_field: str
    outcome: str
    requirement_code: str


class _FullPipelineStateMetadataPayload(TypedDict):
    """Metadata payload nested in full get_pipeline_state responses."""

    name: str | None
    description: str | None


class _FullPipelineStateInspectionPayload(TypedDict):
    """Inspection payload documenting how a full-state alias resolved."""

    requested_component: Any
    resolved_component: str
    accepted_full_state_aliases: list[str]


class _FullPipelineStatePayload(TypedDict):
    """Full-state payload returned by get_pipeline_state."""

    source: dict[str, Any] | None
    sources: dict[str, dict[str, Any]]
    nodes: list[dict[str, Any]]
    outputs: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    metadata: _FullPipelineStateMetadataPayload
    version: int
    inspection: _FullPipelineStateInspectionPayload


class _RepairToolCall(TypedDict):
    tool: str
    arguments: Mapping[str, object]


class _AffectedConsumer(TypedDict):
    id: str
    current_input: str
    new_input: str


class _GraphRepairSuggestion(TypedDict):
    code: str
    connection: str
    strategy: str
    reason: str
    affected_consumers: list[_AffectedConsumer]
    tool_sequence: list[_RepairToolCall]


def _semantic_contracts_payload(
    contracts: tuple[Any, ...],
) -> list[_SemanticEdgeContractPayload]:
    """Serialize a SemanticEdgeContract tuple to JSON-friendly dicts.

    Centralized so ToolResult.to_dict and _execute_preview_pipeline
    emit identical shapes — and so adding a field updates both
    surfaces in one place.

    SemanticEdgeContract intentionally has no .to_dict() of its own:
    serialization happens at consumption sites so L0 stays free of
    JSON-encoding concerns. (See composer_mcp/server.py for the same
    pattern.)
    """
    return [
        _SemanticEdgeContractPayload(
            from_id=sc.from_id,
            to_id=sc.to_id,
            consumer_plugin=sc.consumer_plugin,
            producer_plugin=sc.producer_plugin,
            producer_field=sc.producer_field,
            consumer_field=sc.consumer_field,
            outcome=sc.outcome.value,
            requirement_code=sc.requirement.requirement_code,
        )
        for sc in contracts
    ]


def _compute_validation_delta(
    before: ValidationSummary,
    after: ValidationSummary,
) -> dict[str, Any]:
    """Compute new/resolved entries between two validation states.

    Compares by (component, message) tuple since ValidationEntry
    instances are recreated on each validate() call (no stable identity).
    """
    before_errors = {(e.component, e.message) for e in before.errors}
    after_errors = {(e.component, e.message) for e in after.errors}
    before_warnings = {(w.component, w.message) for w in before.warnings}
    after_warnings = {(w.component, w.message) for w in after.warnings}

    new_errors = [e.to_dict() for e in after.errors if (e.component, e.message) not in before_errors]
    resolved_errors = [e.to_dict() for e in before.errors if (e.component, e.message) not in after_errors]
    new_warnings = [w.to_dict() for w in after.warnings if (w.component, w.message) not in before_warnings]
    resolved_warnings = [w.to_dict() for w in before.warnings if (w.component, w.message) not in after_warnings]

    return {
        "new_errors": new_errors,
        "resolved_errors": resolved_errors,
        "new_warnings": new_warnings,
        "resolved_warnings": resolved_warnings,
    }


def _repair_identifier_fragment(value: str, *, fallback: str) -> str:
    """Return a connection-safe identifier fragment for generated repair skeletons."""
    fragment = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_-")
    if not fragment:
        return fallback
    if not fragment[0].isalnum():
        return f"{fallback}_{fragment}"
    return fragment


def _unique_name(candidate: str, reserved: set[str]) -> str:
    """Return candidate or a suffixed variant that does not collide with reserved."""
    if candidate not in reserved:
        reserved.add(candidate)
        return candidate

    index = 2
    while f"{candidate}_{index}" in reserved:
        index += 1
    unique = f"{candidate}_{index}"
    reserved.add(unique)
    return unique


def _reserved_connection_names(state: CompositionState) -> set[str]:
    """Collect existing route/connection/sink names a repair branch must avoid."""
    names: set[str] = {output.name for output in state.outputs}
    if state.source is not None:
        names.add(state.source.on_success)
        if state.source.on_validation_failure != "discard":
            names.add(state.source.on_validation_failure)

    for node in state.nodes:
        names.add(node.input)
        if node.on_success is not None:
            names.add(node.on_success)
        if node.on_error is not None and node.on_error != "discard":
            names.add(node.on_error)
        if node.routes is not None:
            names.update(node.routes.values())
        if node.fork_to is not None:
            names.update(node.fork_to)
        if node.branches is not None:
            names.update(_coalesce_branch_names(node.branches))
            names.update(_coalesce_branch_connections(node.branches))
    return names


def _duplicate_consumer_repair_suggestions(
    state: CompositionState,
    validation: ValidationSummary,
) -> list[_GraphRepairSuggestion]:
    """Build copyable repair skeletons for duplicate-consumer validation failures."""
    duplicate_error_components = {
        error.component
        for error in validation.errors
        if error.component.startswith("connection:") and error.message.startswith("Duplicate consumer for connection ")
    }
    if not duplicate_error_components:
        return []

    consumers_by_connection: dict[str, list[NodeSpec]] = {}
    for node in state.nodes:
        if node.node_type == "coalesce":
            continue
        if node.input not in consumers_by_connection:
            consumers_by_connection[node.input] = []
        consumers_by_connection[node.input].append(node)

    reserved_node_ids = {node.id for node in state.nodes}
    reserved_connection_names = _reserved_connection_names(state)
    suggestions: list[_GraphRepairSuggestion] = []

    for connection_name, consumer_nodes in sorted(consumers_by_connection.items()):
        if len(consumer_nodes) < 2 or f"connection:{connection_name}" not in duplicate_error_components:
            continue

        connection_fragment = _repair_identifier_fragment(connection_name, fallback="connection")
        gate_id = _unique_name(f"fork_{connection_fragment}", reserved_node_ids)
        branch_names = [
            _unique_name(
                f"{connection_fragment}_to_{_repair_identifier_fragment(node.id, fallback='node')}",
                reserved_connection_names,
            )
            for node in consumer_nodes
        ]
        gate_args: dict[str, object] = {
            "id": gate_id,
            "node_type": "gate",
            "plugin": None,
            "input": connection_name,
            "on_success": None,
            "on_error": None,
            "options": {},
            "condition": "True",
            "routes": {},
            "fork_to": branch_names,
            "branches": None,
            "policy": None,
            "merge": None,
            "trigger": None,
            "output_mode": None,
            "expected_output_count": None,
        }
        tool_sequence: list[_RepairToolCall] = []
        affected_consumers: list[_AffectedConsumer] = []
        for node, branch_name in zip(consumer_nodes, branch_names, strict=True):
            patched_consumer = _serialize_node(node)
            patched_consumer["input"] = branch_name
            tool_sequence.append({"tool": "upsert_node", "arguments": patched_consumer})
            affected_consumers.append(
                {
                    "id": node.id,
                    "current_input": connection_name,
                    "new_input": branch_name,
                }
            )
        tool_sequence.append({"tool": "upsert_node", "arguments": gate_args})
        tool_sequence.append({"tool": "preview_pipeline", "arguments": {}})
        suggestions.append(
            {
                "code": "duplicate_consumer_connection",
                "connection": connection_name,
                "strategy": "insert_fork_gate",
                "reason": "One connection can feed one processing node. Give each consumer a unique branch input, then insert a fork gate that consumes the shared connection and publishes those branch inputs from gate.fork_to.",
                "affected_consumers": affected_consumers,
                "tool_sequence": tool_sequence,
            }
        )

    return suggestions


def _graph_repair_suggestions(
    state: CompositionState,
    validation: ValidationSummary,
) -> list[_GraphRepairSuggestion]:
    """Return structured graph repair suggestions for validation failures."""
    return _duplicate_consumer_repair_suggestions(state, validation)


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Result of a tool execution.

    Attributes:
        success: Whether the operation succeeded.
        updated_state: Full state after mutation (or original if success=False).
        validation: Stage 1 validation result for the updated state.
        affected_nodes: Node IDs changed or with changed edges.
        data: Optional data payload for discovery tools.
        prior_validation: Validation from before the mutation. When set,
            to_dict() includes a ``validation_delta`` showing new and
            resolved entries so the agent can focus on what changed.
        post_call_hints: Forward-looking coaching hints from the plugin
            that was just configured. Resolved by the catalog from
            ``BaseX.get_post_call_hints`` (see
            ``contracts/plugin_assistance.py``). Advisory only — not
            part of any audit hash. ``to_dict`` emits this field
            *only when non-empty* so existing tool consumers see no
            schema change.
    """

    success: bool
    updated_state: CompositionState
    validation: ValidationSummary
    affected_nodes: tuple[str, ...]
    data: Any = None
    prior_validation: ValidationSummary | None = None
    runtime_preflight: ValidationResult | None = None
    post_call_hints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        freeze_fields(self, "affected_nodes", "post_call_hints")
        if self.data is not None:
            freeze_fields(self, "data")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict suitable for LLM tool response.

        Validation entries are serialized as structured dicts with
        component, message, and severity fields (B2 requirement).

        When prior_validation is set, includes a validation_delta with
        new_errors, resolved_errors, new_warnings, resolved_warnings to
        help the agent focus on what changed rather than re-reading the
        full validation state.
        """

        result: dict[str, Any] = {
            "success": self.success,
            "validation": {
                "is_valid": self.validation.is_valid,
                "errors": [e.to_dict() for e in self.validation.errors],
                "warnings": [e.to_dict() for e in self.validation.warnings],
                "suggestions": [e.to_dict() for e in self.validation.suggestions],
                "semantic_contracts": _semantic_contracts_payload(
                    self.validation.semantic_contracts,
                ),
                "graph_repair_suggestions": _graph_repair_suggestions(
                    self.updated_state,
                    self.validation,
                ),
            },
            "affected_nodes": list(self.affected_nodes),
            "version": self.updated_state.version,
        }
        if self.data is not None:
            result["data"] = deep_thaw(self.data)

        if self.runtime_preflight is not None:
            result["runtime_preflight"] = self.runtime_preflight.model_dump()

        if self.prior_validation is not None:
            result["validation_delta"] = _compute_validation_delta(
                self.prior_validation,
                self.validation,
            )

        if self.post_call_hints:
            result["post_call_hints"] = list(self.post_call_hints)

        return result


def diff_states(
    baseline: CompositionState,
    current: CompositionState,
    *,
    baseline_validation: ValidationSummary | None = None,
    current_validation: ValidationSummary | None = None,
) -> dict[str, Any]:
    """Compare two composition states and return a structured change summary.

    Reports added, removed, and modified nodes/edges/outputs, plus source
    and metadata changes. Used by the diff_pipeline MCP tool (B5).

    Args:
        baseline_validation: Pre-computed validation for the baseline state.
        current_validation: Pre-computed validation for the current state.
    """
    changes: dict[str, Any] = {
        "from_version": baseline.version,
        "to_version": current.version,
        "source_changed": False,
        "metadata_changed": False,
        "nodes": {"added": [], "removed": [], "modified": []},
        "edges": {"added": [], "removed": [], "modified": []},
        "outputs": {"added": [], "removed": [], "modified": []},
    }

    # Source
    if baseline.source != current.source:
        changes["source_changed"] = True
        if baseline.source is None:
            changes["source_detail"] = "added"
        elif current.source is None:
            changes["source_detail"] = "removed"
        else:
            changes["source_detail"] = "modified"

    # Metadata
    if baseline.metadata != current.metadata:
        changes["metadata_changed"] = True

    # Nodes
    baseline_nodes = {n.id: n for n in baseline.nodes}
    current_nodes = {n.id: n for n in current.nodes}
    for nid in current_nodes:
        if nid not in baseline_nodes:
            changes["nodes"]["added"].append(nid)
        elif current_nodes[nid] != baseline_nodes[nid]:
            changes["nodes"]["modified"].append(nid)
    for nid in baseline_nodes:
        if nid not in current_nodes:
            changes["nodes"]["removed"].append(nid)

    # Edges
    baseline_edges = {e.id: e for e in baseline.edges}
    current_edges = {e.id: e for e in current.edges}
    for eid in current_edges:
        if eid not in baseline_edges:
            changes["edges"]["added"].append(eid)
        elif current_edges[eid] != baseline_edges[eid]:
            changes["edges"]["modified"].append(eid)
    for eid in baseline_edges:
        if eid not in current_edges:
            changes["edges"]["removed"].append(eid)

    # Outputs
    baseline_outputs = {o.name: o for o in baseline.outputs}
    current_outputs = {o.name: o for o in current.outputs}
    for name in current_outputs:
        if name not in baseline_outputs:
            changes["outputs"]["added"].append(name)
        elif current_outputs[name] != baseline_outputs[name]:
            changes["outputs"]["modified"].append(name)
    for name in baseline_outputs:
        if name not in current_outputs:
            changes["outputs"]["removed"].append(name)

    # Validation delta — reuse pre-computed validations when available
    if baseline_validation is None:
        baseline_validation = baseline.validate()
    if current_validation is None:
        current_validation = current.validate()
    baseline_warnings = {e.message for e in baseline_validation.warnings}
    current_warnings = {e.message for e in current_validation.warnings}
    changes["warnings_introduced"] = sorted(current_warnings - baseline_warnings)
    changes["warnings_resolved"] = sorted(baseline_warnings - current_warnings)

    # Summary stats
    total = sum(len(changes[k][action]) for k in ("nodes", "edges", "outputs") for action in ("added", "removed", "modified"))
    total += int(changes["source_changed"]) + int(changes["metadata_changed"])
    changes["total_changes"] = total

    return changes


# --- Expression Grammar (static) ---

_EXPRESSION_GRAMMAR = """\
Gate Expression Syntax Reference
=================================

Variables:
  row      - The current row as a dict. Access fields via row['field_name'].

Field access:
  row['field_name']       Direct access (raises KeyError if missing)
  row.get('field_name')   Returns None if missing (NO default argument allowed)

Operators:
  ==, !=, <, >, <=, >=   Comparison
  and, or, not            Boolean logic
  in, not in              Membership test
  +, -, *, /, //, %       Arithmetic

Built-in functions (only these are allowed):
  len()    Length of a sequence or string
  abs()    Absolute value of a number

Type coercion functions (int, str, float, bool) are NOT available.
Types are guaranteed by the source schema — no coercion is needed in expressions.

Examples:
  row['confidence'] >= 0.85
  row['status'] == 'approved'
  row['category'] in ('A', 'B', 'C')
  row.get('optional_field') is not None
  row['score'] > 0.5 and row['status'] != 'rejected'
  len(row['name']) > 0

Forbidden:
  row.get('field', default)   Default values fabricate data — use 'is not None' test
  int(row['x'])               Type coercion — coerce at source schema instead
  Imports, lambdas, comprehensions, attribute access (except row.get)
"""


def get_expression_grammar() -> str:
    """Return the gate expression syntax reference."""
    return _EXPRESSION_GRAMMAR


# --- Tool Definitions for LLM ---


ADVISOR_TRIGGER_REACTIVE: Final[str] = "reactive_validation_loop"
ADVISOR_TRIGGER_PROACTIVE_SECURITY: Final[str] = "proactive_security_safety"
ADVISOR_TRIGGER_PROACTIVE_RED_LISTED: Final[str] = "proactive_red_listed_plugin"
ADVISOR_TRIGGER_VALUES: Final[tuple[str, ...]] = (
    ADVISOR_TRIGGER_REACTIVE,
    ADVISOR_TRIGGER_PROACTIVE_SECURITY,
    ADVISOR_TRIGGER_PROACTIVE_RED_LISTED,
)


def get_tool_definitions() -> list[dict[str, Any]]:
    """Return JSON Schema tool definitions for the LLM.

    Returns 39 tools: 13 discovery + 13 mutation + 9 blob tools + 3 secret
    tools + 1 advisor tool. ``request_advisor_hint`` is the only tool that
    is filtered out of the LLM-visible list when the operator's
    ``composer_advisor_enabled`` flag is False (the default) — see
    ``ComposerServiceImpl._get_litellm_tools``.

    The skill at ``src/elspeth/web/composer/skills/pipeline_composer.md``
    enumerates the same tool set in its Step-0 section. The drift gate
    ``TestComposerToolNameDrift::test_skill_step0_matches_get_tool_definitions``
    in ``tests/unit/web/composer/test_skill_drift.py`` enforces equality
    between the runtime list returned here and the skill's bulleted
    categories — adding a tool without updating both sides fails CI.
    """
    return [
        # Discovery tools
        {
            "name": "list_sources",
            "description": "List available source plugins with name and summary.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "list_transforms",
            "description": "List available transform plugins with name and summary.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "list_sinks",
            "description": "List available sink plugins with name and summary.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "get_plugin_schema",
            "description": "Get the full configuration schema for a plugin.",
            "parameters": {
                "type": "object",
                "properties": {
                    "plugin_type": {
                        "type": "string",
                        "enum": ["source", "transform", "sink"],
                        "description": "Plugin type.",
                    },
                    "name": {
                        "type": "string",
                        "description": "Plugin name (e.g. 'csv').",
                    },
                },
                "required": ["plugin_type", "name"],
            },
        },
        {
            "name": "get_expression_grammar",
            "description": "Get the gate expression syntax reference.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        # Mutation tools
        {
            "name": "set_source",
            "description": "Set or replace a named pipeline source.",
            "parameters": {
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
        },
        {
            "name": "upsert_node",
            "description": (
                "Add or update a pipeline node. "
                "Fields are node_type-dependent: "
                "transform/aggregation use plugin+options; "
                "gate uses condition+routes (or fork_to); "
                "coalesce uses branches+policy+merge. "
                "Omit fields that don't apply to your node_type."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Unique node identifier."},
                    "node_type": {
                        "type": "string",
                        "enum": ["transform", "gate", "aggregation", "coalesce"],
                    },
                    "plugin": {
                        "type": ["string", "null"],
                        "description": "Plugin name. Required for transform/aggregation. Null for gate/coalesce.",
                    },
                    "input": {
                        "type": "string",
                        "description": (
                            "Connection-name string this node CONSUMES. MUST equal the value of some "
                            "upstream's on_success (or routes value, or on_error) field. NOT the upstream "
                            "node's id — connections are matched by string, not by graph topology. "
                            "Example: if source.on_success='raw_url_rows', this node sets input='raw_url_rows'."
                        ),
                        "examples": ["raw_url_rows", "fetched_text", "scored_rows"],
                    },
                    "on_success": {
                        "type": ["string", "null"],
                        "description": (
                            "Output connection. Required for transform/aggregation/coalesce. Null for "
                            "gates (routing is via condition/routes). When set, this is the connection-name "
                            "string the node PUBLISHES — some downstream input/sink_name MUST equal this "
                            "value. The runtime matches strings, not topology."
                        ),
                        "examples": ["fetched_text", "scored_rows", "lines_out"],
                    },
                    "on_error": {"type": ["string", "null"], "description": "Error output connection (transform/aggregation only)."},
                    "options": {"type": "object", "description": "Plugin-specific config (transform/aggregation only)."},
                    "condition": {"type": ["string", "null"], "description": "Boolean expression (gate only). Evaluated per row."},
                    "routes": {
                        "type": ["object", "null"],
                        "description": (
                            "Route mapping {true: sink_or_connection_or_discard, false: sink_or_connection_or_discard} "
                            "(gate only, mutually exclusive with fork_to). Use 'discard' to drop that route with "
                            "an audited gate_discarded terminal outcome."
                        ),
                    },
                    "fork_to": {
                        "type": ["array", "null"],
                        "items": {"type": "string"},
                        "description": "Fork destinations — row is copied to all listed paths (gate only, mutually exclusive with routes).",
                    },
                    "branches": {
                        "type": ["array", "object", "null"],
                        "items": {"type": "string"},
                        "additionalProperties": {"type": "string"},
                        "description": (
                            "Branches to merge (coalesce only). Use list form when branch identity and input "
                            "connection are the same, or object form {branch_name: input_connection} when a "
                            "branch flows through transforms before coalescing."
                        ),
                    },
                    "policy": {"type": ["string", "null"], "description": "Merge trigger policy (coalesce only)."},
                    "merge": {"type": ["string", "null"], "description": "Field merge strategy (coalesce only)."},
                    "trigger": {
                        "type": ["object", "null"],
                        "description": "Optional early batch trigger config (aggregation only). Omit, null, or {} for end-of-source-only aggregation.",
                        "additionalProperties": False,
                        "properties": {
                            "count": {
                                "type": ["integer", "null"],
                                "minimum": 1,
                                "description": "Flush after this many accepted rows.",
                            },
                            "timeout_seconds": {
                                "type": ["number", "null"],
                                "exclusiveMinimum": 0,
                                "description": "Flush after this many seconds since the first accepted row.",
                            },
                            "condition": {
                                "type": ["string", "null"],
                                "description": "Boolean expression over row['batch_count'] and row['batch_age_seconds']; do not use end_of_source here.",
                            },
                        },
                    },
                    "output_mode": {
                        "type": ["string", "null"],
                        "enum": ["passthrough", "transform", None],
                        "description": "Aggregation output mode (aggregation only). Defaults to 'transform' if omitted.",
                    },
                    "expected_output_count": {
                        "type": ["integer", "null"],
                        "description": "Expected number of output rows from aggregation (aggregation only). Optional; omit when output count depends on group_by distinct values.",
                    },
                },
                "required": ["id", "node_type", "input"],
            },
        },
        {
            "name": "upsert_edge",
            "description": (
                "Add or update a connection between nodes. When the edge targets a sink, "
                "this also updates the source/node routing field used by runtime "
                "(on_success, on_error, gate routes, or fork destinations)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Unique edge identifier."},
                    "from_node": {"type": "string", "description": "Source node ID or 'source'."},
                    "to_node": {"type": "string", "description": "Destination node ID or sink name."},
                    "edge_type": {
                        "type": "string",
                        "enum": ["on_success", "on_error", "route_true", "route_false", "fork"],
                    },
                    "label": {"type": ["string", "null"], "description": "Display label."},
                },
                "required": ["id", "from_node", "to_node", "edge_type"],
                "examples": [
                    {
                        "id": "e_judge_layers_error",
                        "from_node": "judge_layers",
                        "to_node": "llm_failures",
                        "edge_type": "on_error",
                        "label": "LLM failures",
                    }
                ],
            },
        },
        {
            "name": "remove_node",
            "description": "Remove a node and all its edges.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Node ID to remove."},
                },
                "required": ["id"],
            },
        },
        {
            "name": "remove_edge",
            "description": "Remove an edge by ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Edge ID to remove."},
                },
                "required": ["id"],
            },
        },
        {
            "name": "set_metadata",
            "description": "Update pipeline metadata (name and description only).",
            "parameters": {
                "type": "object",
                "properties": {
                    "patch": {
                        "type": "object",
                        "description": "Partial metadata update. Only included fields are changed.",
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                        },
                    },
                },
                "required": ["patch"],
            },
        },
        {
            "name": "set_output",
            "description": "Add or replace a pipeline output (sink).",
            "parameters": {
                "type": "object",
                "properties": {
                    "sink_name": {
                        "type": "string",
                        "description": (
                            "Sink name. This string is BOTH the sink's identifier (used by "
                            "patch_output_options/remove_output) AND the connection-name the sink "
                            "consumes — it MUST equal some upstream's on_success value. Pick a name "
                            "describing the data being written; it does not need to match an upstream "
                            "node's id."
                        ),
                        "examples": ["lines_out", "scored_results", "errors_quarantine"],
                    },
                    "plugin": {"type": "string", "description": "Sink plugin name (e.g. 'csv', 'json')."},
                    "options": {
                        "type": "object",
                        "description": (
                            "Plugin-specific config. For csv/json file sinks in runnable web pipelines, "
                            "include path, schema, and explicit collision_policy."
                        ),
                    },
                    "on_write_failure": {
                        "type": "string",
                        "description": "How to handle per-row write failures. Use 'discard' to drop with audit record, or a sink name (e.g. 'results_failures') to divert failed rows to that failsink.",
                        "default": "discard",
                    },
                },
                "required": ["sink_name", "plugin", "options"],
            },
        },
        {
            "name": "remove_output",
            "description": "Remove a pipeline output (sink) by name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sink_name": {"type": "string", "description": "Sink name to remove."},
                },
                "required": ["sink_name"],
            },
        },
        {
            "name": "patch_source_options",
            "description": "Apply a shallow merge-patch to a named source's options. "
            "Keys in the patch overwrite existing keys. "
            "Keys set to null are deleted. Missing keys are unchanged.",
            "parameters": {
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
        },
        {
            "name": "patch_node_options",
            "description": "Apply a shallow merge-patch to a node's options. "
            "Keys in the patch overwrite existing keys. "
            "Keys set to null are deleted. Missing keys are unchanged. "
            "Do not use this for node routing fields such as on_success/on_error/input/routes; "
            "use upsert_edge or upsert_node for routing edits.",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {
                        "type": "string",
                        "description": "ID of the node to patch.",
                    },
                    "patch": {
                        "type": "object",
                        "description": (
                            "Merge-patch to apply to plugin options only. "
                            "Node-level routing fields such as on_success, on_error, input, routes, "
                            "and fork_to are siblings of options; edit them with upsert_edge or upsert_node."
                        ),
                    },
                },
                "required": ["node_id", "patch"],
            },
        },
        {
            "name": "patch_output_options",
            "description": "Apply a shallow merge-patch to an output's options. "
            "Keys in the patch overwrite existing keys. "
            "Keys set to null are deleted. Missing keys are unchanged.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sink_name": {
                        "type": "string",
                        "description": "Name of the output (sink) to patch.",
                    },
                    "patch": {
                        "type": "object",
                        "description": "Merge-patch to apply to output options.",
                    },
                },
                "required": ["sink_name", "patch"],
            },
        },
        {
            "name": "set_pipeline",
            "description": "Atomically replace the entire pipeline. Provide the "
            "complete source, nodes, edges, outputs, and metadata in one call. "
            "This is more efficient than calling set_source + upsert_node + "
            "upsert_edge + set_output sequentially.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "object",
                        "description": (
                            "Source configuration: {plugin, on_success, options?, on_validation_failure?, blob_id?, inline_blob?}. "
                            "Use blob_id to bind an already uploaded session blob, or inline_blob to "
                            "materialize user-provided literal data while atomically setting the full pipeline."
                        ),
                        "properties": {
                            "plugin": {"type": "string"},
                            "blob_id": {
                                "type": "string",
                                "description": (
                                    "Existing ready session blob ID to bind as this source. "
                                    "The tool resolves path/blob_ref authoritatively exactly like set_source_from_blob."
                                ),
                            },
                            "options": {
                                "type": "object",
                                "description": (
                                    "Plugin-specific source config. Required by most file/data sources even though "
                                    "the schema leaves it optional so the handler can return plugin-specific repair "
                                    "feedback instead of a generic missing-argument error."
                                ),
                            },
                            "on_success": {
                                "type": "string",
                                "description": (
                                    "Connection-name string the source PUBLISHES. Some downstream "
                                    "consumer (node 'input' or output 'sink_name') MUST equal this. "
                                    "Connections match by string, not by node id."
                                ),
                                "examples": ["raw_url_rows", "csv_rows", "fetched_text"],
                            },
                            "on_validation_failure": {
                                "type": "string",
                                "description": _SOURCE_VALIDATION_FAILURE_DESCRIPTION,
                            },
                            "inline_blob": {
                                "type": "object",
                                "description": (
                                    "Optional inline source content to create as a session blob before binding the source. "
                                    "Fields mirror create_blob: filename, mime_type, content, and optional description."
                                ),
                                "properties": {
                                    "filename": {"type": "string"},
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
                                    },
                                    "content": {"type": "string"},
                                    "description": {"type": "string"},
                                },
                                "required": ["filename", "mime_type", "content"],
                            },
                        },
                        "required": ["plugin", "on_success"],
                    },
                    "sources": {
                        "type": "object",
                        "description": (
                            "Named source roots keyed by stable source name. Use this instead of source for multi-source pipelines. "
                            "Each value has the same shape as source, but blob_id and inline_blob are only supported on the legacy source field in v1."
                        ),
                        "additionalProperties": {
                            "type": "object",
                            "properties": {
                                "plugin": {"type": "string"},
                                "options": {"type": "object"},
                                "on_success": {"type": "string"},
                                "on_validation_failure": {
                                    "type": "string",
                                    "description": _SOURCE_VALIDATION_FAILURE_DESCRIPTION,
                                },
                            },
                            "required": ["plugin", "on_success"],
                        },
                    },
                    "nodes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "node_type": {"type": "string"},
                                "plugin": {"type": "string"},
                                "input": {
                                    "type": "string",
                                    "description": (
                                        "Connection-name string this node CONSUMES. MUST equal some "
                                        "upstream's on_success/routes value/on_error. NOT the upstream "
                                        "node's id. If source.on_success='raw_url_rows', this node sets "
                                        "input='raw_url_rows'."
                                    ),
                                    "examples": ["raw_url_rows", "fetched_text", "scored_rows"],
                                },
                                "on_success": {
                                    "type": "string",
                                    "description": (
                                        "Connection-name string this node PUBLISHES (transform/aggregation/"
                                        "coalesce). Some downstream input/sink_name MUST equal this. Omit "
                                        "for gates (routing is via condition+routes)."
                                    ),
                                    "examples": ["fetched_text", "scored_rows", "lines_out"],
                                },
                                "on_error": {"type": "string"},
                                "options": {"type": "object"},
                                "condition": {"type": "string"},
                                "routes": {
                                    "type": "object",
                                    "description": (
                                        "Gate route mapping to sink names, downstream connection names, 'fork', or "
                                        "'discard' for an audited terminal drop."
                                    ),
                                },
                                "fork_to": {"type": "array", "items": {"type": "string"}},
                                "branches": {
                                    "type": ["array", "object"],
                                    "items": {"type": "string"},
                                    "additionalProperties": {"type": "string"},
                                },
                                "policy": {"type": "string"},
                                "merge": {"type": "string"},
                                "trigger": {"type": "object"},
                                "output_mode": {"type": "string"},
                                "expected_output_count": {"type": "integer"},
                            },
                            "required": ["id", "node_type", "input"],
                        },
                        "description": "Array of node specs: [{id, input, plugin?, node_type, options?, on_success?, on_error?, condition?, routes?, fork_to?, branches?, policy?, merge?, trigger?, output_mode?, expected_output_count?}]",
                    },
                    "edges": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "from_node": {"type": "string"},
                                "to_node": {"type": "string"},
                                "edge_type": {"type": "string"},
                                "label": {"type": "string"},
                            },
                            "required": ["id", "from_node", "to_node", "edge_type"],
                        },
                        "description": "Array of edge specs: [{id, from_node, to_node, edge_type}]",
                    },
                    "outputs": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "sink_name": {
                                    "type": "string",
                                    "description": (
                                        "Sink name. BOTH the sink's identifier AND the connection-name "
                                        "the sink consumes — it MUST equal some upstream's on_success "
                                        "value. Pick a descriptive name; it does not need to match an "
                                        "upstream node's id."
                                    ),
                                    "examples": ["lines_out", "scored_results", "errors_quarantine"],
                                },
                                "plugin": {"type": "string"},
                                "options": {
                                    "type": "object",
                                    "description": (
                                        "Plugin-specific sink config. For csv/json file sinks in runnable web "
                                        "pipelines, include path, schema, and explicit collision_policy."
                                    ),
                                },
                                "on_write_failure": {"type": "string"},
                            },
                            "required": ["sink_name", "plugin"],
                            "examples": [
                                {
                                    "sink_name": "results",
                                    "plugin": "json",
                                    "options": {
                                        "path": "outputs/results.json",
                                        "schema": {"mode": "observed"},
                                        "collision_policy": "auto_increment",
                                    },
                                    "on_write_failure": "discard",
                                }
                            ],
                        },
                        "description": (
                            "Array of output specs: [{sink_name, plugin, options, on_write_failure?}]. "
                            "For csv/json file sinks in runnable web pipelines, options must include "
                            "path, schema, and explicit collision_policy."
                        ),
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Pipeline metadata: {name?, description?}",
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                        },
                    },
                },
                "required": ["nodes", "edges", "outputs"],
            },
        },
        # Source-reset and validation-explanation tools.
        {
            "name": "clear_source",
            "description": "Remove one named source from the pipeline composition state, or all sources when omitted.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source_name": {
                        "type": "string",
                        "description": "Optional source root name to remove. Omit to clear all sources.",
                    },
                },
                "required": [],
            },
        },
        {
            "name": "explain_validation_error",
            "description": "Get a human-readable explanation of a validation error "
            "with suggested fixes. Pass the exact error text from a validation result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "error_text": {
                        "type": "string",
                        "description": "The validation error message to explain.",
                    },
                },
                "required": ["error_text"],
            },
        },
        {
            "name": "request_advisor_hint",
            "description": (
                "ESCAPE HATCH — call when one of the declared trigger criteria applies: "
                "reactive validation-loop recovery after two or more unchanged failures, "
                "proactive security/safety wiring review before `set_pipeline`, or "
                "proactive red-listed plugin review before `set_pipeline`. The proactive "
                "security trigger covers content moderation, prompt-injection defence, "
                "secret routing, PII/regulatory sinks, and externally fetched content "
                "flowing toward LLMs. Forwards your problem statement and context to a "
                "frontier model and returns guidance text. The reply is ADVICE, not "
                "configuration — you must still call the appropriate mutation tool "
                "yourself to apply any change. Budget is finite (sized per compose "
                "request, not per session lifetime) and exhausting it returns a "
                "structured error rather than crashing — inspect budget_remaining "
                "in each response. Do NOT call this tool in a loop, do NOT use it "
                "as a substitute for reading validator output. Disabled by default; "
                "only available when the operator has explicitly enabled it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "trigger": {
                        "type": "string",
                        "enum": list(ADVISOR_TRIGGER_VALUES),
                        "description": (
                            "Why this advisor call is allowed. Use reactive_validation_loop "
                            "only after the recovery sequence and at least two unchanged "
                            "validator failures. Use proactive_security_safety before "
                            "set_pipeline for security/safety-sensitive flows. Use "
                            "proactive_red_listed_plugin before set_pipeline when the plan "
                            "uses a red-listed plugin such as llm, database, dataverse, "
                            "Azure safety transforms, RAG retrieval, or Chroma sinks."
                        ),
                    },
                    "problem_summary": {
                        "type": "string",
                        "description": (
                            "Your own statement of what you are trying to do and "
                            "why you are stuck. One or two sentences. Be specific: "
                            "'I cannot get llm transform options to validate against "
                            "the Azure provider schema' is useful; 'help' is not."
                        ),
                    },
                    "recent_errors": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "The last validator error messages verbatim, most recent first. Include up to 5; do not paraphrase."
                        ),
                    },
                    "attempted_actions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "What you have already tried, one item per attempt. "
                            "Include the tool name and a one-line summary of the "
                            "argument shape. The advisor uses this to avoid "
                            "suggesting things you have already ruled out."
                        ),
                    },
                    "schema_excerpt": {
                        "type": "string",
                        "description": (
                            "Optional — the relevant plugin schema snippet you are "
                            "working against, as returned by `get_plugin_schema`. "
                            "Including this lets the advisor give field-level "
                            "guidance grounded in the exact contract."
                        ),
                    },
                },
                "required": ["trigger", "problem_summary", "recent_errors", "attempted_actions"],
            },
        },
        {
            "name": "get_plugin_assistance",
            "description": (
                "Retrieve plugin-owned guidance for a source, transform, or sink. "
                "Two modes by ``issue_code``:\n"
                "  * Omit ``issue_code`` (or pass null) to get discovery-time guidance "
                "    — a summary of the plugin and composer_hints. (The same hints "
                "    are also carried on list_sources / list_transforms / list_sinks / "
                "    get_plugin_schema responses; this tool is the explicit path.)\n"
                "  * Pass an ``issue_code`` (validators emit these as requirement_code "
                "    on semantic_contracts entries) to get failure-time guidance — "
                "    summary, suggested_fixes, and example before/after configurations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "plugin_type": {
                        "type": "string",
                        "enum": ["source", "transform", "sink"],
                        "description": "Plugin family. 'source', 'transform', or 'sink'.",
                    },
                    "plugin_name": {
                        "type": "string",
                        "description": "Plugin name (e.g. 'csv', 'web_scrape', 'database').",
                    },
                    "issue_code": {
                        "type": ["string", "null"],
                        "description": (
                            "Optional. Stable issue identifier owned by the plugin "
                            "for failure-time guidance. Omit or pass null for "
                            "discovery-time guidance."
                        ),
                    },
                },
                "required": ["plugin_type", "plugin_name"],
            },
        },
        {
            "name": "list_models",
            "description": "List available LLM model identifiers. Without a provider "
            "filter, returns provider names and counts. With a provider filter, "
            "returns matching model IDs (capped at limit). For provider='openrouter/' "
            "the returned slugs are normalised to OpenRouter's HTTP API form "
            "(without the litellm-internal 'openrouter/' routing prefix) — these "
            "are the values to put directly in `model:`.",
            "parameters": {
                "type": "object",
                "properties": {
                    "provider": {
                        "type": "string",
                        "description": "Provider prefix to filter by (e.g. 'openrouter/', 'azure/'). "
                        "Omit to get a provider summary instead of individual models.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max models to return (default 50).",
                    },
                },
                "required": [],
            },
        },
        {
            "name": "get_audit_info",
            "description": (
                "Return facts about ELSPETH's Landscape audit trail. Call this BEFORE "
                "answering any user question that mentions audit logging, audit "
                "database, SQLite/Postgres audit, audit backend, audit export, "
                "Landscape, or 'how do I record what the pipeline did'. Audit is "
                "mandatory and operator-managed; the composer cannot configure the "
                "backend (security boundary — see yaml_generator.py:179, fix S1). "
                "Returns enabled status, composer_modifiable flag, and a canonical "
                "summary to paraphrase. Does NOT return the audit URL/path/DSN — "
                "that is operator-internal and intentionally not surfaced to the LLM."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "preview_pipeline",
            "description": "Preview the current pipeline configuration — returns "
            "validation status, source summary, and node/output overview "
            "without executing. Use this to confirm the pipeline is set up "
            "correctly before running.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "get_pipeline_state",
            "description": "Inspect the full current pipeline state including all "
            "options for source, nodes, and outputs. Use this during correction "
            "loops to see what is currently configured before patching.",
            "parameters": {
                "type": "object",
                "properties": {
                    "component": {
                        "type": "string",
                        "description": (
                            "Optional: return only one component — 'source', a node ID, or an output name. "
                            "Accepted full-state aliases: omit component, pass 'full', 'all', 'pipeline', "
                            "or pass the empty string."
                        ),
                    },
                },
                "required": [],
            },
        },
        {
            "name": "diff_pipeline",
            "description": "Show what changed since the session was loaded or created. "
            "Returns added, removed, and modified nodes/edges/outputs, "
            "plus warnings introduced or resolved.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        # Blob tools
        {
            "name": "list_blobs",
            "description": "List uploaded/created files (blobs) in this session with metadata.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "get_blob_metadata",
            "description": "Get metadata for a specific blob (file) by ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "blob_id": {"type": "string", "description": "Blob ID."},
                },
                "required": ["blob_id"],
            },
        },
        {
            "name": "set_source_from_blob",
            "description": "Wire a blob as the pipeline source. Resolves the blob's storage path internally and infers the source plugin from its MIME type. "
            "Use 'options' for plugin-specific config (e.g., 'column' and 'schema' for text sources).",
            "parameters": {
                "type": "object",
                "properties": {
                    "blob_id": {"type": "string", "description": "Blob ID to use as source."},
                    "plugin": {"type": "string", "description": "Source plugin override (e.g. 'csv'). Inferred from MIME type if omitted."},
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
                        "description": "Plugin-specific config (merged with blob path). Required fields vary by plugin: "
                        "text sources need 'column' (output field name) and 'schema' (e.g., {mode: 'observed'}).",
                    },
                },
                "required": ["blob_id", "on_success"],
            },
        },
        {
            "name": "create_blob",
            "description": "Create a new file (blob) from inline content. "
            "Use this to create seed input files (URLs, JSON, CSV snippets) "
            "mid-conversation without requiring manual upload.",
            "parameters": {
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
            },
        },
        {
            "name": "update_blob",
            "description": "Update the content of an existing blob (file). Overwrites the file content while preserving metadata.",
            "parameters": {
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
            },
        },
        {
            "name": "delete_blob",
            "description": "Delete a blob (file) and its storage.",
            "parameters": {
                "type": "object",
                "properties": {
                    "blob_id": {
                        "type": "string",
                        "description": "ID of the blob to delete.",
                    },
                },
                "required": ["blob_id"],
            },
        },
        {
            "name": "get_blob_content",
            "description": "Retrieve the content of a blob (file) for inspection. Large files are truncated to 50,000 characters.",
            "parameters": {
                "type": "object",
                "properties": {
                    "blob_id": {
                        "type": "string",
                        "description": "ID of the blob to read.",
                    },
                },
                "required": ["blob_id"],
            },
        },
        {
            "name": "list_recipes",
            "description": (
                "List the registered pipeline recipes — deterministic scaffolds for common simple "
                "intents. Each recipe declares its required slots; apply_pipeline_recipe then "
                "instantiates the recipe with operator-supplied slot values. Recipes accelerate "
                "the highest-frequency 'classify CSV with LLM' and 'split rows by threshold' "
                "patterns; for shapes outside the recipe set, hand-author with set_pipeline."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "apply_pipeline_recipe",
            "description": (
                "Apply a registered pipeline recipe with operator-supplied slot values and replace "
                "the current pipeline state with the resulting configuration. Slots are validated "
                "against the recipe's declared schema before scaffolding — invalid slots are "
                "rejected with a repair hint. Call list_recipes to discover available recipes and "
                "their slot schemas. The resulting state is identical to a hand-authored "
                "set_pipeline call; the model can refine via patch_*_options afterwards."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "recipe_name": {
                        "type": "string",
                        "description": "Recipe identifier (e.g., 'classify-rows-llm-jsonl')",
                    },
                    "slots": {
                        "type": "object",
                        "description": "Operator-supplied slot values; must match the recipe's slot schema",
                    },
                },
                "required": ["recipe_name", "slots"],
            },
        },
        {
            "name": "inspect_source",
            "description": (
                "Return bounded structural facts about a blob-backed source: source kind, observed "
                "headers, sample row count, inferred scalar types per column, URL candidates, and "
                "warnings. Reads at most 8 KiB of the blob and parses at most 100 rows. Use this "
                "before declaring a fixed CSV/JSON schema — observed headers and inferred types "
                "tell you which fields the source actually contains and what numeric coercion is "
                "needed before any gate or value_transform numeric op. Never returns raw row "
                "content; only summary facts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "blob_id": {
                        "type": "string",
                        "description": "ID of the blob to inspect.",
                    },
                },
                "required": ["blob_id"],
            },
        },
        # Secret tools
        {
            "name": "list_secret_refs",
            "description": "List available secret references (API keys, credentials). Shows names and scopes, never values.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "validate_secret_ref",
            "description": "Check if a secret reference exists and is accessible to the current user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Secret reference name (e.g. 'OPENROUTER_API_KEY')."},
                },
                "required": ["name"],
            },
        },
        # Composer-LLM-callable tool surface for surfacing an interpretation
        # of a subjective or under-specified term for user review.
        # The description below is normative documentation for the LLM (mirrored
        # in the composer skill markdown) and is reviewed by the audit panel as
        # part of the request_interpretation_review event row's provenance.
        #
        # Position note: this tool is inserted BEFORE ``wire_secret_ref`` so
        # the trailing tool name remains ``wire_secret_ref`` — the Anthropic
        # cache-marker test (``test_trailing_tool_name_is_locked``) pins the
        # trailing position to preserve prompt-cache stability across deploys.
        {
            "name": "request_interpretation_review",
            "description": (
                "Ask the user to review your interpretation of a subjective or "
                "underspecified term they used. Call this BEFORE you finalise "
                "the prompt template for any LLM transform whose prompt depends "
                "on the term. Surface ONE term per call. The composition state "
                "MUST already contain the affected LLM transform (call upsert_node "
                "first) and its prompt_template MUST contain the placeholder "
                "{{interpretation:<term>}}. The user will see your draft and "
                "either accept it or amend it. Do not ask the user in assistant "
                "prose; this tool is the review surface. If no composition state "
                "exists yet, stage the LLM transform with a placeholder first, "
                "wait for that tool result, then call this tool. Do not call this "
                "for concrete operators (e.g., 'rate 1-10') or for terms the "
                "user already defined in the conversation."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "required": ["affected_node_id", "user_term", "llm_draft"],
                "properties": {
                    "affected_node_id": {
                        "type": "string",
                        "description": "node_id of the LLM transform whose prompt template depends on this term",
                    },
                    "user_term": {
                        "type": "string",
                        "description": "The user-provided term, verbatim (e.g., 'cool', 'important', 'risky')",
                    },
                    "llm_draft": {
                        "type": "string",
                        "description": "Your draft interpretation of the term, in your own words, suitable to embed as a phrase in the prompt template",
                    },
                },
            },
        },
        {
            "name": "wire_secret_ref",
            "description": "Place a secret reference marker in the pipeline config. The secret will be resolved at execution time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Secret reference name."},
                    "target": {
                        "type": "string",
                        "enum": ["source", "node", "output"],
                        "description": "Which component to wire the secret into.",
                    },
                    "target_id": {"type": "string", "description": "Node ID or output name (required for node/output targets)."},
                    "option_key": {"type": "string", "description": "Config option key to set (e.g. 'api_key')."},
                },
                "required": ["name", "target", "option_key"],
            },
        },
    ]


# --- Tool Registry ---

# Dual-registry invariant (F-18):
# ELSPETH dispatches composer tools through TWO disjoint registries.
#
#   ``_DISCOVERY_TOOLS`` / ``_MUTATION_TOOLS`` (and the blob/secret peer
#   registries) hold state-pure SYNCHRONOUS handlers. They never touch
#   the session DB write surface, never await; ``execute_tool`` calls
#   them inline inside a worker thread.
#
#   ``_SESSION_AWARE_TOOL_HANDLERS`` holds ASYNC handlers that need a
#   session_id, tool_call_id, and service-method callable. They are NOT
#   reached through ``execute_tool``; the compose loop intercepts them
#   ahead of ``run_sync_in_worker(execute_tool, ...)`` and awaits them
#   directly (precedent: ``request_advisor_hint`` interception at
#   service.py around line 2820).
#
# **Invariant (F-18):**
#   * Every tool name in ``get_tool_definitions()`` appears in EXACTLY
#     one registry across all of: ``_DISCOVERY_TOOLS``, ``_MUTATION_TOOLS``,
#     ``_BLOB_DISCOVERY_TOOLS``, ``_BLOB_MUTATION_TOOLS``,
#     ``_SECRET_DISCOVERY_TOOLS``, ``_SECRET_MUTATION_TOOLS``,
#     ``_SESSION_AWARE_TOOL_HANDLERS``.
#   * Every handler in ``_SESSION_AWARE_TOOL_HANDLERS`` satisfies
#     ``asyncio.iscoroutinefunction(h) is True``.
#   * Every handler in the sync registries satisfies
#     ``asyncio.iscoroutinefunction(h) is False``.
#
# The invariant is mechanically asserted at module import below (after
# all registries are populated) and a runtime test
# (``test_request_interpretation_review_tool.py::test_dual_registry_invariant``)
# re-checks it so a future regression — for example, dropping an async
# tool into the sync registry by copy-paste — is caught structurally
# rather than producing silent "coroutine was never awaited" warnings.

# Unified handler signature: (arguments, state, catalog, data_dir) -> ToolResult.
# Handlers that don't need all parameters ignore them.
ToolHandler = Callable[
    [dict[str, Any], CompositionState, CatalogService, str | None],
    ToolResult,
]

RuntimePreflight = Callable[[CompositionState], ValidationResult]


class _UpsertNodeArgumentsModel(BaseModel):
    id: str
    node_type: NodeType
    input: str
    plugin: str | None = None
    on_success: str | None = None
    on_error: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)
    condition: str | None = None
    routes: dict[str, str] | None = None
    fork_to: list[str] | None = None
    branches: list[str] | dict[str, str] | None = None
    policy: str | None = None
    merge: str | None = None
    trigger: dict[str, Any] | None = None
    output_mode: str | None = None
    expected_output_count: int | None = None

    model_config = ConfigDict(extra="forbid")


class _UpsertEdgeArgumentsModel(BaseModel):
    id: str
    from_node: str
    to_node: str
    edge_type: EdgeType
    label: str | None = None

    model_config = ConfigDict(extra="forbid")


class _RemoveByIdArgumentsModel(BaseModel):
    id: str

    model_config = ConfigDict(extra="forbid")


class _SetMetadataPatchModel(BaseModel):
    name: str | None = None
    description: str | None = None

    model_config = ConfigDict(extra="forbid")


class _SetMetadataArgumentsModel(BaseModel):
    patch: _SetMetadataPatchModel

    model_config = ConfigDict(extra="forbid")


class _SetOutputArgumentsModel(BaseModel):
    sink_name: str
    plugin: str
    options: dict[str, Any]
    on_write_failure: str = "discard"

    model_config = ConfigDict(extra="forbid")


class _RemoveOutputArgumentsModel(BaseModel):
    sink_name: str

    model_config = ConfigDict(extra="forbid")


class _RequestInterpretationReviewArgumentsModel(BaseModel):
    """Tier-3 trust-boundary model for the ``request_interpretation_review`` tool.

    All three fields are LLM-supplied and constrained mechanically:

    * ``affected_node_id`` — short identifier; 256-char cap matches the wire
      cap used by ``upsert_node.id``.
    * ``user_term`` and ``llm_draft`` — capped at 8192 chars to defend against
      pathological inputs that would distend the audit row beyond the
      schema's 8192-byte expectation (see ``interpretation_events_table``
      column definitions; the schema-level body limit also bounds these
      at the ASGI boundary).

    ``extra="forbid"`` rejects unknown keys structurally so a misrouted
    argument shape (e.g., the LLM passing ``id`` instead of
    ``affected_node_id``) fails fast with a clear ARG_ERROR rather than
    silently dropping the typo and validating a partial payload.
    """

    affected_node_id: str = Field(min_length=1, max_length=256)
    user_term: str = Field(min_length=1, max_length=8192)
    llm_draft: str = Field(min_length=1, max_length=8192)

    model_config = ConfigDict(extra="forbid")


def _validate_mutation_arguments(model: type[BaseModel], arguments: object, argument_name: str) -> BaseModel:
    try:
        return model.model_validate(arguments)
    except PydanticValidationError as exc:
        raise ToolArgumentError(
            argument=argument_name,
            expected=f"object conforming to {model.__name__}",
            actual_type=type(exc).__name__,
        ) from exc


# Discovery tool handlers (normalized signatures)


def _handle_list_sources(
    arguments: dict[str, Any],
    state: CompositionState,
    catalog: CatalogService,
    data_dir: str | None = None,
) -> ToolResult:
    return _discovery_result(state, catalog.list_sources())


def _handle_list_transforms(
    arguments: dict[str, Any],
    state: CompositionState,
    catalog: CatalogService,
    data_dir: str | None = None,
) -> ToolResult:
    return _discovery_result(state, catalog.list_transforms())


def _handle_list_sinks(
    arguments: dict[str, Any],
    state: CompositionState,
    catalog: CatalogService,
    data_dir: str | None = None,
) -> ToolResult:
    return _discovery_result(state, catalog.list_sinks())


def _handle_get_plugin_schema(
    arguments: dict[str, Any],
    state: CompositionState,
    catalog: CatalogService,
    data_dir: str | None = None,
) -> ToolResult:
    try:
        schema = catalog.get_schema(arguments["plugin_type"], arguments["name"])
        return _discovery_result(state, schema)
    except (ValueError, KeyError) as exc:
        # ValueError: catalog contract for "unknown plugin/type"
        # KeyError: LLM omitted required argument (Tier 3)
        return _failure_result(state, str(exc))


def _handle_get_expression_grammar(
    arguments: dict[str, Any],
    state: CompositionState,
    catalog: CatalogService,
    data_dir: str | None = None,
) -> ToolResult:
    return _discovery_result(state, get_expression_grammar())


# Mutation tool handler wrappers (normalize 2/3-arg handlers to 4-arg)


def _attach_post_call_hints(
    result: ToolResult,
    catalog: CatalogService,
    *,
    plugin_type: PluginKind,
    tool_name: str,
    plugin_name: str | None,
    config_snapshot: Mapping[str, object],
) -> ToolResult:
    """Resolve postscript hints from the catalog and attach them to a successful result.

    No-ops when the mutation failed (we don't second-guess validation
    errors with coaching), when ``plugin_name`` is ``None`` (gates,
    coalesces — no plugin to resolve against), or when the plugin's
    ``get_post_call_hints`` returns an empty tuple (no hint to attach,
    so emit a result that doesn't carry the optional field at all).

    See ``contracts/plugin_assistance.py`` for the discipline and
    ``ToolResult.to_dict`` for the emission rule.
    """
    if not result.success or plugin_name is None:
        return result
    hints = catalog.post_call_hints(
        plugin_type=plugin_type,
        plugin_name=plugin_name,
        tool_name=tool_name,
        config_snapshot=config_snapshot,
    )
    if not hints:
        return result
    return replace(result, post_call_hints=hints)


def _handle_set_source(
    arguments: dict[str, Any],
    state: CompositionState,
    catalog: CatalogService,
    data_dir: str | None = None,
) -> ToolResult:
    result = _execute_set_source(arguments, state, catalog, data_dir)
    source_name = arguments.get("source_name", "source")
    source = result.updated_state.sources.get(source_name) if isinstance(source_name, str) else None
    if source is None:
        return result
    return _attach_post_call_hints(
        result,
        catalog,
        plugin_type="source",
        tool_name="set_source",
        plugin_name=source.plugin,
        config_snapshot=source.options,
    )


def _handle_upsert_node(
    arguments: dict[str, Any],
    state: CompositionState,
    catalog: CatalogService,
    data_dir: str | None = None,
) -> ToolResult:
    result = _execute_upsert_node(arguments, state, catalog)
    # The node may be a gate or coalesce (plugin=None) — _attach handles
    # that case. Extract the node identity from validated args so we
    # look up the right entry on the post-mutation state.
    try:
        validated = _UpsertNodeArgumentsModel.model_validate(arguments)
    except PydanticValidationError:
        # Validation failed inside _execute_upsert_node; the result
        # carries the failure. Skip hint resolution.
        return result
    node_id = validated.id
    node = next((n for n in result.updated_state.nodes if n.id == node_id), None)
    if node is None:
        return result
    return _attach_post_call_hints(
        result,
        catalog,
        plugin_type="transform",
        tool_name="upsert_node",
        plugin_name=node.plugin,
        config_snapshot=node.options,
    )


def _handle_upsert_edge(
    arguments: dict[str, Any],
    state: CompositionState,
    catalog: CatalogService,
    data_dir: str | None = None,
) -> ToolResult:
    return _execute_upsert_edge(arguments, state)


def _handle_remove_node(
    arguments: dict[str, Any],
    state: CompositionState,
    catalog: CatalogService,
    data_dir: str | None = None,
) -> ToolResult:
    return _execute_remove_node(arguments, state)


def _handle_remove_edge(
    arguments: dict[str, Any],
    state: CompositionState,
    catalog: CatalogService,
    data_dir: str | None = None,
) -> ToolResult:
    return _execute_remove_edge(arguments, state)


def _handle_set_metadata(
    arguments: dict[str, Any],
    state: CompositionState,
    catalog: CatalogService,
    data_dir: str | None = None,
) -> ToolResult:
    return _execute_set_metadata(arguments, state)


def _handle_set_output(
    arguments: dict[str, Any],
    state: CompositionState,
    catalog: CatalogService,
    data_dir: str | None = None,
) -> ToolResult:
    return _execute_set_output(arguments, state, catalog, data_dir)


def _handle_remove_output(
    arguments: dict[str, Any],
    state: CompositionState,
    catalog: CatalogService,
    data_dir: str | None = None,
) -> ToolResult:
    return _execute_remove_output(arguments, state)


def _discovery_result(state: CompositionState, data: Any) -> ToolResult:
    """Build a ToolResult for a discovery (read-only) tool."""
    validation = state.validate()
    return ToolResult(
        success=True,
        updated_state=state,
        validation=validation,
        affected_nodes=(),
        data=data,
    )


def _failure_result(
    state: CompositionState,
    error_msg: str,
) -> ToolResult:
    """Build a ToolResult for a failed mutation.

    The rejection reason (``error_msg``) is also prepended to
    ``validation.errors`` as a synthetic ``ValidationEntry`` with
    component ``"rejected_mutation"``. State-level errors from
    ``state.validate()`` (e.g. "No source configured.") describe the
    *unchanged* state and follow the rejection reason. This puts the
    action-rejection signal ahead of the stale-state signal for any
    consumer that reads ``validation.errors`` in array order — closing
    the convergence gap surfaced by composer session 58d7ede3 where the
    LLM repeated a near-identical ``set_pipeline`` because the array led
    with "No source configured." instead of the real option-shape
    error.
    """
    validation = _prepend_rejection_entry(state.validate(), error_msg)
    return ToolResult(
        success=False,
        updated_state=state,
        validation=validation,
        affected_nodes=(),
        data={_DATA_ERROR_KEY: error_msg},
    )


def _prepend_rejection_entry(
    base: ValidationSummary,
    error_msg: str,
) -> ValidationSummary:
    """Return a ValidationSummary with a leading rejected_mutation entry.

    Preserves all non-error fields (warnings, suggestions,
    edge_contracts, semantic_contracts) verbatim. ``is_valid`` is
    forced to False because a rejection entry is by construction a
    high-severity error.
    """
    rejection = ValidationEntry(
        component="rejected_mutation",
        message=error_msg,
        severity="high",
    )
    return ValidationSummary(
        is_valid=False,
        errors=(rejection, *base.errors),
        warnings=base.warnings,
        suggestions=base.suggestions,
        edge_contracts=base.edge_contracts,
        semantic_contracts=base.semantic_contracts,
    )


def _credential_wiring_contract_failure(
    state: CompositionState,
    *,
    component_id: str,
    component_type: str,
    options: Any,
) -> ToolResult | None:
    """Reject literal credentials before a mutation writes them into state.

    The returned message advertises the *inline* secret_ref form first
    because that is the only path that works for new nodes:

    - ``set_pipeline`` is atomic, so a node whose options omit a required
      credential field fails pydantic validation and the whole mutation
      rolls back — meaning ``wire_secret_ref`` cannot be used to attach
      the secret post-hoc (the node never lands in state).
    - ``collect_credential_field_violations`` short-circuits on
      ``{secret_ref: NAME}`` markers and ``set_pipeline`` strips those
      markers before pydantic validation, so passing the marker inline
      in the node's options is the supported new-node path.

    The post-hoc ``wire_secret_ref`` sequence is still documented as
    the secondary path for nodes that already exist in state.
    """
    fields = tuple(dict.fromkeys(collect_credential_field_violations(options)))
    if not fields:
        return None

    credential_fields = tuple(f"{component_id}:{field}" for field in fields)
    field_list = ", ".join(credential_fields)
    repair_sequence = ("list_secret_refs", "validate_secret_ref", "wire_secret_ref")
    repair_text = "list_secret_refs -> validate_secret_ref -> wire_secret_ref"
    inline_instruction = (
        "Set `<field>: {secret_ref: NAME}` directly in the node's options "
        "when calling set_pipeline / upsert_node. (The marker is stripped "
        "before option validation and resolved at execution time.)"
    )
    post_hoc_instruction = f"Alternatively, after the node already exists in state, call {repair_text} to attach the marker post-hoc."
    error_msg = (
        f"Credential field(s) contain literal value(s): {field_list}. "
        f"Literal credential values were not stored. {inline_instruction} "
        f"{post_hoc_instruction}"
    )
    # Symmetric with _failure_result: lead validation.errors with the
    # rejection reason so LLMs reading the array in order see the
    # actionable message before any stale-state errors.
    validation = _prepend_rejection_entry(state.validate(), error_msg)
    return ToolResult(
        success=False,
        updated_state=state,
        validation=validation,
        affected_nodes=(),
        data={
            _DATA_ERROR_KEY: error_msg,
            "credential_fields": credential_fields,
            "components": (
                {
                    "component_id": component_id,
                    "component_type": component_type,
                    "fields": fields,
                },
            ),
            "repair": {
                "inline_form": {
                    "instruction": inline_instruction,
                    "example_options": {field: {"secret_ref": "<NAME>"} for field in fields},
                },
                "post_hoc_form": {
                    "instruction": post_hoc_instruction,
                    "tool_sequence": repair_sequence,
                },
            },
        },
    )


def _mutation_result(
    new_state: CompositionState,
    affected: tuple[str, ...],
    *,
    prior_validation: ValidationSummary | None = None,
    data: Any = None,
    post_call_hints: tuple[str, ...] = (),
) -> ToolResult:
    """Build a ToolResult for a successful mutation.

    ``post_call_hints`` is the (possibly empty) tuple returned by the
    catalog's ``post_call_hints`` method for the just-configured
    plugin. Tool handlers compute it before calling here so the hint
    surface participates in the same envelope as ``validation`` and
    ``affected_nodes``. See ``contracts/plugin_assistance.py`` for
    the discipline; ``ToolResult.to_dict`` emits the field only when
    non-empty.
    """
    validation = new_state.validate()
    return ToolResult(
        success=True,
        updated_state=new_state,
        validation=validation,
        affected_nodes=affected,
        prior_validation=prior_validation,
        data=data,
        post_call_hints=post_call_hints,
    )


def _vf_destination_note(
    state: CompositionState,
    on_vf: str,
) -> dict[str, str] | None:
    """Advisory note when on_validation_failure references an unknown output.

    Returns a dict with a ``note`` key suitable for ``ToolResult.data``,
    or ``None`` when no advisory is needed (destination is ``"discard"``
    or matches a configured output).
    """
    if on_vf == "discard":
        return None
    output_names = {o.name for o in state.outputs}
    if on_vf not in output_names:
        current = sorted(output_names) if output_names else "(none)"
        return {
            "note": (
                f"on_validation_failure='{on_vf}' does not match any configured output. "
                "Use 'discard' to drop invalid rows without routing, or "
                f"add an output named '{on_vf}' before running the pipeline. "
                f"Current outputs: {current}."
            ),
        }
    return None


def _apply_merge_patch(
    target: Mapping[str, Any],
    patch: dict[str, Any],
) -> dict[str, Any]:
    """Shallow merge-patch: overwrite or delete top-level keys in target."""
    result = dict(target)
    for key, value in patch.items():
        if value is None:
            result.pop(key, None)
        else:
            result[key] = value
    return result


def _validate_plugin_name(
    catalog: CatalogService,
    plugin_type: PluginKind,
    name: str,
) -> str | None:
    """Return an error message if the plugin name is not in the catalog, or None if valid."""
    try:
        catalog.get_schema(plugin_type, name)
    except (ValueError, KeyError) as exc:
        return f"Unknown {plugin_type} plugin '{name}': {exc}"
    return None


def _validate_aggregation_trigger(trigger: Any) -> str | None:
    """Return an error message if an aggregation trigger does not match runtime settings."""
    if trigger is None:
        return None
    try:
        TriggerConfig.model_validate(trigger)
    except PydanticValidationError as exc:
        detail = "; ".join(str(error["msg"]) for error in exc.errors())
        return f"Invalid aggregation trigger: {detail}"
    return None


# --- Blob helpers (sync — called from worker thread via compose()) ---

_MIME_TO_SOURCE: dict[str, tuple[str, dict[str, str]]] = {
    "text/csv": ("csv", {}),
    "application/json": ("json", {}),
    "application/x-jsonlines": ("json", {"format": "jsonl"}),
    "application/jsonl": ("json", {"format": "jsonl"}),
    "text/jsonl": ("json", {"format": "jsonl"}),
    "text/plain": ("text", {}),
}


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


def _source_blob_payload(blob: BlobToolRecord) -> SourceBlobPayload:
    """Return source-blob metadata without leaking storage_path."""
    return {
        "blob_id": blob["id"],
        "filename": blob["filename"],
        "mime_type": blob["mime_type"],
        "size_bytes": blob["size_bytes"],
        "content_hash": blob["content_hash"],
    }


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
) -> _ResolvedSourceBlob | ToolResult:
    """Resolve an existing ready blob into authoritative source options."""
    if session_engine is None or session_id is None:
        return _failure_result(state, "Blob tools require session context.")
    blob = _sync_get_blob(session_engine, blob_id, session_id)
    if blob is None:
        return _failure_result(state, f"Blob '{blob_id}' not found.")

    if blob["status"] != "ready":
        return _failure_result(state, f"Blob is not ready (status: {blob['status']}).")

    mime_extra: dict[str, str] = {}
    if explicit_plugin:
        plugin = explicit_plugin
    else:
        mime_entry = _MIME_TO_SOURCE.get(blob["mime_type"])
        if mime_entry is None:
            return _failure_result(
                state,
                f"Cannot infer source plugin for MIME type '{blob['mime_type']}'. Please specify the 'plugin' parameter explicitly.",
            )
        plugin, mime_extra = mime_entry

    try:
        catalog.get_schema("source", plugin)
    except (ValueError, KeyError) as exc:
        return _failure_result(state, f"Unknown source plugin '{plugin}': {exc}")

    merged_options = {
        **caller_options,
        **mime_extra,
        "path": blob["storage_path"],
        "blob_ref": blob["id"],
    }
    prevalidation_error = _prevalidate_source(plugin, merged_options, on_validation_failure)
    if prevalidation_error is not None:
        return _failure_result(state, prevalidation_error)

    return _ResolvedSourceBlob(
        plugin=plugin,
        options=merged_options,
        payload=_source_blob_payload(blob),
    )


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


def _validate_source_path(
    options: Mapping[str, Any],
    data_dir: str | None,
) -> str | None:
    """S2: Validate that path/file options are under allowed source directories.

    Returns an error message if validation fails, None if OK.
    Uses Path.resolve() + is_relative_to() to defeat ../ traversal.
    """
    if data_dir is None:
        return None

    allowed = allowed_source_directories(data_dir)

    for key in ("path", "file"):
        if key in options:
            resolved = resolve_data_path(options[key], data_dir)
            if not any(resolved.is_relative_to(d) for d in allowed):
                return (
                    f"Path violation (S2): '{options[key]}' is outside the "
                    f"allowed directories. Source file paths "
                    f"must be under {data_dir}/blobs/."
                )
    return None


def _validate_sink_path(
    options: dict[str, Any],
    data_dir: str | None,
) -> str | None:
    """Validate that sink path options are under allowed output directories.

    Returns an error message if validation fails, None if OK.
    Mirrors _validate_source_path but uses _allowed_sink_directories.
    """
    if data_dir is None:
        return None

    allowed = allowed_sink_directories(data_dir)

    for key in ("path", "file"):
        if key in options:
            resolved = resolve_data_path(options[key], data_dir)
            if not any(resolved.is_relative_to(d) for d in allowed):
                return (
                    f"Path violation (S2): '{options[key]}' is outside the "
                    f"allowed directories. Sink output paths "
                    f"must be under {data_dir}/outputs/ or {data_dir}/blobs/."
                )
    return None


def _prevalidate_plugin_options(
    plugin_type: PluginKind,
    plugin_name: str,
    options: dict[str, Any],
    *,
    injected_fields: dict[str, Any] | None = None,
) -> str | None:
    """Pre-validate plugin options against the plugin's config model.

    Catches missing required options (e.g., schema, operations) and
    malformed values (e.g., invalid field specs) BEFORE storing them in
    CompositionState. Returns None if valid, or a descriptive error
    message suitable for returning to the LLM agent.

    The plugin's own Pydantic config model is the authority — this
    function asks the plugin what it needs rather than hardcoding
    knowledge about individual plugins.

    Secret-ref markers (``{"secret_ref": "NAME"}``) are stripped before
    validation. The underlying Pydantic errors are filtered to exclude
    errors on secret-ref'd fields — those fields ARE provisioned, just
    deferred to execution time when ``resolve_secret_refs`` replaces them
    with actual values.

    Args:
        plugin_type: "source", "transform", or "sink".
        plugin_name: Plugin name (e.g., "csv", "value_transform").
        options: Options dict as provided by the LLM agent.
        injected_fields: Synthetic values for fields that come from
            other parts of the pipeline spec (e.g., on_validation_failure
            for sources). Merged into options for validation only —
            not stored.
    """
    from pydantic import ValidationError

    from elspeth.plugins.infrastructure.config_base import PluginConfigError
    from elspeth.plugins.infrastructure.validation import (
        UnknownPluginTypeError,
        get_sink_config_model,
        get_source_config_model,
        get_transform_config_model,
    )

    secret_ref_placement_error = _secret_ref_placement_error(plugin_type, plugin_name, options)
    if secret_ref_placement_error is not None:
        return secret_ref_placement_error

    try:
        if plugin_type == "source":
            config_cls = get_source_config_model(plugin_name)
        elif plugin_type == "transform":
            config_cls = get_transform_config_model(plugin_name, options)
        elif plugin_type == "sink":
            config_cls = get_sink_config_model(plugin_name)
        else:
            # PluginKind is Literal["source", "transform", "sink"] — unreachable.
            raise AssertionError(f"_prevalidate_plugin_options: unexpected plugin_type={plugin_type!r}")
    except UnknownPluginTypeError:
        return f"Unknown {plugin_type} plugin '{plugin_name}'. Call list_{plugin_type}s to see available {plugin_type} plugins."
    except ValueError as exc:
        # Config model selection raised (e.g. unknown LLM provider) — surface it.
        return f"Invalid options for {plugin_type} '{plugin_name}': {exc}"

    if config_cls is None:
        return None

    # Options may contain frozen containers (MappingProxyType, tuple) from
    # CompositionState.  Thaw them so Pydantic receives plain dicts/lists.
    merged = deep_thaw(options)
    if injected_fields:
        for k, v in injected_fields.items():
            if k not in merged:
                merged[k] = v
    if plugin_type == "transform" and plugin_name == "llm":
        _mask_pending_interpretation_placeholders_for_authoring_validation(merged)

    # Strip secret_ref markers before validation.  A secret-ref'd field
    # IS provisioned (the user called wire_secret_ref), just deferred to
    # execution time.  Stripping it may cause Pydantic to report
    # "field required" — we filter those errors out below.
    secret_ref_keys: set[str] = set()
    for key, value in list(merged.items()):
        if isinstance(value, Mapping) and len(value) == 1 and "secret_ref" in value and isinstance(value["secret_ref"], str):
            secret_ref_keys.add(key)
            del merged[key]

    try:
        config_cls.from_dict(merged, plugin_name=plugin_name)
        return None
    except PluginConfigError as exc:
        if not secret_ref_keys:
            # No secret refs were stripped — report the error as-is.
            msg = exc.cause if exc.cause is not None else str(exc)
            return f"Invalid options for {plugin_type} '{plugin_name}': {msg}"

        # Secret refs were stripped.  Filter out errors on those fields.
        cause = exc.__cause__
        if not isinstance(cause, ValidationError):
            # ValueError path (model validators) — can't filter per-field.
            msg = exc.cause if exc.cause is not None else str(exc)
            return f"Invalid options for {plugin_type} '{plugin_name}': {msg}"

        remaining = [e for e in cause.errors() if not (e["loc"] and e["loc"][0] in secret_ref_keys)]
        if not remaining:
            return None

        # Re-format only the non-secret errors.
        lines = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in remaining)
        return f"Invalid options for {plugin_type} '{plugin_name}': {lines}"


def _mask_pending_interpretation_placeholders_for_authoring_validation(
    options: dict[str, Any],
) -> None:
    """Allow unresolved interpretation placeholders during composer authoring.

    ``{{interpretation:<term>}}`` is a Phase 5b composer-review token, not a
    runtime Jinja variable. The LLM must be able to stage a pending LLM node
    carrying that token so ``request_interpretation_review`` can create the
    audit row and the user can resolve it. Runtime remains strict: execution
    rejects unresolved placeholders before YAML generation, and resolved
    prompts validate through the normal LLM config path.
    """

    if "resolved_prompt_template_hash" in options:
        return
    prompt_template = options.get("prompt_template")
    if not isinstance(prompt_template, str):
        return
    options["prompt_template"] = INTERPRETATION_PLACEHOLDER_RE.sub(
        "pending interpretation",
        prompt_template,
    )


def _secret_ref_placement_error(
    plugin_type: PluginKind,
    plugin_name: str,
    options: dict[str, Any],
) -> str | None:
    """Return a policy error for secret_ref markers in non-credential fields."""
    secret_ref_placement_violations = collect_disallowed_secret_ref_markers(
        options,
        additional_allowed_fields=allowed_secret_ref_fields(plugin_type, plugin_name),
    )
    if not secret_ref_placement_violations:
        return None

    violation_text = ", ".join(f"{v.field_path} -> {v.secret_name}" for v in secret_ref_placement_violations)
    allowed_text = allowed_secret_ref_fields_text(plugin_type, plugin_name)
    return (
        f"Invalid secret_ref placement for {plugin_type} '{plugin_name}': {violation_text}; "
        "only credential-bearing fields may carry secret_ref markers. "
        f"Allowed credential-bearing fields: {allowed_text}."
    )


_WEB_ONLY_SOURCE_KEYS = frozenset({"blob_ref"})
_FILE_SINKS_REQUIRING_COLLISION_POLICY = frozenset({"csv", "json"})
_FILE_SINK_REPAIR_EXTENSIONS: Final[dict[str, str]] = {"csv": "csv", "json": "json"}
_WRITE_COLLISION_POLICIES = frozenset({"fail_if_exists", "auto_increment"})
_APPEND_COLLISION_POLICIES = frozenset({"append_or_create"})


def _missing_output_options_repair_error(
    *,
    sink_name: str,
    plugin_name: str,
    on_write_failure: str,
    validation_error: str | None,
) -> str:
    """Return an exact output-object repair hint for omitted sink options."""
    if plugin_name in _FILE_SINK_REPAIR_EXTENSIONS:
        path_fragment = _repair_identifier_fragment(sink_name, fallback="output")
        extension = _FILE_SINK_REPAIR_EXTENSIONS[plugin_name]
        repair_output = {
            "sink_name": sink_name,
            "plugin": plugin_name,
            "options": {
                "path": f"outputs/{path_fragment}.{extension}",
                "schema": {"mode": "observed"},
                "collision_policy": "auto_increment",
            },
            "on_write_failure": on_write_failure,
        }
        detail = f" Empty options were rejected: {validation_error}" if validation_error is not None else ""
        return (
            f"Output '{sink_name}' is missing options. For {plugin_name} file sinks, include "
            f"an options object with path, schema, and collision_policy. Use this output object "
            f"shape and adjust the path/schema if needed: {json.dumps(repair_output)}.{detail}"
        )

    repair_output = {
        "sink_name": sink_name,
        "plugin": plugin_name,
        "options": {},
        "on_write_failure": on_write_failure,
    }
    detail = f" Empty options were rejected: {validation_error}" if validation_error is not None else ""
    return (
        f"Output '{sink_name}' is missing options. Include the sink plugin's options object. "
        f"If this sink accepts empty configuration, use: {json.dumps(repair_output)}; otherwise "
        f"call get_plugin_schema for sink '{plugin_name}' and fill the required options.{detail}"
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


def validate_composer_file_sink_collision_policy(
    plugin_name: str,
    options: Mapping[str, Any],
    *,
    require_explicit: bool,
) -> str | None:
    """Require generated runnable file sinks to choose collision behavior."""
    if not require_explicit or plugin_name not in _FILE_SINKS_REQUIRING_COLLISION_POLICY:
        return None

    if "collision_policy" not in options:
        return (
            f"File sink '{plugin_name}' must set collision_policy explicitly. "
            "Use 'fail_if_exists' to refuse a taken output path, "
            "'auto_increment' to choose a free sibling path, or "
            "'append_or_create' with mode='append'."
        )

    mode = options.get("mode", "write")
    policy = options["collision_policy"]
    if mode == "append":
        if policy not in _APPEND_COLLISION_POLICIES:
            return f"File sink '{plugin_name}' with mode='append' must use collision_policy='append_or_create'."
    else:
        if policy not in _WRITE_COLLISION_POLICIES:
            return (
                f"File sink '{plugin_name}' with mode='write' must use "
                "collision_policy='fail_if_exists' or collision_policy='auto_increment'."
            )

    return None


def _prevalidate_source(
    plugin_name: str,
    options: Mapping[str, Any],
    on_validation_failure: str = _DEFAULT_SOURCE_VALIDATION_FAILURE,
) -> str | None:
    """Pre-validate source options, injecting on_validation_failure and filtering web-only keys."""
    filtered = {k: v for k, v in options.items() if k not in _WEB_ONLY_SOURCE_KEYS}
    return _prevalidate_plugin_options(
        "source",
        plugin_name,
        filtered,
        injected_fields={"on_validation_failure": on_validation_failure},
    )


def _source_component_id(source_name: str) -> str:
    """Return the legacy/default or named source component identifier."""
    return "source" if source_name == "source" else f"source:{source_name}"


def _prevalidate_transform(plugin_name: str, options: dict[str, Any]) -> str | None:
    """Pre-validate transform options."""
    return _prevalidate_plugin_options("transform", plugin_name, options)


def _prevalidate_sink(plugin_name: str, options: dict[str, Any]) -> str | None:
    """Pre-validate sink options."""
    return _prevalidate_plugin_options("sink", plugin_name, options)


def _execute_set_source(
    args: dict[str, Any],
    state: CompositionState,
    catalog: CatalogService,
    data_dir: str | None = None,
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
    plugin_error = _validate_plugin_name(catalog, "source", plugin)
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
    credential_error = _credential_wiring_contract_failure(
        state,
        component_id=_source_component_id(source_name),
        component_type="source",
        options=options,
    )
    if credential_error is not None:
        return credential_error

    # S2: Validate source path allowlist
    path_error = _validate_source_path(options, data_dir)
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


def _execute_upsert_node(
    args: dict[str, Any],
    state: CompositionState,
    catalog: CatalogService,
) -> ToolResult:
    """Add or update a pipeline node."""
    validated = cast(_UpsertNodeArgumentsModel, _validate_mutation_arguments(_UpsertNodeArgumentsModel, args, "upsert_node arguments"))
    node_id = validated.id
    node_type = validated.node_type
    plugin = validated.plugin
    node_options = validated.options
    credential_error = _credential_wiring_contract_failure(
        state,
        component_id=node_id,
        component_type="node",
        options=node_options,
    )
    if credential_error is not None:
        return credential_error

    # Validate plugin for types that require one.
    # Gates and coalesces intentionally have plugin=None (they're expression-based or
    # structural, not plugin-driven), so the "and plugin is not None" guard covers them.
    # NodeSpec documents this: "plugin: Plugin name. None for gates and coalesces."
    if node_type in ("transform", "aggregation") and plugin is not None:
        plugin_error = _validate_plugin_name(catalog, "transform", plugin)
        if plugin_error is not None:
            return _failure_result(state, plugin_error)

        batch_placement_error = _batch_aware_placement_error(node_id, node_type, plugin, validated.output_mode)
        if batch_placement_error is not None:
            return _failure_result(state, batch_placement_error)

        batch_required_error = _batch_aware_required_input_fields_error(node_id, plugin, node_options)
        if batch_required_error is not None:
            return _failure_result(state, batch_required_error)

        prevalidation_error = _prevalidate_transform(plugin, node_options)
        if prevalidation_error is not None:
            return _failure_result(state, prevalidation_error)

    # Validate gate condition expression at composition time.
    # Gives the LLM immediate feedback on syntax/security errors.
    condition = validated.condition
    if node_type == "gate" and condition is not None:
        expr_error = _validate_gate_expression(condition)
        if expr_error is not None:
            return _failure_result(state, f"Node '{node_id}': {expr_error}")
    if node_type == "aggregation":
        trigger_error = _validate_aggregation_trigger(validated.trigger)
        if trigger_error is not None:
            return _failure_result(state, f"Node '{node_id}': {trigger_error}")

    fork_to: tuple[str, ...] | None = tuple(validated.fork_to) if validated.fork_to is not None else None

    branches: CoalesceBranches | None = None
    if validated.branches is not None:
        branches = dict(validated.branches) if isinstance(validated.branches, Mapping) else tuple(validated.branches)

    node = NodeSpec(
        id=node_id,
        node_type=node_type,
        plugin=plugin,
        input=validated.input,
        on_success=validated.on_success,
        on_error=validated.on_error or ("discard" if node_type in ("transform", "aggregation") else None),
        options=node_options,
        condition=validated.condition,
        routes=validated.routes,
        fork_to=fork_to,
        branches=branches,
        policy=validated.policy,
        merge=validated.merge,
        trigger=validated.trigger,
        output_mode=validated.output_mode,
        expected_output_count=validated.expected_output_count,
    )

    new_state = state.with_node(node)

    # Affected: the node itself plus nodes with edges referencing it
    affected = {node_id}
    for edge in new_state.edges:
        if edge.from_node == node_id or edge.to_node == node_id:
            affected.add(edge.from_node)
            affected.add(edge.to_node)

    return _mutation_result(new_state, tuple(sorted(affected)))


def _execute_upsert_edge(
    args: dict[str, Any],
    state: CompositionState,
) -> ToolResult:
    """Add or update an edge.

    When the edge targets an output (sink), synchronises the source
    node's connection field so that generate_yaml() produces a
    working pipeline.  Edges to non-output nodes are visual only.
    """
    validated = cast(_UpsertEdgeArgumentsModel, _validate_mutation_arguments(_UpsertEdgeArgumentsModel, args, "upsert_edge arguments"))
    from_node = validated.from_node
    to_node = validated.to_node
    edge_type = validated.edge_type

    edge = EdgeSpec(
        id=validated.id,
        from_node=from_node,
        to_node=to_node,
        edge_type=edge_type,
        label=validated.label,
    )
    new_state = state.with_edge(edge)

    # Synchronise connection field when the edge targets an output.
    # generate_yaml() and the engine use on_success/on_error values
    # (not edges) to route data to sinks, so the connection field
    # must match the output name for the pipeline to work at runtime.
    output_names = {o.name for o in new_state.outputs}
    if to_node in output_names:
        if from_node == "source":
            if edge_type != "on_success":
                return _failure_result(state, "Source sink edges must use 'on_success'.")
            if new_state.source is not None and new_state.source.on_success != to_node:
                new_source = replace(new_state.source, on_success=to_node)
                new_state = new_state.with_source(new_source)
        else:
            node = next((n for n in new_state.nodes if n.id == from_node), None)
            if node is not None:
                if edge_type == "on_success":
                    if node.node_type == "gate":
                        return _failure_result(state, f"Gate '{from_node}' sink edges must use route_true, route_false, or fork.")
                    if node.on_success != to_node:
                        new_state = new_state.with_node(replace(node, on_success=to_node))
                elif edge_type == "on_error":
                    if node.node_type == "gate":
                        return _failure_result(state, f"Gate '{from_node}' sink edges must use route_true, route_false, or fork.")
                    if node.on_error != to_node:
                        new_state = new_state.with_node(replace(node, on_error=to_node))
                elif edge_type in ("route_true", "route_false"):
                    if node.node_type != "gate":
                        return _failure_result(state, f"Only gates can use '{edge_type}' edges to sinks.")
                    route_key = "true" if edge_type == "route_true" else "false"
                    routes = dict(node.routes or {})
                    if routes.get(route_key) != to_node:
                        routes[route_key] = to_node
                        new_state = new_state.with_node(replace(node, routes=routes))
                elif edge_type == "fork":
                    if node.node_type != "gate":
                        return _failure_result(state, "Only gates can use 'fork' edges to sinks.")
                    fork_targets = tuple(dict.fromkeys((*(node.fork_to or ()), to_node)))
                    if node.fork_to != fork_targets:
                        new_state = new_state.with_node(replace(node, fork_to=fork_targets))

    return _mutation_result(new_state, (from_node, to_node))


def _execute_remove_node(
    args: dict[str, Any],
    state: CompositionState,
) -> ToolResult:
    """Remove a node and its edges."""
    validated = cast(_RemoveByIdArgumentsModel, _validate_mutation_arguments(_RemoveByIdArgumentsModel, args, "remove_node arguments"))
    node_id = validated.id

    # Collect affected nodes before removal (edges that reference this node)
    affected = {node_id}
    for edge in state.edges:
        if edge.from_node == node_id or edge.to_node == node_id:
            affected.add(edge.from_node)
            affected.add(edge.to_node)

    new_state = state.without_node(node_id)
    if new_state is None:
        return _failure_result(state, f"Node '{node_id}' not found.")

    return _mutation_result(new_state, tuple(sorted(affected)))


def _execute_remove_edge(
    args: dict[str, Any],
    state: CompositionState,
) -> ToolResult:
    """Remove an edge."""
    validated = cast(_RemoveByIdArgumentsModel, _validate_mutation_arguments(_RemoveByIdArgumentsModel, args, "remove_edge arguments"))
    edge_id = validated.id

    # Find the edge to get affected nodes
    edge = next((e for e in state.edges if e.id == edge_id), None)
    if edge is None:
        return _failure_result(state, f"Edge '{edge_id}' not found.")

    affected = (edge.from_node, edge.to_node)
    new_state = state.without_edge(edge_id)
    if new_state is None:
        return _failure_result(state, f"Edge '{edge_id}' not found.")

    return _mutation_result(new_state, affected)


def _execute_set_metadata(
    args: dict[str, Any],
    state: CompositionState,
) -> ToolResult:
    """Update pipeline metadata."""
    validated = cast(_SetMetadataArgumentsModel, _validate_mutation_arguments(_SetMetadataArgumentsModel, args, "set_metadata arguments"))
    patch = validated.patch.model_dump(exclude_none=True)

    new_state = state.with_metadata(patch)
    return _mutation_result(new_state, ())


def _execute_set_output(
    args: dict[str, Any],
    state: CompositionState,
    catalog: CatalogService,
    data_dir: str | None = None,
) -> ToolResult:
    """Add or replace a pipeline output (sink)."""
    validated = cast(_SetOutputArgumentsModel, _validate_mutation_arguments(_SetOutputArgumentsModel, args, "set_output arguments"))
    plugin = validated.plugin
    # Validate plugin exists in catalog
    plugin_error = _validate_plugin_name(catalog, "sink", plugin)
    if plugin_error is not None:
        return _failure_result(state, plugin_error)

    # S2: Validate sink path allowlist (mirrors source path check)
    sink_options = validated.options
    credential_error = _credential_wiring_contract_failure(
        state,
        component_id=validated.sink_name,
        component_type="output",
        options=sink_options,
    )
    if credential_error is not None:
        return credential_error
    path_error = _validate_sink_path(sink_options, data_dir)
    if path_error is not None:
        return _failure_result(state, path_error)

    prevalidation_error = _prevalidate_sink(plugin, sink_options)
    if prevalidation_error is not None:
        return _failure_result(state, prevalidation_error)
    collision_error = validate_composer_file_sink_collision_policy(
        plugin,
        sink_options,
        require_explicit=data_dir is not None,
    )
    if collision_error is not None:
        return _failure_result(state, collision_error)

    output = OutputSpec(
        name=validated.sink_name,
        plugin=plugin,
        options=sink_options,
        on_write_failure=validated.on_write_failure,
    )
    new_state = state.with_output(output)
    return _mutation_result(new_state, (validated.sink_name,))


def _execute_remove_output(
    args: dict[str, Any],
    state: CompositionState,
) -> ToolResult:
    """Remove a pipeline output (sink) by name."""
    validated = cast(
        _RemoveOutputArgumentsModel, _validate_mutation_arguments(_RemoveOutputArgumentsModel, args, "remove_output arguments")
    )
    sink_name = validated.sink_name
    new_state = state.without_output(sink_name)
    if new_state is None:
        return _failure_result(state, f"Output '{sink_name}' not found.")
    return _mutation_result(new_state, (sink_name,))


# --- Blob tool handlers ---


def _handle_list_blobs(
    arguments: dict[str, Any],
    state: CompositionState,
    catalog: CatalogService,
    data_dir: str | None = None,
    *,
    session_engine: Engine | None = None,
    session_id: str | None = None,
) -> ToolResult:
    if session_engine is None or session_id is None:
        return _failure_result(state, "Blob tools require session context.")
    blobs = _sync_list_blobs(session_engine, session_id)
    return _discovery_result(state, blobs)


def _handle_get_blob_metadata(
    arguments: dict[str, Any],
    state: CompositionState,
    catalog: CatalogService,
    data_dir: str | None = None,
    *,
    session_engine: Engine | None = None,
    session_id: str | None = None,
) -> ToolResult:
    if session_engine is None or session_id is None:
        return _failure_result(state, "Blob tools require session context.")
    blob = _sync_get_blob(session_engine, arguments["blob_id"], session_id)
    if blob is None:
        return _failure_result(state, f"Blob '{arguments['blob_id']}' not found.")
    # Exclude storage_path from response
    safe_blob = {k: v for k, v in blob.items() if k != "storage_path"}
    return _discovery_result(state, safe_blob)


def _execute_set_source_from_blob(
    arguments: dict[str, Any],
    state: CompositionState,
    catalog: CatalogService,
    data_dir: str | None = None,
    *,
    session_engine: Engine | None = None,
    session_id: str | None = None,
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
        catalog=catalog,
        session_engine=session_engine,
        session_id=session_id,
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

# Default per-session blob storage quota (matches BlobServiceImpl).
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
) -> str | None:
    """Check if adding bytes would exceed the session blob quota.

    Returns an error message if quota exceeded, None if OK.
    Runs inside an existing transaction for TOCTOU safety.
    """
    current_total = conn.execute(
        select(func.coalesce(func.sum(blobs_table.c.size_bytes), 0)).where(blobs_table.c.session_id == session_id)
    ).scalar()
    current_total = int(current_total)
    resolved_quota = _resolve_blob_quota_bytes(quota_bytes)
    if current_total + additional_bytes > resolved_quota:
        return f"Session blob quota exceeded: {current_total + additional_bytes} bytes would exceed {resolved_quota} byte limit."
    return None


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
    catalog: CatalogService,
    data_dir: str | None = None,
    *,
    session_engine: Engine | None = None,
    session_id: str | None = None,
    user_message_id: str | None = None,
    max_blob_storage_per_session_bytes: int | None = None,
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
    if session_engine is None or session_id is None:
        return _failure_result(state, "Blob tools require session context.")
    if data_dir is None:
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
    # Provenance classification. The ``create_blob``
    # tool is invoked by the LLM as a self-directed action — the content
    # the LLM passes is, by construction, content the LLM authored as part
    # of its tool-call response. However, in this commit we tag every
    # create_blob payload as VERBATIM with NULL creating_* fields,
    # matching the set_pipeline.inline_blob path, until the call-loop
    # context (model identifier, version, provider, prompt hash) can
    # populate LLM_GENERATED. The
    # ``created_from_message_id`` still names the user turn that
    # triggered the LLM's response, so the audit walk works today.
    prepared = _prepare_blob_create(
        validated.model_dump(),
        data_dir=data_dir,
        session_id=session_id,
        creation_modality=CreationModality.VERBATIM,
        created_from_message_id=user_message_id,
    )

    quota_error = _persist_prepared_blob_create(
        prepared,
        session_engine=session_engine,
        session_id=session_id,
        max_blob_storage_per_session_bytes=max_blob_storage_per_session_bytes,
    )
    if quota_error is not None:
        return _failure_result(state, quota_error)

    return _discovery_result(state, _blob_create_payload(prepared))


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
    lock = _SESSION_BLOB_LOCKS.get(session_id)
    if lock is not None:
        return lock
    with _SESSION_BLOB_LOCKS_REGISTRY_MUTEX:
        lock = _SESSION_BLOB_LOCKS.get(session_id)
        if lock is None:
            lock = threading.Lock()
            _SESSION_BLOB_LOCKS[session_id] = lock
        return lock


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
    catalog: CatalogService,
    data_dir: str | None = None,
    *,
    session_engine: Engine | None = None,
    session_id: str | None = None,
    max_blob_storage_per_session_bytes: int | None = None,
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
    content = validated.content

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
        content_bytes = content.encode("utf-8")
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

                    # Atomic quota check.  ``size_bytes`` is re-read
                    # inside the transaction so the delta reflects the
                    # current DB row rather than a pre-transaction
                    # snapshot (stale under writers that bypass the
                    # composer session lock — e.g. ``BlobServiceImpl``
                    # paths that share the same session_engine).
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
                            quota_bytes=max_blob_storage_per_session_bytes,
                        )
                        if quota_error is not None:
                            # Raising inside the ``with`` rolls the DB
                            # transaction back before the outer handler
                            # runs.  ``os.replace`` has not executed,
                            # so storage_path is still the prior bytes
                            # and no rollback write is required.
                            raise _BlobQuotaExceededInTxn(quota_error)

                    conn.execute(
                        update(blobs_table)
                        .where(
                            blobs_table.c.id == blob_id,
                            blobs_table.c.session_id == session_id,
                        )
                        .values(size_bytes=new_size, content_hash=file_hash)
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


def _execute_delete_blob(
    arguments: dict[str, Any],
    state: CompositionState,
    catalog: CatalogService,
    data_dir: str | None = None,
    *,
    session_engine: Engine | None = None,
    session_id: str | None = None,
) -> ToolResult:
    """Delete a blob and its storage file."""
    if session_engine is None or session_id is None:
        return _failure_result(state, "Blob tools require session context.")

    blob_id = arguments["blob_id"]

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
    blob_id = blob["id"]
    stored_hash = blob["content_hash"]
    if stored_hash is None:
        raise AuditIntegrityError(f"Tier 1: ready blob {blob_id} has NULL content_hash — DB integrity anomaly, cannot verify")
    actual_hash = content_hash(data)
    if not hmac.compare_digest(actual_hash, stored_hash):
        raise BlobIntegrityError(blob_id, expected=stored_hash, actual=actual_hash)


def _execute_get_blob_content(
    arguments: dict[str, Any],
    state: CompositionState,
    catalog: CatalogService,
    data_dir: str | None = None,
    *,
    session_engine: Engine | None = None,
    session_id: str | None = None,
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
    if session_engine is None or session_id is None:
        return _failure_result(state, "Blob tools require session context.")

    blob_id = arguments["blob_id"]
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


def _execute_inspect_source(
    arguments: dict[str, Any],
    state: CompositionState,
    catalog: CatalogService,
    data_dir: str | None = None,
    *,
    session_engine: Engine | None = None,
    session_id: str | None = None,
) -> ToolResult:
    """Inspect a blob-backed source and return bounded structural facts.

    Mirrors the lifecycle and integrity guards of ``_execute_get_blob_content``
    (only ``ready`` blobs are readable; SHA-256 verified; UnicodeDecodeError
    surfaced as tool-failure) but returns ``SourceInspectionFacts`` rather
    than raw content. Reads at most 8 KiB and parses at most 100 rows.

    Never returns raw row content — only summary facts (headers, inferred
    types, URL candidates, warnings, redacted identity).
    """
    if session_engine is None or session_id is None:
        return _failure_result(state, "Blob tools require session context.")

    blob_id = arguments["blob_id"]
    blob = _sync_get_blob(session_engine, blob_id, session_id)
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


def _execute_list_recipes(
    arguments: dict[str, Any],
    state: CompositionState,
    catalog: CatalogService,
    data_dir: str | None = None,
) -> ToolResult:
    """Return discovery metadata for every registered pipeline recipe."""
    return _discovery_result(state, {"recipes": list_recipes()})


def _execute_apply_pipeline_recipe(
    arguments: dict[str, Any],
    state: CompositionState,
    catalog: CatalogService,
    data_dir: str | None = None,
    *,
    session_engine: Engine | None = None,
    session_id: str | None = None,
    user_message_id: str | None = None,
    max_blob_storage_per_session_bytes: int | None = None,
) -> ToolResult:
    """Validate a recipe's slots, build set_pipeline args, and dispatch to set_pipeline.

    Tier-3 boundary: ``arguments`` is an LLM-supplied dict.  Validated
    via :class:`ApplyPipelineRecipeArgumentsModel` (the single source of
    truth for the argument schema — supersedes the deleted
    ``_TOOL_REQUIRED_PATHS["apply_pipeline_recipe"]`` entry in
    ``service.py``, rev-3 N7 / rev-4 M1).  On
    :class:`pydantic.ValidationError` the handler re-raises as
    :class:`ToolArgumentError` so the compose loop's ARG_ERROR routing
    at ``service.py:2480`` receives the right exception class.

    Semantic vs argument-shape failures
    ------------------------------------
    Pydantic enforces argument shape (type, required-fields, extra=forbid).
    The empty-``recipe_name`` semantic check and the
    :class:`RecipeValidationError` slot-shape check remain in this handler
    and produce recoverable ``_failure_result`` responses with repair
    hints (``Call list_recipes to discover available recipes``).  Two
    channels for two failure shapes (type vs semantic) — same pattern as
    :class:`SetSourceArgumentsModel` plugin-not-in-catalog handling.

    ``set_pipeline`` is full state replacement, so a ``replaced_pipeline_note`` is
    emitted to make the destructive replacement visible to the LLM/operator. The
    note is suppressed when the prior pipeline is empty — a no-op replacement
    needs no flag, and emitting one would be noise on a fresh-session apply.
    """
    try:
        validated = ApplyPipelineRecipeArgumentsModel.model_validate(arguments)
    except PydanticValidationError as exc:
        raise ToolArgumentError(
            argument="apply_pipeline_recipe arguments",
            expected="object conforming to ApplyPipelineRecipeArgumentsModel",
            actual_type=type(exc).__name__,
        ) from exc

    recipe_name = validated.recipe_name
    raw_slots = validated.slots
    if not recipe_name:
        # Empty-string recipe_name passes Pydantic's ``str`` validation
        # but the handler treats it as a recoverable semantic failure
        # with a repair-hint pointing the LLM at list_recipes (rather
        # than the generic ARG_ERROR envelope a Pydantic min_length=1
        # would produce).
        return _failure_result(
            state,
            "apply_pipeline_recipe requires a non-empty 'recipe_name' string. Call list_recipes to discover available recipes.",
        )

    try:
        pipeline_args = apply_recipe(recipe_name, dict(raw_slots))
    except RecipeValidationError as exc:
        return _failure_result(state, str(exc))

    # Capture pre-replacement counts BEFORE delegating to the destructive
    # set_pipeline path. Frozen-dataclass fields, so capturing the integers
    # now is sufficient — the post-call result.updated_state is a fresh
    # CompositionState produced by set_pipeline.
    pre_source_present = state.source is not None
    pre_node_count = len(state.nodes)
    pre_output_count = len(state.outputs)

    # Delegate to the existing set_pipeline executor — recipes produce the
    # exact arguments shape set_pipeline accepts, so validation and state
    # mutation flow through the canonical mutation path.
    result = _execute_set_pipeline(
        pipeline_args,
        state,
        catalog,
        data_dir,
        session_engine=session_engine,
        session_id=session_id,
        user_message_id=user_message_id,
        max_blob_storage_per_session_bytes=max_blob_storage_per_session_bytes,
    )

    # Only annotate successful replacements over a non-empty prior state.
    # On failure, set_pipeline returned ``state`` unchanged and the note
    # would be misleading. On a fresh-session apply, there is nothing to
    # call out and a note would be noise.
    if not result.success:
        return result
    if not (pre_source_present or pre_node_count or pre_output_count):
        return result

    note = (
        f"apply_pipeline_recipe replaced the existing pipeline "
        f"(prior state had source={'set' if pre_source_present else 'unset'}, "
        f"{pre_node_count} node(s), {pre_output_count} output(s)). "
        "Recipes are full-state scaffolds; the prior composition was discarded."
    )

    # Preserve any existing data payload from set_pipeline (e.g. inline-blob
    # creation summary) by merging into a single dict. set_pipeline's
    # ``data`` is currently None on the recipe path because recipes don't
    # use inline_blob, but merging is forward-compatible.
    existing_data = result.data
    if existing_data is None:
        merged_data: Any = {"replaced_pipeline_note": note}
    elif isinstance(existing_data, Mapping):
        merged_data = {**dict(existing_data), "replaced_pipeline_note": note}
    else:
        # set_pipeline contract: ``data`` is None or a Mapping. Anything
        # else is a contract drift bug — surface the note alongside in a
        # wrapper rather than silently dropping either.
        merged_data = {"replaced_pipeline_note": note, "set_pipeline_data": existing_data}

    return replace(result, data=merged_data)


# Blob tool handler type — extended signature with session context
BlobToolHandler = Callable[..., ToolResult]

# --- Secret tool handlers ---

# Secret tool handler type — extended signature with secret_service + user_id
SecretToolHandler = Callable[..., ToolResult]


def _handle_list_secret_refs(
    arguments: dict[str, Any],
    state: CompositionState,
    catalog: CatalogService,
    data_dir: str | None = None,
    *,
    secret_service: Any | None = None,
    user_id: str | None = None,
) -> ToolResult:
    if secret_service is None or user_id is None:
        return _failure_result(state, "Secret tools require secret service context.")
    items = secret_service.list_refs(user_id)
    # Return inventory dicts — NEVER include values
    data = [{"name": item.name, "scope": item.scope, "available": item.available} for item in items]
    return _discovery_result(state, data)


def _handle_validate_secret_ref(
    arguments: dict[str, Any],
    state: CompositionState,
    catalog: CatalogService,
    data_dir: str | None = None,
    *,
    secret_service: Any | None = None,
    user_id: str | None = None,
) -> ToolResult:
    if secret_service is None or user_id is None:
        return _failure_result(state, "Secret tools require secret service context.")
    name = arguments["name"]
    available = secret_service.has_ref(user_id, name)
    return _discovery_result(state, {"name": name, "available": available})


def _execute_wire_secret_ref(
    arguments: dict[str, Any],
    state: CompositionState,
    catalog: CatalogService,
    data_dir: str | None = None,
    *,
    secret_service: Any | None = None,
    user_id: str | None = None,
) -> ToolResult:
    if secret_service is None or user_id is None:
        return _failure_result(state, "Secret tools require secret service context.")

    name = arguments["name"]
    target = arguments["target"]
    option_key = arguments["option_key"]
    target_id = arguments.get("target_id")

    # Validate the secret ref exists
    if not secret_service.has_ref(user_id, name):
        return _failure_result(state, f"Secret reference '{name}' not found or not accessible.")

    marker = {"secret_ref": name}

    if target == "source":
        if state.source is None:
            return _failure_result(state, "No source configured — set a source first.")
        patched_options = dict(deep_thaw(state.source.options))
        patched_options[option_key] = marker
        placement_error = _secret_ref_placement_error("source", state.source.plugin, patched_options)
        if placement_error is not None:
            return _failure_result(state, placement_error)
        new_source = SourceSpec(
            plugin=state.source.plugin,
            on_success=state.source.on_success,
            options=patched_options,
            on_validation_failure=state.source.on_validation_failure,
        )
        new_state = state.with_source(new_source)
        return _mutation_result(new_state, ("source",))

    elif target == "node":
        if target_id is None:
            return _failure_result(state, "target_id is required for node targets.")
        node = next((n for n in state.nodes if n.id == target_id), None)
        if node is None:
            return _failure_result(state, f"Node '{target_id}' not found.")
        if node.node_type not in ("transform", "aggregation") or node.plugin is None:
            return _failure_result(
                state,
                "Secret references can only be wired into source, transform, aggregation, or output plugin options.",
            )
        patched_options = dict(deep_thaw(node.options))
        patched_options[option_key] = marker
        placement_error = _secret_ref_placement_error("transform", node.plugin, patched_options)
        if placement_error is not None:
            return _failure_result(state, placement_error)
        new_node = NodeSpec(
            id=node.id,
            node_type=node.node_type,
            plugin=node.plugin,
            input=node.input,
            on_success=node.on_success,
            on_error=node.on_error,
            options=patched_options,
            condition=node.condition,
            routes=deep_thaw(node.routes) if node.routes is not None else None,
            fork_to=node.fork_to,
            branches=node.branches,
            policy=node.policy,
            merge=node.merge,
            trigger=deep_thaw(node.trigger) if node.trigger is not None else None,
            output_mode=node.output_mode,
            expected_output_count=node.expected_output_count,
        )
        new_state = state.with_node(new_node)
        return _mutation_result(new_state, (target_id,))

    elif target == "output":
        if target_id is None:
            return _failure_result(state, "target_id is required for output targets.")
        output = next((o for o in state.outputs if o.name == target_id), None)
        if output is None:
            return _failure_result(state, f"Output '{target_id}' not found.")
        patched_options = dict(deep_thaw(output.options))
        patched_options[option_key] = marker
        placement_error = _secret_ref_placement_error("sink", output.plugin, patched_options)
        if placement_error is not None:
            return _failure_result(state, placement_error)
        new_output = OutputSpec(
            name=output.name,
            plugin=output.plugin,
            options=patched_options,
            on_write_failure=output.on_write_failure,
        )
        new_state = state.with_output(new_output)
        return _mutation_result(new_state, (target_id,))

    else:
        return _failure_result(state, f"Unknown target type: '{target}'.")


# --- Atomic set_pipeline handler ---


def _execute_set_pipeline(
    args: dict[str, Any],
    state: CompositionState,
    catalog: CatalogService,
    data_dir: str | None = None,
    *,
    session_engine: Engine | None = None,
    session_id: str | None = None,
    user_message_id: str | None = None,
    max_blob_storage_per_session_bytes: int | None = None,
) -> ToolResult:
    """Atomically replace the entire pipeline composition state.

    Tier-3 boundary: ``args`` is an LLM-supplied dict.  Validated via the
    Pydantic redaction-bearing model :class:`SetPipelineArgumentsModel` (the
    single source of truth for the argument schema — supersedes the deleted
    ``_TOOL_REQUIRED_PATHS["set_pipeline"]`` entry in ``service.py``,
    rev-3 N7 / rev-4 M1).

    On :class:`pydantic.ValidationError` the handler re-raises as
    :class:`ToolArgumentError` so the compose loop's ARG_ERROR routing at
    ``service.py:2480`` receives the right exception class.

    Dispatcher-wired kwargs
    -----------------------
    The dispatcher at :func:`execute_tool` (``tools.py:5530-5540``) supplies
    ``session_engine`` and ``session_id`` as kwargs.  These are NOT part of
    the LLM-supplied ``arguments`` dict — they are wired from the composer
    service request context — so they are NOT modelled in
    :class:`SetPipelineArgumentsModel`.  The Pydantic model validates only
    the LLM-supplied dict; the kwargs enter through the function signature.

    Semantic vs argument-shape failures
    ------------------------------------
    Pydantic enforces argument shape (type, required-fields, extra=forbid).
    Per-component semantic checks (plugin existence in catalog, path
    allowlist, manual blob_ref injection rejection, source.blob_id +
    source.inline_blob exclusivity, gate-condition expression validity)
    remain in this handler and produce recoverable ``_failure_result``
    responses with repair hints.  Two channels for two failure shapes
    (type vs semantic) — same pattern as
    :class:`SetSourceArgumentsModel` plugin-not-in-catalog handling.
    """
    try:
        validated = SetPipelineArgumentsModel.model_validate(args)
    except PydanticValidationError as exc:
        raise ToolArgumentError(
            argument="set_pipeline arguments",
            expected="object conforming to SetPipelineArgumentsModel",
            actual_type=type(exc).__name__,
        ) from exc

    if validated.source is not None and validated.sources is not None:
        return _failure_result(state, "set_pipeline must use either source or sources, not both.")
    if validated.source is None and validated.sources is None:
        return _failure_result(state, "set_pipeline requires source or sources.")

    source_specs: dict[str, SourceSpec] = {}
    prepared_inline_blob: _PreparedBlobCreate | None = None
    resolved_source_blob: _ResolvedSourceBlob | None = None
    single_source_on_vf: str | None = None

    if validated.sources is not None:
        if not validated.sources:
            return _failure_result(state, "set_pipeline sources must include at least one named source.")
        for source_name, source_model in validated.sources.items():
            if not source_name.strip():
                return _failure_result(state, "set_pipeline sources keys must be non-empty source names.")
            if source_model.blob_id is not None or source_model.inline_blob is not None:
                return _failure_result(
                    state,
                    f"set_pipeline sources.{source_name} cannot use blob_id or inline_blob in v1. "
                    "Bind blob-backed sources with set_source_from_blob, or use source for a single blob-backed pipeline.",
                )
            src_plugin = source_model.plugin
            plugin_error = _validate_plugin_name(catalog, "source", src_plugin)
            if plugin_error is not None:
                return _failure_result(state, f"Source '{source_name}': {plugin_error}")
            src_options = dict(source_model.options)
            manual_blob_ref_error = _reject_manual_source_blob_ref(src_options, tool_name="set_pipeline")
            if manual_blob_ref_error is not None:
                return _failure_result(state, f"Source '{source_name}': {manual_blob_ref_error}")
            credential_error = _credential_wiring_contract_failure(
                state,
                component_id=_source_component_id(source_name),
                component_type="source",
                options=src_options,
            )
            if credential_error is not None:
                return credential_error
            src_on_vf = source_model.on_validation_failure or _DEFAULT_SOURCE_VALIDATION_FAILURE
            path_error = _validate_source_path(src_options, data_dir)
            if path_error is not None:
                return _failure_result(state, f"Source '{source_name}': {path_error}")
            src_prevalidation = _prevalidate_source(src_plugin, src_options, src_on_vf)
            if src_prevalidation is not None:
                return _failure_result(state, f"Source '{source_name}': {src_prevalidation}")
            source_specs[source_name] = SourceSpec(
                plugin=src_plugin,
                on_success=source_model.on_success,
                options=src_options,
                on_validation_failure=src_on_vf,
            )
    else:
        legacy_source_model = validated.source
        if legacy_source_model is None:
            raise AssertionError("validated.source unexpectedly None after source/sources gate")
        src_plugin = legacy_source_model.plugin
        plugin_error = _validate_plugin_name(catalog, "source", src_plugin)
        if plugin_error is not None:
            return _failure_result(state, plugin_error)

        # Inline user-provided source data can be materialised as a blob inside
        # this same atomic pipeline mutation. The generated path/blob_ref are
        # authoritative exactly as if create_blob + set_source_from_blob had been
        # called, but the LLM gets one audited tool decision instead of a serial
        # blob-then-source-then-pipeline conversation.
        legacy_src_options: Mapping[str, Any] = dict(legacy_source_model.options)
        manual_blob_ref_error = _reject_manual_source_blob_ref(legacy_src_options, tool_name="set_pipeline", inline_blob_supported=True)
        if manual_blob_ref_error is not None:
            return _failure_result(state, manual_blob_ref_error)
        credential_error = _credential_wiring_contract_failure(
            state,
            component_id="source",
            component_type="source",
            options=legacy_src_options,
        )
        if credential_error is not None:
            return credential_error
        source_blob_id = legacy_source_model.blob_id
        inline_blob = legacy_source_model.inline_blob
        src_on_vf = legacy_source_model.on_validation_failure or _DEFAULT_SOURCE_VALIDATION_FAILURE
        single_source_on_vf = src_on_vf
        if source_blob_id is not None and inline_blob is not None:
            return _failure_result(state, "set_pipeline source must use either an existing blob_id or inline_blob, not both.")
        if source_blob_id is not None:
            resolved = _resolve_source_blob(
                blob_id=source_blob_id,
                explicit_plugin=src_plugin,
                caller_options=legacy_src_options,
                on_validation_failure=src_on_vf,
                state=state,
                catalog=catalog,
                session_engine=session_engine,
                session_id=session_id,
            )
            if isinstance(resolved, ToolResult):
                return resolved
            resolved_source_blob = resolved
            src_plugin = resolved.plugin
            legacy_src_options = resolved.options
        if inline_blob is not None:
            if session_engine is None or session_id is None:
                return _failure_result(state, "set_pipeline source.inline_blob requires session context.")
            if data_dir is None:
                return _failure_result(state, "set_pipeline source.inline_blob requires data_dir for storage.")
            prepared_inline_blob = _prepare_blob_create(
                inline_blob.model_dump(),
                data_dir=data_dir,
                session_id=session_id,
                creation_modality=CreationModality.VERBATIM,
                created_from_message_id=user_message_id,
            )
            header_conflict = _header_only_inline_csv_conflict(
                prepared_inline_blob,
                session_engine=session_engine,
                session_id=session_id,
            )
            if header_conflict is not None:
                return _failure_result(state, header_conflict)

            mime_entry = _MIME_TO_SOURCE.get(prepared_inline_blob.mime_type)
            mime_options: dict[str, str] = {}
            if mime_entry is not None:
                inferred_plugin, inferred_options = mime_entry
                if inferred_plugin == src_plugin:
                    mime_options = inferred_options
            legacy_src_options = {
                **legacy_src_options,
                **mime_options,
                "path": str(prepared_inline_blob.storage_path),
                "blob_ref": prepared_inline_blob.blob_id,
            }

        path_error = _validate_source_path(legacy_src_options, data_dir)
        if path_error is not None:
            return _failure_result(state, path_error)

        src_prevalidation = _prevalidate_source(src_plugin, legacy_src_options, src_on_vf)
        if src_prevalidation is not None:
            return _failure_result(state, src_prevalidation)
        source_specs["source"] = SourceSpec(
            plugin=src_plugin,
            on_success=legacy_source_model.on_success,
            options=legacy_src_options,
            on_validation_failure=src_on_vf,
        )

    # 2. Validate node plugins and options
    for node in validated.nodes:
        node_id = node.id
        node_type = node.node_type
        node_plugin = node.plugin
        node_options = node.options
        credential_error = _credential_wiring_contract_failure(
            state,
            component_id=node_id,
            component_type="node",
            options=node_options,
        )
        if credential_error is not None:
            return credential_error
        if node_type in ("transform", "aggregation") and node_plugin is not None:
            plugin_error = _validate_plugin_name(catalog, "transform", node_plugin)
            if plugin_error is not None:
                return _failure_result(state, f"Node '{node_id}': {plugin_error}")
            batch_placement_error = _batch_aware_placement_error(node_id, node_type, node_plugin, node.output_mode)
            if batch_placement_error is not None:
                return _failure_result(state, f"Node '{node_id}': {batch_placement_error}")
            batch_required_error = _batch_aware_required_input_fields_error(node_id, node_plugin, node_options)
            if batch_required_error is not None:
                return _failure_result(state, f"Node '{node_id}': {batch_required_error}")

            node_prevalidation = _prevalidate_transform(node_plugin, node_options)
            if node_prevalidation is not None:
                return _failure_result(state, f"Node '{node_id}': {node_prevalidation}")

        # Validate gate condition expression at composition time.
        if node_type == "gate" and node.condition is not None:
            expr_error = _validate_gate_expression(node.condition)
            if expr_error is not None:
                return _failure_result(state, f"Node '{node_id}': {expr_error}")

    # 3. Validate output plugins and options
    #
    # ``options_missing`` distinguishes "operator omitted the options key
    # entirely" from "operator supplied options: {}".  Post-Pydantic the
    # default-factory replaces an absent key with ``{}`` on the model side,
    # so we look at the raw ``args`` dict to recover the operator's
    # original intent for the repair-hint branch.  The semantic-validation
    # branch (file-sink collision policy, path allowlist) still runs on
    # the validated dict.
    raw_outputs = args.get("outputs") if isinstance(args, Mapping) else None
    for index, output in enumerate(validated.outputs):
        out_name = output.sink_name
        out_plugin = output.plugin
        plugin_error = _validate_plugin_name(catalog, "sink", out_plugin)
        if plugin_error is not None:
            return _failure_result(state, f"Output '{out_name}': {plugin_error}")
        out_options = output.options
        raw_out_args: Mapping[str, Any] = {}
        if isinstance(raw_outputs, list) and 0 <= index < len(raw_outputs):
            raw_entry = raw_outputs[index]
            if isinstance(raw_entry, Mapping):
                raw_out_args = raw_entry
        options_missing = "options" not in raw_out_args
        if options_missing:
            out_prevalidation = _prevalidate_sink(out_plugin, out_options)
            out_collision_error = validate_composer_file_sink_collision_policy(
                out_plugin,
                out_options,
                require_explicit=data_dir is not None,
            )
            validation_error = out_prevalidation if out_prevalidation is not None else out_collision_error
            if validation_error is not None:
                return _failure_result(
                    state,
                    _missing_output_options_repair_error(
                        sink_name=out_name,
                        plugin_name=out_plugin,
                        on_write_failure=output.on_write_failure if output.on_write_failure is not None else "discard",
                        validation_error=validation_error,
                    ),
                )
        credential_error = _credential_wiring_contract_failure(
            state,
            component_id=out_name,
            component_type="output",
            options=out_options,
        )
        if credential_error is not None:
            return credential_error
        out_path_error = _validate_sink_path(out_options, data_dir)
        if out_path_error is not None:
            return _failure_result(state, f"Output '{out_name}': {out_path_error}")
        out_prevalidation = _prevalidate_sink(out_plugin, out_options)
        if out_prevalidation is not None:
            return _failure_result(state, f"Output '{out_name}': {out_prevalidation}")
        out_collision_error = validate_composer_file_sink_collision_policy(
            out_plugin,
            out_options,
            require_explicit=data_dir is not None,
        )
        if out_collision_error is not None:
            return _failure_result(state, f"Output '{out_name}': {out_collision_error}")

    # 4. Construct specs (same field extraction as individual handlers)
    # ``node_type`` / ``edge_type`` are typed as ``str`` on
    # ``_PipelineNodeModel`` / ``_PipelineEdgeModel`` to preserve Tier-3
    # LLM-recoverable feedback (the handler reports unknown enum values
    # via semantic _failure_result; a Pydantic Literal rejection would
    # surface as ARG_ERROR with no repair guidance).  At the point we
    # construct :class:`NodeSpec` / :class:`EdgeSpec`, the downstream
    # validation (``_validate_plugin_name``, graph topology) and the
    # ``_batch_aware_placement_error`` checks above have not yet
    # rejected non-canonical enum values explicitly — that responsibility
    # remains on the state-level dataclass invariants.  The ``cast`` here
    # narrows the static type without re-validating; semantically wrong
    # enum values flow through to be rejected at runtime by NodeSpec /
    # EdgeSpec / CompositionState invariants.
    node_specs = []
    for n in validated.nodes:
        fork_to = tuple(n.fork_to) if n.fork_to is not None else None
        branches = dict(n.branches) if isinstance(n.branches, Mapping) else tuple(n.branches) if n.branches is not None else None
        nt = n.node_type
        node_specs.append(
            NodeSpec(
                id=n.id,
                node_type=cast(NodeType, nt),
                plugin=n.plugin,
                input=n.input,
                on_success=n.on_success,
                on_error=n.on_error or ("discard" if nt in ("transform", "aggregation") else None),
                options=n.options,
                condition=n.condition,
                routes=n.routes,
                fork_to=fork_to,
                branches=branches,
                policy=n.policy,
                merge=n.merge,
                # ``n.trigger`` is a typed :class:`_NodeTriggerModel` (or None) per F3 —
                # convert to a plain dict at the NodeSpec boundary because
                # :class:`NodeSpec.trigger` is typed ``Mapping[str, Any] | None`` and
                # is deep-frozen by ``freeze_fields("trigger")``; the freeze contract
                # requires a Mapping, not a Pydantic model instance.
                trigger=n.trigger.model_dump() if n.trigger is not None else None,
                output_mode=n.output_mode,
                expected_output_count=n.expected_output_count,
            )
        )

    edge_specs = []
    for e in validated.edges:
        edge_specs.append(
            EdgeSpec(
                id=e.id,
                from_node=e.from_node,
                to_node=e.to_node,
                edge_type=cast(EdgeType, e.edge_type),
                label=e.label,
            )
        )

    output_specs = []
    for o in validated.outputs:
        output_specs.append(
            OutputSpec(
                name=o.sink_name,
                plugin=o.plugin,
                options=o.options,
                on_write_failure=o.on_write_failure if o.on_write_failure is not None else "discard",
            )
        )

    # PipelineMetadata's __init__ supplies its own defaults for ``name`` and
    # ``description``; honour those by passing through only explicitly
    # supplied fields.  ``validated.metadata`` is None when the LLM omitted
    # the ``metadata`` key entirely.
    meta_kwargs: dict[str, str] = {}
    if validated.metadata is not None:
        if validated.metadata.name is not None:
            meta_kwargs["name"] = validated.metadata.name
        if validated.metadata.description is not None:
            meta_kwargs["description"] = validated.metadata.description
    metadata_spec = PipelineMetadata(**meta_kwargs)

    # 5. Build new state
    new_state = CompositionState(
        source=next(iter(source_specs.values())),
        sources=source_specs,
        nodes=tuple(node_specs),
        edges=tuple(edge_specs),
        outputs=tuple(output_specs),
        metadata=metadata_spec,
        version=state.version + 1,
    )

    if prepared_inline_blob is not None:
        if session_engine is None or session_id is None:
            return _failure_result(state, "set_pipeline source.inline_blob requires session context.")
        quota_error = _persist_prepared_blob_create(
            prepared_inline_blob,
            session_engine=session_engine,
            session_id=session_id,
            max_blob_storage_per_session_bytes=max_blob_storage_per_session_bytes,
        )
        if quota_error is not None:
            return _failure_result(state, quota_error)

    # 6. Report all nodes + source + outputs as affected
    affected = (*(_source_component_id(name) for name in source_specs), *(n.id for n in node_specs), *(o.name for o in output_specs))
    data: dict[str, Any] | None = _vf_destination_note(new_state, single_source_on_vf) if single_source_on_vf is not None else None
    if resolved_source_blob is not None:
        source_blob_payload = {"source_blob": resolved_source_blob.payload}
        data = source_blob_payload if data is None else {**data, **source_blob_payload}
    if prepared_inline_blob is not None:
        inline_payload = {"inline_blob": _blob_create_payload(prepared_inline_blob)}
        data = inline_payload if data is None else {**data, **inline_payload}
    return _mutation_result(
        new_state,
        affected,
        data=data,
    )


def _handle_set_pipeline(
    arguments: dict[str, Any],
    state: CompositionState,
    catalog: CatalogService,
    data_dir: str | None = None,
) -> ToolResult:
    return _execute_set_pipeline(arguments, state, catalog, data_dir)


# --- Merge-patch mutation handlers ---


def _execute_patch_source_options(
    args: dict[str, Any],
    state: CompositionState,
    data_dir: str | None = None,
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
    current_source = state.sources.get(source_name)
    if current_source is None:
        return _failure_result(state, f"No source named '{source_name}' configured to patch.")
    patch = validated.patch

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
    path_error = _validate_source_path(new_options, data_dir)
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

    new_source = SourceSpec(
        plugin=current_source.plugin,
        options=new_options,
        on_success=current_source.on_success,
        on_validation_failure=current_source.on_validation_failure,
    )
    new_state = state.with_named_source(source_name, new_source)
    affected = (_source_component_id(source_name),)
    return _mutation_result(new_state, affected)


def _handle_patch_source_options(
    arguments: dict[str, Any],
    state: CompositionState,
    catalog: CatalogService,
    data_dir: str | None = None,
) -> ToolResult:
    result = _execute_patch_source_options(arguments, state, data_dir)
    source_name = arguments.get("source_name", "source")
    source = result.updated_state.sources.get(source_name) if isinstance(source_name, str) else None
    if source is None:
        return result
    return _attach_post_call_hints(
        result,
        catalog,
        plugin_type="source",
        tool_name="patch_source_options",
        plugin_name=source.plugin,
        config_snapshot=source.options,
    )


def _node_routing_option_patch_error(patch: Mapping[str, Any]) -> str | None:
    """Return guidance when plugin-option patches contain node routing fields."""
    if not (_NODE_ROUTING_OPTION_PATCH_KEYS & patch.keys()):
        return None
    for key in ("on_error", "on_success", "input", "routes", "fork_to"):
        if key not in patch:
            continue
        if key == "on_error":
            return (
                "on_error is a node-level routing field, not a plugin option. "
                "Use upsert_edge with edge_type='on_error' when routing failures to an existing sink, "
                "or use upsert_node with on_error as a sibling of options for other routing edits."
            )
        if key == "on_success":
            return (
                "on_success is a node-level routing field, not a plugin option. "
                "Use upsert_edge with edge_type='on_success' when routing success rows to an existing sink, "
                "or use upsert_node with on_success as a sibling of options for other routing edits."
            )
        if key == "input":
            return (
                "input is a node-level connection field, not a plugin option. "
                "Use upsert_node with input as a sibling of options to change the connection this node consumes."
            )
        if key in {"routes", "fork_to"}:
            return (
                f"{key} is a gate-level routing field, not a plugin option. "
                "Use upsert_edge with edge_type='route_true', edge_type='route_false', or edge_type='fork' "
                f"for sink routing, or use upsert_node with {key} as a sibling of options."
            )
    return None


def _execute_patch_node_options(
    args: dict[str, Any],
    state: CompositionState,
) -> ToolResult:
    """Apply a merge-patch to a node's plugin options.

    Tier-3 boundary: ``args`` is an LLM-supplied dict.  Validated via the
    Pydantic redaction-bearing model :class:`PatchNodeOptionsArgumentsModel`
    (the single source of truth for the argument schema — supersedes the
    deleted ``_TOOL_REQUIRED_PATHS["patch_node_options"]`` entry in
    ``service.py``, rev-3 N7 / rev-4 M1).

    On :class:`pydantic.ValidationError` the handler re-raises as
    :class:`ToolArgumentError` so the compose loop's ARG_ERROR routing at
    ``service.py:2480`` receives the right exception class.

    Routing-key guard: :func:`_node_routing_option_patch_error` rejects
    routing-field keys in ``patch`` (on_error, on_success, input, routes,
    fork_to).  This is a value-domain check that Pydantic cannot express;
    it runs AFTER Pydantic validation — same discipline as
    ``set_pipeline``'s blob_id/inline_blob mutual-exclusion check.
    """
    try:
        validated = PatchNodeOptionsArgumentsModel.model_validate(args)
    except PydanticValidationError as exc:
        raise ToolArgumentError(
            argument="patch_node_options arguments",
            expected="object conforming to PatchNodeOptionsArgumentsModel",
            actual_type=type(exc).__name__,
        ) from exc
    node_id = validated.node_id
    patch = validated.patch
    current = next((n for n in state.nodes if n.id == node_id), None)
    if current is None:
        return _failure_result(state, f"Node '{node_id}' not found.")
    routing_patch_error = _node_routing_option_patch_error(patch)
    if routing_patch_error is not None:
        return _failure_result(state, routing_patch_error)
    new_options = _apply_merge_patch(current.options, patch)
    credential_error = _credential_wiring_contract_failure(
        state,
        component_id=node_id,
        component_type="node",
        options=new_options,
    )
    if credential_error is not None:
        return credential_error

    if current.node_type in ("transform", "aggregation") and current.plugin is not None:
        prevalidation_error = _prevalidate_transform(current.plugin, new_options)
        if prevalidation_error is not None:
            return _failure_result(state, prevalidation_error)

    new_node = NodeSpec(
        id=current.id,
        node_type=current.node_type,
        plugin=current.plugin,
        input=current.input,
        on_success=current.on_success,
        on_error=current.on_error,
        options=new_options,
        condition=current.condition,
        routes=current.routes,
        fork_to=current.fork_to,
        branches=current.branches,
        policy=current.policy,
        merge=current.merge,
        trigger=current.trigger,
        output_mode=current.output_mode,
        expected_output_count=current.expected_output_count,
    )
    new_state = state.with_node(new_node)
    return _mutation_result(new_state, (node_id,))


def _handle_patch_node_options(
    arguments: dict[str, Any],
    state: CompositionState,
    catalog: CatalogService,
    data_dir: str | None = None,
) -> ToolResult:
    result = _execute_patch_node_options(arguments, state)
    if not result.success:
        return result
    try:
        validated = PatchNodeOptionsArgumentsModel.model_validate(arguments)
    except PydanticValidationError:
        return result
    node_id = validated.node_id
    node = next((n for n in result.updated_state.nodes if n.id == node_id), None)
    if node is None:
        return result
    return _attach_post_call_hints(
        result,
        catalog,
        plugin_type="transform",
        tool_name="patch_node_options",
        plugin_name=node.plugin,
        config_snapshot=node.options,
    )


def _execute_patch_output_options(
    args: dict[str, Any],
    state: CompositionState,
    data_dir: str | None = None,
) -> ToolResult:
    """Apply a merge-patch to an output's plugin options.

    Tier-3 boundary: ``args`` is an LLM-supplied dict.  Validated via the
    Pydantic redaction-bearing model :class:`PatchOutputOptionsArgumentsModel`
    (the single source of truth for the argument schema — supersedes the
    deleted ``_TOOL_REQUIRED_PATHS["patch_output_options"]`` entry in
    ``service.py``, rev-3 N7 / rev-4 M1).

    On :class:`pydantic.ValidationError` the handler re-raises as
    :class:`ToolArgumentError` so the compose loop's ARG_ERROR routing at
    ``service.py:2480`` receives the right exception class.
    """
    try:
        validated = PatchOutputOptionsArgumentsModel.model_validate(args)
    except PydanticValidationError as exc:
        raise ToolArgumentError(
            argument="patch_output_options arguments",
            expected="object conforming to PatchOutputOptionsArgumentsModel",
            actual_type=type(exc).__name__,
        ) from exc
    sink_name = validated.sink_name
    patch = validated.patch
    current = next((o for o in state.outputs if o.name == sink_name), None)
    if current is None:
        return _failure_result(state, f"Output '{sink_name}' not found.")
    new_options = _apply_merge_patch(current.options, patch)
    credential_error = _credential_wiring_contract_failure(
        state,
        component_id=sink_name,
        component_type="output",
        options=new_options,
    )
    if credential_error is not None:
        return credential_error

    # S2: Validate patched sink paths against allowlist
    path_error = _validate_sink_path(new_options, data_dir)
    if path_error is not None:
        return _failure_result(state, path_error)

    prevalidation_error = _prevalidate_sink(current.plugin, new_options)
    if prevalidation_error is not None:
        return _failure_result(state, prevalidation_error)
    collision_error = validate_composer_file_sink_collision_policy(
        current.plugin,
        new_options,
        require_explicit=data_dir is not None,
    )
    if collision_error is not None:
        return _failure_result(state, collision_error)

    new_output = OutputSpec(
        name=current.name,
        plugin=current.plugin,
        options=new_options,
        on_write_failure=current.on_write_failure,
    )
    new_state = state.with_output(new_output)
    return _mutation_result(new_state, (sink_name,))


def _handle_patch_output_options(
    arguments: dict[str, Any],
    state: CompositionState,
    catalog: CatalogService,
    data_dir: str | None = None,
) -> ToolResult:
    result = _execute_patch_output_options(arguments, state, data_dir)
    if not result.success:
        return result
    try:
        validated = PatchOutputOptionsArgumentsModel.model_validate(arguments)
    except PydanticValidationError:
        return result
    sink_name = validated.sink_name
    output = next((o for o in result.updated_state.outputs if o.name == sink_name), None)
    if output is None:
        return result
    return _attach_post_call_hints(
        result,
        catalog,
        plugin_type="sink",
        tool_name="patch_output_options",
        plugin_name=output.plugin,
        config_snapshot=output.options,
    )


# --- Source-reset and validation-explanation handlers ---


def _execute_clear_source(
    args: dict[str, Any],
    state: CompositionState,
) -> ToolResult:
    """Remove the pipeline source."""
    if not state.sources:
        return _failure_result(state, "No source configured to clear.")
    source_name = args.get("source_name")
    if source_name is None:
        new_state = state.without_source()
        return _mutation_result(new_state, ("source",))
    if type(source_name) is not str or not source_name.strip():
        return _failure_result(state, "source_name must be a non-empty string when provided.")
    new_named_state = state.without_named_source(source_name)
    if new_named_state is None:
        return _failure_result(state, f"No source named '{source_name}' configured to clear.")
    affected = (_source_component_id(source_name),)
    return _mutation_result(new_named_state, affected)


def _handle_clear_source(
    arguments: dict[str, Any],
    state: CompositionState,
    catalog: CatalogService,
    data_dir: str | None = None,
) -> ToolResult:
    return _execute_clear_source(arguments, state)


# Validation error pattern catalogue for explain_validation_error.
# Each entry: (regex pattern, explanation, suggested fix).
_VALIDATION_ERROR_PATTERNS: list[tuple[str, str, str]] = [
    (
        r"No source configured",
        "The pipeline has no data source. Every pipeline needs at least one named source to read input data from.",
        "Use set_source to configure a source plugin (e.g. csv, json, dataverse).",
    ),
    (
        r"No sinks configured",
        "The pipeline has no outputs. At least one sink is needed to write results.",
        "Use set_output to add an output (e.g. csv, json).",
    ),
    (
        r"references unknown node '(.+)' as from_node",
        "An edge references a node that doesn't exist in the pipeline as its source.",
        "Check the edge's from_node value. Either add the missing node with upsert_node or fix the edge with upsert_edge.",
    ),
    (
        r"references unknown node '(.+)' as to_node",
        "An edge references a node or output that doesn't exist in the pipeline as its target.",
        "Check the edge's to_node value. Either add the missing node/output or fix the edge.",
    ),
    (
        r"Duplicate node ID: '(.+)'",
        "Two nodes have the same ID. Each node must have a unique identifier.",
        "Rename one of the duplicate nodes using upsert_node with a different id.",
    ),
    (
        r"Duplicate output name: '(.+)'",
        "Two outputs have the same name. Each output must have a unique name.",
        "Rename one of the duplicate outputs using set_output with a different sink_name.",
    ),
    (
        r"Duplicate edge ID: '(.+)'",
        "Two edges have the same ID. Each edge must have a unique identifier.",
        "Remove the duplicate edge with remove_edge and re-add with a unique id.",
    ),
    (
        r"Gate '(.+)' is missing required field '(.+)'",
        "A gate node is missing a required configuration field (condition or routes).",
        "Update the gate with upsert_node, providing the missing field.",
    ),
    (
        r"Transform '(.+)' must not have '(.+)' field",
        "A transform node has a field that only gates should have (condition or routes).",
        "Update the node with upsert_node. Set node_type to 'gate' if routing is needed, or remove the field.",
    ),
    (
        r"Coalesce '(.+)' is missing required field '(.+)'",
        "A coalesce node is missing a required field (branches or policy).",
        "Update the coalesce node with upsert_node, providing the missing field.",
    ),
    (
        r"Aggregation '(.+)' is missing required field 'plugin'",
        "An aggregation node needs a plugin to define its aggregation behaviour.",
        "Update the aggregation with upsert_node, specifying the plugin name.",
    ),
    (
        r"input '(.+)' is not reachable",
        "A node's input connection point is not produced by the runtime routing fields.",
        "Set source.on_success or an upstream node's on_success/on_error/route/fork_to so it matches this node's input.",
    ),
    (
        r"Unknown .+ plugin '(.+)'",
        "The specified plugin name is not available in the catalog.",
        "Use list_sources, list_transforms, or list_sinks to see available plugins.",
    ),
    (
        r"Path violation \(S2\).*[Ss]ource",
        "The source file path is outside the allowed directories.",
        "Source paths must be under the blobs/ directory. Upload a file first or use set_source_from_blob.",
    ),
    (
        r"Path violation \(S2\).*[Ss]ink",
        "The sink output path is outside the allowed directories.",
        "Sink output paths must be under the outputs/ or blobs/ directory.",
    ),
    (
        r"Path violation \(S2\)",
        "A file path is outside the allowed directories.",
        "Source paths must be under the blobs/ directory. Sink output paths must be under the outputs/ or blobs/ directory.",
    ),
    (
        r"Invalid options for source '(.+)':",
        "The source plugin configuration is invalid. A required option may be missing or have an invalid value.",
        "Use get_pipeline_state with component='source' to see current options, then use patch_source_options to fix.",
    ),
    (
        r"Invalid options for transform '(.+)':",
        "A transform node has invalid configuration. A required option may be missing or have an invalid value.",
        "Use get_pipeline_state to see the node's current options, then use patch_node_options to fix.",
    ),
    (
        r"Invalid options for sink '(.+)':",
        "A sink output has invalid configuration. A required option may be missing (e.g. path for file-based sinks).",
        "Use get_pipeline_state to see the output's current options, then use patch_output_options to fix.",
    ),
    (
        r"Schema contract violation: '.*' -> 'output:[^']+'",
        "A sink schema requires fields that its upstream producer does not guarantee.",
        "Call preview_pipeline to inspect edge_contracts, then either relax the sink schema with patch_output_options or update the upstream schema with patch_source_options or patch_node_options and re-preview until the edge shows satisfied=true.",
    ),
    (
        r"Schema contract violation:",
        "A downstream node requires fields that its upstream producer does not guarantee.",
        "Call preview_pipeline to inspect edge_contracts, then update the upstream schema with patch_source_options or patch_node_options and re-preview until the edge shows satisfied=true.",
    ),
]


def _extract_validator_expected_hint(error_text: str) -> str | None:
    """Pull the ``Expected ...`` span out of a validator error string.

    Pydantic and our schema-spec validators frequently emit errors like
    ``"Field spec at index 0 is a dict with 2 keys. Expected single-key
    dict like {'field_name': 'type'} or a string like 'field_name: type'."``.
    The static catalogue fix below the substring discards that hint, so
    the model only sees ``"Use get_pipeline_state ... patch_source_options"``
    — which doesn't tell it what shape to actually emit. Returning the
    ``Expected ...`` span verbatim lets the caller append it to
    ``suggested_fix`` so the model can copy the shape directly.

    The hint terminates at the next sentence boundary (``.`` followed by
    whitespace or end-of-string) so a trailing ``"Got X. Other noise."``
    doesn't get swept up.
    """
    idx = error_text.find("Expected ")
    if idx == -1:
        return None
    rest = error_text[idx:]
    end = len(rest)
    for i, ch in enumerate(rest):
        if ch == "." and (i + 1 == len(rest) or rest[i + 1].isspace()):
            end = i + 1
            break
    return rest[:end].strip()


def _augment_with_expected_hint(fix: str, error_text: str) -> str:
    """Append the validator ``Expected ...`` hint to ``fix`` when present."""
    hint = _extract_validator_expected_hint(error_text)
    if hint is None:
        return fix
    return f"{fix} {hint}"


def _execute_explain_validation_error(
    args: dict[str, Any],
    state: CompositionState,
    catalog: CatalogService,
    data_dir: str | None = None,
) -> ToolResult:
    """Explain a validation error with human-readable diagnosis and fix."""
    error_text = args["error_text"]
    for pattern, explanation, fix in _VALIDATION_ERROR_PATTERNS:
        if re.search(pattern, error_text):
            return ToolResult(
                success=True,
                updated_state=state,
                validation=state.validate(),
                affected_nodes=(),
                data={
                    "error_text": error_text,
                    "explanation": explanation,
                    "suggested_fix": _augment_with_expected_hint(fix, error_text),
                },
            )
    # No match — return a generic response
    return ToolResult(
        success=True,
        updated_state=state,
        validation=state.validate(),
        affected_nodes=(),
        data={
            "error_text": error_text,
            "explanation": "This error is not in the known pattern catalogue.",
            "suggested_fix": _augment_with_expected_hint(
                "Review the error message and the pipeline structure. Use get_pipeline_state to inspect the current composition.",
                error_text,
            ),
        },
    )


def _serialize_plugin_assistance_example(
    example: Any,
) -> dict[str, Any]:
    """Serialize a PluginAssistanceExample for LLM consumption.

    Mirrors the serialize-at-the-boundary pattern used by
    ``_semantic_edge_contract_to_payload`` (composer_mcp/server.py) and
    ``serialize_semantic_contracts`` (web/execution/_semantic_helpers.py):
    L0 contract types intentionally have no ``.to_dict()``; the rendering
    site owns the JSON shape so contracts stay free of encoding concerns.
    """
    return {
        "title": example.title,
        "before": deep_thaw(example.before) if example.before is not None else None,
        "after": deep_thaw(example.after) if example.after is not None else None,
    }


def _execute_get_plugin_assistance(
    args: dict[str, Any],
    state: CompositionState,
    catalog: CatalogService,
    data_dir: str | None = None,
) -> ToolResult:
    """Return plugin-owned guidance for a source, transform, or sink.

    Dual-use by ``issue_code``:

    * ``issue_code is None`` (or absent) — discovery-time guidance. The
      plugin returns a one-line ``summary`` and ``composer_hints``
      (same surface that list_* and get_plugin_schema already carry).
    * ``issue_code is not None`` — failure-time guidance. The
      semantic validator emits ``requirement_code`` values like
      ``line_explode.source_field.line_framed_text``; the agent echoes
      that code in to retrieve ``suggested_fixes`` + example
      before/after configs.

    When the plugin has no assistance to publish, returns success with
    a "no assistance published" payload (summary=None, empty lists)
    rather than failing — the absence is itself a useful signal.

    Unknown plugin name or invalid plugin_type surfaces here as a tool
    failure with the original message so the agent can correct the call.
    """
    from elspeth.plugins.infrastructure.manager import (
        PluginNotFoundError,
        get_shared_plugin_manager,
    )

    plugin_type_raw = args["plugin_type"]
    plugin_name = args["plugin_name"]
    issue_code = args.get("issue_code")

    if plugin_type_raw not in ("source", "transform", "sink"):
        return _failure_result(
            state,
            f"Unknown plugin_type: {plugin_type_raw!r}. Must be one of: 'source', 'transform', 'sink'.",
        )
    plugin_type: PluginKind = plugin_type_raw

    manager = get_shared_plugin_manager()
    try:
        if plugin_type == "source":
            plugin_cls: Any = manager.get_source_by_name(plugin_name)
        elif plugin_type == "transform":
            plugin_cls = manager.get_transform_by_name(plugin_name)
        else:
            plugin_cls = manager.get_sink_by_name(plugin_name)
    except PluginNotFoundError as exc:
        return _failure_result(state, str(exc))

    assistance = plugin_cls.get_agent_assistance(issue_code=issue_code)

    if assistance is None:
        payload: dict[str, Any] = {
            "plugin_type": plugin_type,
            "plugin_name": plugin_name,
            "issue_code": issue_code,
            "summary": None,
            "suggested_fixes": [],
            "examples": [],
            "composer_hints": [],
        }
        return _discovery_result(state, payload)

    payload = {
        "plugin_type": plugin_type,
        "plugin_name": assistance.plugin_name,
        "issue_code": assistance.issue_code,
        "summary": assistance.summary,
        "suggested_fixes": list(assistance.suggested_fixes),
        "examples": [_serialize_plugin_assistance_example(ex) for ex in assistance.examples],
        "composer_hints": list(assistance.composer_hints),
    }
    return _discovery_result(state, payload)


def _execute_get_audit_info(
    args: dict[str, Any],
    state: CompositionState,
    catalog: CatalogService,
    data_dir: str | None = None,
) -> ToolResult:
    """Return constant facts about the Landscape audit trail.

    Audit is mandatory (`LandscapeSettings` rejects `enabled=false` at
    config validation time) and the backend URL is operator-managed via
    `WebSettings.get_landscape_url()` — security fix S1, see
    `web/composer/yaml_generator.py:179`. Letting the composer set the
    audit backend would let a user prompt redirect the audit trail to an
    attacker DB, disable encryption, or split audit across stores.

    The returned payload is a constant — no `WebSettings` access — so
    operator-internal config (URL, backend type, encryption-key env var)
    never reaches the LLM context. The model paraphrases `summary`; it
    does not need the URL itself.
    """
    payload = {
        "enabled": True,
        "composer_modifiable": False,
        "summary": (
            "Landscape audit is mandatory and always on for every pipeline run. "
            "The audit backend (database type, location, encryption) is configured "
            "by the operator at deploy time and is intentionally NOT addressable "
            "from the composer — letting the composer set it would be a security "
            "regression (audit-DSN injection, encryption bypass, audit split-brain). "
            "When a user asks for 'audit logging', 'SQLite audit', or similar: "
            "acknowledge that audit is already enabled for every run, do NOT add a "
            "sink shape for it, and do NOT silently remove an audit-shaped node by "
            "treating it as 'unconnected'. To inspect past runs, point the user at "
            "the Landscape MCP forensic tools."
        ),
        "audit_export_summary": (
            "A separate optional feature ('landscape.export') can copy each run's "
            "audit data to an additional sink for offline review. This is also "
            "operator-configured and is not currently composer-controllable. If a "
            "user asks for 'export the audit data to a file', explain that this is "
            "an operator-side configuration and is not part of the pipeline the "
            "composer is building."
        ),
    }
    return _discovery_result(state, payload)


def _execute_list_models(
    args: dict[str, Any],
    state: CompositionState,
    catalog: CatalogService,
    data_dir: str | None = None,
) -> ToolResult:
    """List available LLM model identifiers.

    Without a provider filter, returns provider names and model counts
    to avoid dumping hundreds of entries. With a provider filter,
    returns matching model IDs capped at ``limit``.

    Reads via :func:`read_litellm_model_list` so this tool and the
    value-source compliance walker share a single source of truth for
    what counts as "a known model."
    """
    from elspeth.plugins.transforms.llm.model_catalog import read_litellm_model_list

    all_models: list[str] = list(read_litellm_model_list())

    provider = args.get("provider")
    limit = args.get("limit", 50)
    if not isinstance(limit, int) or limit < 1:
        limit = 50

    if provider is not None and isinstance(provider, str):
        if provider == "":
            # Empty string means "models without a provider prefix"
            filtered = [m for m in all_models if "/" not in m]
        else:
            filtered = [m for m in all_models if m.startswith(provider)]
        # OpenRouter consumers (the actual HTTP API at /chat/completions)
        # expect un-prefixed slugs (e.g. ``openai/gpt-4o``, not
        # ``openrouter/openai/gpt-4o``). The litellm representation carries
        # an ``openrouter/`` routing prefix that ELSPETH's
        # OpenRouterLLMProvider does not strip — so the tool returns the
        # un-prefixed form to match what the user must put in their YAML
        # and what the value-source compliance validator accepts. The
        # OPENROUTER_LITELLM_PREFIX constant lives next to the catalog
        # reader so both sites strip identically.
        from elspeth.plugins.transforms.llm.model_catalog import OPENROUTER_LITELLM_PREFIX

        normalised = provider.rstrip("/")
        if normalised == OPENROUTER_LITELLM_PREFIX.rstrip("/"):
            prefix_len = len(OPENROUTER_LITELLM_PREFIX)
            filtered = [m[prefix_len:] for m in filtered]
        truncated = len(filtered) > limit
        data: dict[str, Any] = {
            "models": filtered[:limit],
            "count": len(filtered),
            "truncated": truncated,
        }
    else:
        # Group by provider prefix to avoid token waste
        providers: dict[str, int] = {}
        for m in all_models:
            prefix = m.split("/", 1)[0] if "/" in m else ""
            providers[prefix] = providers.get(prefix, 0) + 1
        data = {
            "providers": providers,
            "total_models": len(all_models),
            "hint": "Use provider parameter to list models for a specific provider. An empty string key means models without a provider prefix.",
        }

    return _discovery_result(state, data)


def _serialize_source(source: SourceSpec) -> dict[str, Any]:
    """Serialize a SourceSpec to a plain dict for LLM consumption."""
    return {
        "plugin": source.plugin,
        "on_success": source.on_success,
        "options": deep_thaw(source.options),
        "on_validation_failure": source.on_validation_failure,
    }


def _serialize_node(node: NodeSpec) -> dict[str, Any]:
    """Serialize a NodeSpec to a plain dict for LLM consumption.

    Includes all fields (even None) so the LLM sees the full schema.
    """
    return {
        "id": node.id,
        "node_type": node.node_type,
        "plugin": node.plugin,
        "input": node.input,
        "on_success": node.on_success,
        "on_error": node.on_error,
        "options": deep_thaw(node.options),
        "condition": node.condition,
        "routes": deep_thaw(node.routes) if node.routes else None,
        "fork_to": list(node.fork_to) if node.fork_to else None,
        "branches": _serialize_branches(node.branches) if node.branches else None,
        "policy": node.policy,
        "merge": node.merge,
        "trigger": deep_thaw(node.trigger) if node.trigger else None,
        "output_mode": node.output_mode,
        "expected_output_count": node.expected_output_count,
    }


def _serialize_output(output: OutputSpec) -> dict[str, Any]:
    """Serialize an OutputSpec to a plain dict for LLM consumption."""
    return {
        "sink_name": output.name,
        "plugin": output.plugin,
        "options": deep_thaw(output.options),
        "on_write_failure": output.on_write_failure,
    }


def _serialize_edge(edge: EdgeSpec) -> dict[str, Any]:
    """Serialize an EdgeSpec to a plain dict for LLM consumption."""
    return {
        "id": edge.id,
        "from_node": edge.from_node,
        "to_node": edge.to_node,
        "edge_type": edge.edge_type,
        "label": edge.label,
    }


def _is_full_state_component_alias(component: Any) -> bool:
    """Return whether a component argument explicitly requests full state."""
    return isinstance(component, str) and component.strip().lower() in _FULL_STATE_COMPONENT_ALIAS_SET


def _serialize_full_pipeline_state(state: CompositionState, *, requested_component: Any) -> _FullPipelineStatePayload:
    """Serialize the full state and expose accepted full-state spellings."""
    return {
        "source": _serialize_source(state.source) if state.source is not None else None,
        "sources": {name: _serialize_source(source) for name, source in state.sources.items()},
        "nodes": [_serialize_node(n) for n in state.nodes],
        "outputs": [_serialize_output(o) for o in state.outputs],
        "edges": [_serialize_edge(e) for e in state.edges],
        "metadata": {"name": state.metadata.name, "description": state.metadata.description},
        "version": state.version,
        "inspection": {
            "requested_component": requested_component,
            "resolved_component": "full",
            "accepted_full_state_aliases": list(_FULL_STATE_COMPONENT_ALIASES),
        },
    }


def _execute_get_pipeline_state(
    args: dict[str, Any],
    state: CompositionState,
    catalog: CatalogService,
    data_dir: str | None = None,
) -> ToolResult:
    """Return full pipeline state including all options.

    If ``component`` is specified, returns only that component's details.
    Otherwise returns the full state: source, all nodes with options, all
    outputs with options, edges, and metadata.
    """
    component = args.get("component")

    if component == "source":
        data: Any = {
            "source": _serialize_source(state.source) if state.source is not None else None,
            "sources": {name: _serialize_source(source) for name, source in state.sources.items()},
        }
    elif component is not None:
        # Try node, then output
        node = next((n for n in state.nodes if n.id == component), None)
        if node is not None:
            data = {"node": _serialize_node(node)}
        else:
            output = next((o for o in state.outputs if o.name == component), None)
            if output is not None:
                data = {"output": _serialize_output(output)}
            elif _is_full_state_component_alias(component):
                data = _serialize_full_pipeline_state(state, requested_component=component)
            else:
                return _failure_result(
                    state,
                    f"Component '{component}' not found. Specify 'source', a node ID, an output name, "
                    "or a full-state alias ('full', 'all', 'pipeline', or empty string).",
                )
    else:
        data = _serialize_full_pipeline_state(state, requested_component=None)

    data = redact_source_storage_path(data)
    return _discovery_result(state, data)


def _authoring_validation_payload(state: CompositionState, validation: ValidationSummary) -> dict[str, Any]:
    return {
        "is_valid": validation.is_valid,
        "errors": [e.to_dict() for e in validation.errors],
        "warnings": [e.to_dict() for e in validation.warnings],
        "suggestions": [e.to_dict() for e in validation.suggestions],
        "edge_contracts": [ec.to_dict() for ec in validation.edge_contracts],
        "semantic_contracts": _semantic_contracts_payload(validation.semantic_contracts),
        "graph_repair_suggestions": _graph_repair_suggestions(state, validation),
    }


# Canonical registry of every diagnostic code ``compute_proof_diagnostics``
# may emit at ``severity='blocking'``. The skill markdown that drives the
# composer LLM cites these codes by name, and the forced-repair loop keys off
# ``severity == 'blocking'`` to decide whether to inject a repair message.
# Drift between this set and the actual emission sites would silently break
# the LLM's expected vocabulary, so ``_blocking_diagnostic`` enforces that
# every blocking dict's ``code`` appears here at construction time.
_BLOCKING_DIAGNOSTIC_CODES: Final[frozenset[str]] = frozenset(
    {
        "aggregation_numeric_value_field_type_mismatch_against_source_schema",
        "csv_duplicate_headers",
        "csv_fixed_schema_omits_observed_columns",
        "gate_expression_type_mismatch_against_source_schema",
        "text_source_url_without_web_scrape",
        "source_inspection_failed",
    }
)


def _blocking_diagnostic(
    *,
    code: str,
    message: str,
    suggested_repair: str,
    evidence_locator: dict[str, Any],
) -> dict[str, Any]:
    """Construct a blocking diagnostic dict and assert the code is registered.

    Offensive: a contributor who adds a new blocker without registering it in
    ``_BLOCKING_DIAGNOSTIC_CODES`` (and the matching skill-markdown vocabulary)
    crashes immediately rather than shipping an unrecognised code into the
    audit trail and the LLM's repair-message context.
    """
    if code not in _BLOCKING_DIAGNOSTIC_CODES:
        raise AssertionError(
            f"blocking diagnostic code {code!r} is not registered in "
            f"_BLOCKING_DIAGNOSTIC_CODES. Add it there (and to the skill-"
            f"markdown vocabulary the composer LLM consumes) before emitting "
            f"a blocking diagnostic with this code."
        )
    return {
        "code": code,
        "severity": "blocking",
        "message": message,
        "suggested_repair": suggested_repair,
        "evidence_locator": evidence_locator,
    }


def _source_schema_mode(source: SourceSpec) -> str | None:
    schema = source.options.get("schema")
    if not isinstance(schema, Mapping):
        return None
    mode = schema.get("mode")
    if not isinstance(mode, str):
        return None
    return mode.strip().lower()


def _sample_csv_rows(content: bytes, *, filename: str, max_rows: int = 100) -> tuple[dict[str, str], ...]:
    text = content[: 8 * 1024].decode("utf-8", errors="replace")
    delimiter = "\t" if filename.lower().endswith(".tsv") else ","
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    rows: list[dict[str, str]] = []
    for index, row in enumerate(reader):
        if index >= max_rows:
            break
        rows.append({key: value for key, value in row.items() if isinstance(key, str) and value is not None})
    return tuple(rows)


def _row_fields_referenced_by_condition(condition: str) -> tuple[str, ...]:
    tree = ast.parse(condition, mode="eval")
    fields: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == "row"
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        ):
            fields.append(node.slice.value)
            continue
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "row"
            and len(node.args) == 1
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            fields.append(node.args[0].value)
    return tuple(dict.fromkeys(fields))


def _gate_expression_type_diagnostics_for_observed_csv(
    state: CompositionState,
    source: SourceSpec,
    *,
    blob_id: str,
    filename: str,
    content: bytes,
) -> list[dict[str, Any]]:
    """Evaluate direct source-fed gates against sampled observed CSV rows.

    Observed CSV sources emit raw strings because there are no declared field
    types to coerce against. A gate such as ``row['amount'] >= 1000`` is
    syntactically valid but fails at runtime when the evaluator compares
    ``str`` with ``int``. This preview proof step uses the same expression
    evaluator against bounded raw rows and reports the type mismatch without
    surfacing row values.
    """
    if _source_schema_mode(source) != "observed":
        return []

    rows = _sample_csv_rows(content, filename=filename)
    if not rows:
        return []

    from elspeth.core.expression_parser import ExpressionEvaluationError, ExpressionParser

    diagnostics: list[dict[str, Any]] = []
    direct_gate_nodes = (
        node for node in state.nodes if node.node_type == "gate" and node.input == source.on_success and node.condition is not None
    )
    for node in direct_gate_nodes:
        condition = node.condition
        if condition is None:
            continue
        if _validate_gate_expression(condition) is not None:
            continue
        parser = ExpressionParser(condition)
        fields = _row_fields_referenced_by_condition(condition)
        field = fields[0] if fields else None
        for row_index, row in enumerate(rows):
            try:
                parser.evaluate(row)
            except ExpressionEvaluationError as exc:
                diagnostics.append(
                    _blocking_diagnostic(
                        code="gate_expression_type_mismatch_against_source_schema",
                        message=(
                            f"Gate '{node.id}' condition {condition!r} fails against sampled observed CSV "
                            f"rows before runtime: {exc}. Observed CSV source values are strings unless the "
                            "source schema declares explicit field types."
                        ),
                        suggested_repair=(
                            "Patch the source schema to declare the compared field with an explicit numeric "
                            "type, for example schema.mode='fixed' or 'flexible' with schema.fields including "
                            f"{field + ': int' if field is not None else '<field>: int'}, then re-run preview_pipeline."
                        ),
                        evidence_locator={
                            "source": "blob",
                            "blob_id": str(blob_id),
                            "node_id": node.id,
                            "field": field,
                            "fields": list(fields),
                            "sample_row_index": row_index,
                            "source_schema_mode": "observed",
                        },
                    )
                )
                break
    return diagnostics


_NUMERIC_VALUE_FIELD_AGGREGATION_PLUGINS: Final[frozenset[str]] = frozenset(
    {
        "batch_distribution_profile",
        "batch_outlier_annotator",
        "batch_stats",
        "batch_threshold_summary",
    }
)


def _value_transform_preserves_field(node: NodeSpec, field_name: str) -> bool:
    operations = node.options.get("operations")
    if not isinstance(operations, (list, tuple)):
        return False
    for operation in operations:
        if not isinstance(operation, Mapping):
            return False
        target = operation.get("target")
        if target == field_name:
            return False
    return True


def _source_field_reaches_connection_without_type_change(
    state: CompositionState,
    connection_name: str,
    *,
    field_name: str,
) -> bool:
    """Return True when a source field flows to a connection unchanged.

    This intentionally recognises only field-preserving nodes. Unknown
    transforms may coerce, overwrite, delete, or synthesize the field, so the
    proof step abstains instead of emitting a false positive.
    """
    from elspeth.web.composer._producer_resolver import ProducerResolver

    resolver = ProducerResolver.build(
        source=state.source,
        nodes=state.nodes,
        sink_names=frozenset(output.name for output in state.outputs),
    )
    current = connection_name
    visited: set[str] = set()
    while True:
        if current in visited:
            return False
        visited.add(current)

        producer = resolver.find_producer_for(current)
        if producer is None:
            return False
        if producer.producer_id == "source":
            return True

        node = resolver.get_node(producer.producer_id)
        if node is None:
            return False
        if node.node_type == "gate":
            current = node.input
            continue
        if node.plugin == "value_transform" and _value_transform_preserves_field(node, field_name):
            current = node.input
            continue
        if node.plugin == "passthrough":
            current = node.input
            continue
        return False


def _numeric_aggregation_diagnostics_for_observed_csv(
    state: CompositionState,
    source: SourceSpec,
    *,
    blob_id: str,
    inferred_types: Mapping[str, str] | None,
    observed_headers: tuple[str, ...] | None,
) -> list[dict[str, Any]]:
    """Block observed CSV strings before numeric aggregation runtime failures."""
    if _source_schema_mode(source) != "observed" or observed_headers is None:
        return []

    observed_header_set = set(observed_headers)
    diagnostics: list[dict[str, Any]] = []
    for node in state.nodes:
        if node.node_type != "aggregation" or node.plugin not in _NUMERIC_VALUE_FIELD_AGGREGATION_PLUGINS:
            continue
        options, _owner = get_aggregation_contract_options(node.options, owner=f"node:{node.id}")
        value_field = options.get("value_field")
        if type(value_field) is not str or not value_field.strip():
            continue
        value_field = value_field.strip()
        if value_field not in observed_header_set:
            continue
        if not _source_field_reaches_connection_without_type_change(state, node.input, field_name=value_field):
            continue

        inferred_type = inferred_types.get(value_field) if inferred_types is not None else None
        diagnostics.append(
            _blocking_diagnostic(
                code="aggregation_numeric_value_field_type_mismatch_against_source_schema",
                message=(
                    f"Aggregation '{node.id}' ({node.plugin}) uses numeric value_field '{value_field}', "
                    "but it is flowing from an observed CSV source. Observed CSV source values are strings "
                    "unless the source schema declares explicit field types or an upstream type_coerce node "
                    "converts the field before aggregation."
                ),
                suggested_repair=(
                    "Patch the source schema to declare the aggregated field with an explicit numeric type "
                    f"(for example {value_field}: float), or insert a type_coerce node upstream of the aggregation. "
                    "If the field is categorical and you want counts/frequencies, use batch_top_k instead of a "
                    "numeric aggregation."
                ),
                evidence_locator={
                    "source": "blob",
                    "blob_id": str(blob_id),
                    "node_id": node.id,
                    "plugin": node.plugin,
                    "field": value_field,
                    "observed_type": "str",
                    "inferred_sample_type": inferred_type or "unknown",
                    "source_runtime_type": "str",
                    "source_schema_mode": "observed",
                },
            )
        )

    return diagnostics


def compute_proof_diagnostics(
    state: CompositionState,
    *,
    session_engine: Engine | None = None,
    session_id: str | None = None,
) -> list[dict[str, Any]]:
    """Compute machine-readable proof diagnostics for a composer state.

    Promotes ``preview_pipeline`` from a "state validates" check into a
    "state is plausibly runnable against observed input" proof. Returns a
    machine-readable list of diagnostics — each entry has::

        {
            "code": "csv_fixed_schema_omits_observed_columns",
            "severity": "blocking" | "warning" | "info",
            "message": "human-readable description",
            "suggested_repair": "tool/options the LLM should call",
            "evidence_locator": {"source": "...", "node_id": "...", ...},
        }

    Diagnostics surfaced:

      * ``csv_fixed_schema_omits_observed_columns`` — fixed CSV schema +
        on_validation_failure=discard + at least one observed column
        absent from declared fields. The combination silently discards
        every row, which is the #1 historical convergence-failure mode.
      * ``text_source_url_without_web_scrape`` — text source whose blob
        content is a single URL but no web_scrape node downstream. The
        URL string itself reaches sinks instead of the URL's content.
      * ``gate_expression_type_mismatch_against_source_schema`` — observed
        CSV source values are still strings, and a direct source-fed gate
        condition fails when evaluated against sampled rows before runtime.
      * ``aggregation_numeric_value_field_type_mismatch_against_source_schema`` —
        observed CSV strings flow unchanged into a numeric aggregation
        ``value_field`` before runtime can reject the batch.
      * ``source_inspection_warning`` — every warning surfaced by
        ``inspect_blob_content`` is mirrored here at ``info`` severity
        so the model sees them in the same array as blocking issues.

    Bounded I/O: at most one blob read per call, bounded by
    ``inspect_blob_content``'s 8 KiB / 100 row caps.

    No-op (returns an empty list) if the source is not blob-backed or
    if session context is absent.
    """
    diagnostics: list[dict[str, Any]] = []

    source = state.source
    if source is None:
        return diagnostics

    # Only blob-backed sources are inspectable from preview_pipeline; for
    # path-based sources we have no bytes to peek at. SourceSpec.options is
    # internally typed as Mapping[str, Any] (Tier-1 dataclass invariant — no
    # isinstance probe needed); see _proof_repair_is_applicable for the
    # canonical pattern this site mirrors.
    blob_id = source.options.get("blob_ref")
    if blob_id is None or session_engine is None or session_id is None:
        return diagnostics

    blob = _sync_get_blob(session_engine, str(blob_id), session_id)
    # ``blob`` is a BlobToolRecord (TypedDict produced by
    # ``_blob_row_to_tool_dict`` from a validated blobs row). Direct
    # subscript access is mandatory — a missing key is a Tier-1
    # contract violation in our own dict shape, not external data.
    if blob is None or blob["status"] != "ready":
        return diagnostics

    storage_path = Path(blob["storage_path"])
    if not storage_path.exists():
        diagnostics.append(
            _blocking_diagnostic(
                code="source_inspection_failed",
                message=(f"Source blob '{blob_id}' storage file is missing — pipeline cannot run until the blob is re-uploaded."),
                suggested_repair="create_blob with the original content and re-wire via set_source_from_blob",
                evidence_locator={"source": "blob", "blob_id": str(blob_id)},
            )
        )
        return diagnostics

    # Tier 1 (our data, our file): an OSError between exists() and
    # read_bytes() is a real anomaly (concurrent delete, fs corruption,
    # permission revocation). Per CLAUDE.md offensive-programming
    # policy, let it propagate so the operator sees an informative
    # exception rather than a synthesised soft-degraded diagnostic
    # that could let downstream act on absent bytes.
    content = storage_path.read_bytes()

    # Tier 1 integrity verification — same shared helper as the two
    # other composer-tool blob readers. Without this, the proof step
    # would feed unverified bytes into ``inspect_blob_content`` and
    # repair-loop, undermining the audit trail's "decisions made on
    # verified inputs" invariant.
    _verify_blob_content_integrity(blob, content)

    facts = inspect_blob_content(
        content=content,
        filename=blob["filename"],
        mime_type=blob["mime_type"],
        content_hash=blob["content_hash"],
    )

    # 1. Fixed CSV schema omits observed columns + discard => silent all-row drop.
    if facts.source_kind in {"csv", "json", "jsonl"}:
        # source.options is Tier-1 (Mapping[str, Any]); the *value* at "schema"
        # is unstructured and may be absent, so the inner shape probes below
        # remain.
        schema = source.options.get("schema")
        if isinstance(schema, Mapping) and schema.get("mode") == "fixed":
            declared = schema.get("fields") or ()
            if isinstance(declared, (list, tuple)):
                missing = derive_extra_column_risk(facts, tuple(declared))
                if missing and source.on_validation_failure == "discard":
                    diagnostics.append(
                        _blocking_diagnostic(
                            code="csv_fixed_schema_omits_observed_columns",
                            message=(
                                f"Source schema is mode=fixed but omits observed columns "
                                f"{list(missing)} (observed: {list(facts.observed_headers or ())}). "
                                "Combined with on_validation_failure='discard', every row will be "
                                "dropped because each contains an undeclared column."
                            ),
                            suggested_repair=(
                                "patch_source_options with schema.mode='flexible' to accept extra "
                                "columns, OR add the missing columns to schema.fields, OR set "
                                "on_validation_failure to a configured output for inspection."
                            ),
                            evidence_locator={
                                "source": "blob",
                                "blob_id": str(blob_id),
                                "missing_columns": list(missing),
                                "observed_columns": list(facts.observed_headers or ()),
                            },
                        )
                    )

    # 2. Observed CSV + numeric gate predicate => preview/runtime agreement gap.
    if facts.source_kind == "csv":
        diagnostics.extend(
            _gate_expression_type_diagnostics_for_observed_csv(
                state,
                source,
                blob_id=str(blob_id),
                filename=blob["filename"],
                content=content,
            )
        )
        diagnostics.extend(
            _numeric_aggregation_diagnostics_for_observed_csv(
                state,
                source,
                blob_id=str(blob_id),
                inferred_types=facts.inferred_types,
                observed_headers=facts.observed_headers,
            )
        )

    # 3. Text source containing a single URL but no web_scrape downstream.
    if facts.source_kind == "text" and facts.url_candidates:
        node_plugins = {(n.plugin or "").lower() for n in state.nodes}
        if "web_scrape" not in node_plugins:
            diagnostics.append(
                _blocking_diagnostic(
                    code="text_source_url_without_web_scrape",
                    message=(
                        f"Source blob contains URL(s) {list(facts.url_candidates)} but no "
                        "web_scrape transform is wired downstream. The URL string itself will "
                        "flow to sinks, not the URL's content."
                    ),
                    suggested_repair=(
                        "upsert_node({node_type: 'transform', plugin: 'web_scrape', "
                        "input: <source on_success>, options: {url_field: '<column>'}}) and route "
                        "the source on_success to it."
                    ),
                    evidence_locator={
                        "source": "blob",
                        "blob_id": str(blob_id),
                        "url_candidates": list(facts.url_candidates),
                    },
                )
            )

    # 4. Surface inspection warnings as info-severity diagnostics so the model
    #    sees them in the same array as blocking issues. These are *advisory*
    #    only — the model can ignore them if the operator's intent justifies.
    #
    #    Exception: ``csv_duplicate_headers`` is promoted to blocking. Duplicate
    #    headers cause silent column collapse in csv.DictReader (last-write-
    #    wins) and similar libraries, fabricating a single column from multiple
    #    source columns. That is a Tier-1 audit-integrity violation — the
    #    audit trail would silently contain data that "looks single-column"
    #    when the source had two — and must force the repair loop, not pass
    #    through as advisory. The repair vocabulary is: rename headers,
    #    declare ``columns`` explicitly, configure ``field_mapping``, or set
    #    ``on_validation_failure`` to a configured quarantine output.
    for warning in facts.warnings:
        if warning.startswith("csv_duplicate_headers:"):
            diagnostics.append(
                _blocking_diagnostic(
                    code="csv_duplicate_headers",
                    message=warning,
                    suggested_repair=(
                        "Rename the duplicate header(s) at the source, OR declare "
                        "explicit `columns` in the source options, OR configure "
                        "`field_mapping` to disambiguate the collapsed names, OR "
                        "set `on_validation_failure` to a configured quarantine "
                        "output so the silent column collapse does not poison the "
                        "audit trail."
                    ),
                    evidence_locator={"source": "blob", "blob_id": str(blob_id)},
                )
            )
            continue
        diagnostics.append(
            {
                "code": "source_inspection_warning",
                "severity": "info",
                "message": warning,
                "suggested_repair": None,
                "evidence_locator": {"source": "blob", "blob_id": str(blob_id)},
            }
        )

    return diagnostics


def _execute_preview_pipeline(
    args: dict[str, Any],
    state: CompositionState,
    catalog: CatalogService,
    data_dir: str | None = None,
    *,
    runtime_preflight: RuntimePreflight | None = None,
    session_engine: Engine | None = None,
    session_id: str | None = None,
) -> ToolResult:
    """Preview pipeline configuration — dry-run validation with source summary.

    Returns ``authoring_validation`` (Stage 1), ``runtime_preflight``
    (Stage 2 from the caller-supplied callback), and ``proof_diagnostics``
    (Stage 3 — operator-input-aware proof against the observed source
    blob). The presence of any blocking ``proof_diagnostics`` entry means
    ``is_valid=False`` even when authoring + runtime checks pass.
    """
    validation = state.validate()
    _AUTHORING_VALIDATION_COUNTER.add(
        1,
        {"outcome": "valid" if validation.is_valid else "invalid"},
    )
    authoring_payload = _authoring_validation_payload(state, validation)
    runtime_result = runtime_preflight(state) if runtime_preflight is not None else None

    proof_diagnostics = compute_proof_diagnostics(
        state,
        session_engine=session_engine,
        session_id=session_id,
    )
    has_blocking_proof = any(d["severity"] == "blocking" for d in proof_diagnostics)

    is_valid = validation.is_valid
    if runtime_result is not None:
        is_valid = is_valid and runtime_result.is_valid
    if has_blocking_proof:
        is_valid = False

    summary: dict[str, Any] = {
        "is_valid": is_valid,
        "errors": authoring_payload["errors"],
        "warnings": authoring_payload["warnings"],
        "suggestions": authoring_payload["suggestions"],
        "edge_contracts": authoring_payload["edge_contracts"],
        "semantic_contracts": authoring_payload["semantic_contracts"],
        "graph_repair_suggestions": authoring_payload["graph_repair_suggestions"],
        "authoring_validation": authoring_payload,
        "runtime_preflight": runtime_result.model_dump() if runtime_result is not None else None,
        "proof_diagnostics": proof_diagnostics,
        "source": None,
        "node_count": len(state.nodes),
        "output_count": len(state.outputs),
        "nodes": [{"id": n.id, "node_type": n.node_type, "plugin": n.plugin} for n in state.nodes],
        "outputs": [{"name": o.name, "plugin": o.plugin} for o in state.outputs],
    }

    if state.source is not None:
        summary["source"] = {
            "plugin": state.source.plugin,
            "on_success": state.source.on_success,
            "has_schema_config": _source_options_have_schema(state.source.options),
        }

    return ToolResult(
        success=True,
        updated_state=state,
        validation=validation,
        affected_nodes=(),
        data=summary,
        runtime_preflight=runtime_result,
    )


def _execute_diff_pipeline(
    args: dict[str, Any],
    state: CompositionState,
    catalog: CatalogService,
    data_dir: str | None = None,
    *,
    baseline: CompositionState | None = None,
    current_validation: ValidationSummary | None = None,
) -> ToolResult:
    """Compute a diff/change summary against a baseline state.

    The baseline is passed explicitly by the MCP server or web composer.
    If no baseline is available, returns a notice instead.

    Args:
        current_validation: Pre-computed validation for the current state.
            Threaded from the caller to avoid redundant recomputation.
    """
    if baseline is None:
        return _discovery_result(
            state,
            {
                _DATA_ERROR_KEY: "No baseline available. Load or create a session first.",
                "current_version": state.version,
            },
        )

    changes = diff_states(baseline, state, current_validation=current_validation)
    return _discovery_result(state, changes)


# --- Registries ---
# Must be after all handler definitions to avoid NameError.

_DISCOVERY_TOOLS: dict[str, ToolHandler] = {
    "list_sources": _handle_list_sources,
    "list_transforms": _handle_list_transforms,
    "list_sinks": _handle_list_sinks,
    "get_plugin_schema": _handle_get_plugin_schema,
    "get_expression_grammar": _handle_get_expression_grammar,
    "explain_validation_error": _execute_explain_validation_error,
    "get_plugin_assistance": _execute_get_plugin_assistance,
    "list_models": _execute_list_models,
    "get_audit_info": _execute_get_audit_info,
    "list_recipes": _execute_list_recipes,
    "get_pipeline_state": _execute_get_pipeline_state,
    "preview_pipeline": _execute_preview_pipeline,
    "diff_pipeline": _execute_diff_pipeline,
}

# All discovery tools are cacheable. If a non-cacheable discovery tool is
# re-added in future (e.g. get_current_state which returns live mutable
# state), add it to _DISCOVERY_TOOLS but NOT to this frozenset.
# preview_pipeline is excluded because its result incorporates runtime_preflight
# output that is externally injected and may change between compose turns.
_CACHEABLE_DISCOVERY_TOOLS: frozenset[str] = frozenset(_DISCOVERY_TOOLS.keys()) - {
    "diff_pipeline",
    "get_pipeline_state",
    "preview_pipeline",
}

_MUTATION_TOOLS: dict[str, ToolHandler] = {
    "set_source": _handle_set_source,
    "upsert_node": _handle_upsert_node,
    "upsert_edge": _handle_upsert_edge,
    "remove_node": _handle_remove_node,
    "remove_edge": _handle_remove_edge,
    "set_metadata": _handle_set_metadata,
    "set_output": _handle_set_output,
    "remove_output": _handle_remove_output,
    "patch_source_options": _handle_patch_source_options,
    "patch_node_options": _handle_patch_node_options,
    "patch_output_options": _handle_patch_output_options,
    "set_pipeline": _handle_set_pipeline,
    "clear_source": _handle_clear_source,
}

# Blob tools use an extended handler signature with session context kwargs
_BLOB_DISCOVERY_TOOLS: dict[str, BlobToolHandler] = {
    "list_blobs": _handle_list_blobs,
    "get_blob_metadata": _handle_get_blob_metadata,
    "get_blob_content": _execute_get_blob_content,
    "inspect_source": _execute_inspect_source,
}

_BLOB_MUTATION_TOOLS: dict[str, BlobToolHandler] = {
    "set_source_from_blob": _execute_set_source_from_blob,
    "create_blob": _execute_create_blob,
    "update_blob": _execute_update_blob,
    "delete_blob": _execute_delete_blob,
    "apply_pipeline_recipe": _execute_apply_pipeline_recipe,
}
_BLOB_QUOTA_MUTATION_TOOLS: frozenset[str] = frozenset(
    {
        "create_blob",
        "update_blob",
        "apply_pipeline_recipe",
    }
)

# Blob-mutation tools that create a NEW blob row and therefore accept the
# ``user_message_id`` provenance kwarg.
# ``set_source_from_blob`` and ``delete_blob`` operate on existing rows
# whose provenance was fixed at create time; ``update_blob`` rewrites the
# file but leaves the original blob row's provenance intact.
# ``apply_pipeline_recipe`` synthesises a ``set_pipeline`` call internally
# that may carry an inline blob, so it also needs the kwarg.
_BLOB_PROVENANCE_MUTATION_TOOLS: frozenset[str] = frozenset(
    {
        "create_blob",
        "apply_pipeline_recipe",
    }
)

# Secret tools use an extended handler signature with secret_service + user_id kwargs
_SECRET_DISCOVERY_TOOLS: dict[str, SecretToolHandler] = {
    "list_secret_refs": _handle_list_secret_refs,
    "validate_secret_ref": _handle_validate_secret_ref,
}

_SECRET_MUTATION_TOOLS: dict[str, SecretToolHandler] = {
    "wire_secret_ref": _execute_wire_secret_ref,
}

# Module-level assertions: registries must not overlap.
_all_tools = (
    set(_DISCOVERY_TOOLS)
    | set(_MUTATION_TOOLS)
    | set(_BLOB_DISCOVERY_TOOLS)
    | set(_BLOB_MUTATION_TOOLS)
    | set(_SECRET_DISCOVERY_TOOLS)
    | set(_SECRET_MUTATION_TOOLS)
)
assert len(_all_tools) == (
    len(_DISCOVERY_TOOLS)
    + len(_MUTATION_TOOLS)
    + len(_BLOB_DISCOVERY_TOOLS)
    + len(_BLOB_MUTATION_TOOLS)
    + len(_SECRET_DISCOVERY_TOOLS)
    + len(_SECRET_MUTATION_TOOLS)
), "Tool registry overlap detected"

assert set(_DISCOVERY_TOOLS) >= _CACHEABLE_DISCOVERY_TOOLS, (
    f"Cacheable tools not in discovery registry: {_CACHEABLE_DISCOVERY_TOOLS - set(_DISCOVERY_TOOLS)}"
)


def is_discovery_tool(name: str) -> bool:
    """Return True if the tool is a discovery (read-only) tool."""
    return name in _DISCOVERY_TOOLS or name in _BLOB_DISCOVERY_TOOLS or name in _SECRET_DISCOVERY_TOOLS


def is_mutation_tool(name: str) -> bool:
    """Return True when a composer tool can mutate session state or owned artifacts."""
    return name in _MUTATION_TOOLS or name in _BLOB_MUTATION_TOOLS or name in _SECRET_MUTATION_TOOLS


# Tools that mutate the session blob store as a side effect but do NOT advance
# CompositionState.version. These are session-scoped artifact writes (file
# uploads, deletions); the pipeline only reacts to them when a *composition*
# mutation (set_pipeline, set_source_from_blob, apply_pipeline_recipe)
# references the resulting blob. Under trust_mode="explicit_approve" these
# must not be intercepted as proposals — the accept endpoint enforces a
# state-version advance (see web/sessions/routes.py:accept_composition_proposal)
# that these tools cannot satisfy by design, so an intercepted proposal is
# structurally unacceptable. The composition mutations that reference the
# resulting blob remain gated and carry the meaningful operator approval.
_BLOB_STORE_ONLY_MUTATION_TOOLS: frozenset[str] = frozenset({"create_blob", "update_blob", "delete_blob"})


def is_blob_store_only_mutation_tool(name: str) -> bool:
    """Return True for blob-store side-effect tools that never advance CompositionState.

    See ``_BLOB_STORE_ONLY_MUTATION_TOOLS`` for the rationale on excluding
    these from the ``trust_mode == "explicit_approve"`` proposal-interception
    gate.
    """
    return name in _BLOB_STORE_ONLY_MUTATION_TOOLS


def is_cacheable_discovery_tool(name: str) -> bool:
    """Return True if the tool's results can be cached within a compose() call."""
    return name in _CACHEABLE_DISCOVERY_TOOLS


# --- Tool Executor ---


def _inject_prior_validation(
    result: ToolResult,
    prior: ValidationSummary,
) -> ToolResult:
    """Attach prior validation to a successful mutation result for delta computation.

    Returns the result unchanged if the mutation failed or already carries
    prior_validation (set explicitly by the handler).
    """
    if result.success and result.prior_validation is None:
        return replace(result, prior_validation=prior)
    return result


def execute_tool(
    tool_name: str,
    arguments: dict[str, Any],
    state: CompositionState,
    catalog: CatalogService,
    data_dir: str | None = None,
    session_engine: Engine | None = None,
    session_id: str | None = None,
    secret_service: Any | None = None,
    user_id: str | None = None,
    baseline: CompositionState | None = None,
    prior_validation: ValidationSummary | None = None,
    runtime_preflight: RuntimePreflight | None = None,
    max_blob_storage_per_session_bytes: int | None = None,
    user_message_id: str | None = None,
) -> ToolResult:
    """Execute a composition tool by name.

    Dispatches via registry dict. Discovery tools return data without
    modifying state. Mutation tools return ToolResult with updated state
    and validation. Unknown tool names return a failure result.

    Args:
        data_dir: Base data directory for S2 path allowlist enforcement.
            When provided, source options containing ``path`` or ``file``
            keys are restricted to ``{data_dir}/blobs/``.
        session_engine: SQLAlchemy engine for the session database.
            Required for blob tools to perform synchronous blob lookups.
        session_id: Current session ID. Required for blob tools.
        secret_service: WebSecretService instance. Required for secret tools.
        user_id: Current user ID. Required for secret tools.
        baseline: Baseline state for diff_pipeline comparisons.
        prior_validation: Pre-computed validation for the current state.
            When provided, mutation tools reuse this instead of calling
            state.validate() for the pre-mutation delta. Callers should
            thread the previous ToolResult.validation forward — the state
            is immutable, so validation is deterministic.
        runtime_preflight: Optional callback for runtime-equivalent preflight.
            Only applied to preview_pipeline. Pre-computed in the async
            compose loop and injected here as a cheap synchronous callback
            so execute_tool() stays synchronous.
        max_blob_storage_per_session_bytes: Configured per-session blob
            storage quota for assistant-created session artifacts. Defaults to
            the historical BlobServiceImpl-compatible value for direct tests
            and non-web callers.
    """
    # preview_pipeline has an extended signature with runtime_preflight kwarg
    # plus session context (session_engine, session_id) so the proof step
    # can inspect blob-backed sources.
    if tool_name == "preview_pipeline":
        return _execute_preview_pipeline(
            arguments,
            state,
            catalog,
            data_dir,
            runtime_preflight=runtime_preflight,
            session_engine=session_engine,
            session_id=session_id,
        )

    # diff_pipeline has an extended signature with baseline kwarg
    if tool_name == "diff_pipeline":
        return _execute_diff_pipeline(
            arguments,
            state,
            catalog,
            data_dir,
            baseline=baseline,
            current_validation=prior_validation,
        )

    # set_pipeline has the standard mutation shape for ordinary sources, but
    # can also own source.inline_blob, which requires session context to create
    # the backing blob before returning the new state.
    if tool_name == "set_pipeline":
        prior = prior_validation if prior_validation is not None else state.validate()
        result = _execute_set_pipeline(
            arguments,
            state,
            catalog,
            data_dir,
            session_engine=session_engine,
            session_id=session_id,
            user_message_id=user_message_id,
            max_blob_storage_per_session_bytes=max_blob_storage_per_session_bytes,
        )
        return _inject_prior_validation(result, prior)

    # Check standard tools first
    discovery_handler = _DISCOVERY_TOOLS.get(tool_name)
    if discovery_handler is not None:
        return discovery_handler(arguments, state, catalog, data_dir)

    mutation_handler = _MUTATION_TOOLS.get(tool_name)
    if mutation_handler is not None:
        prior = prior_validation if prior_validation is not None else state.validate()
        result = mutation_handler(arguments, state, catalog, data_dir)
        return _inject_prior_validation(result, prior)

    # Check blob tools (extended signature with session context)
    blob_discovery = _BLOB_DISCOVERY_TOOLS.get(tool_name)
    if blob_discovery is not None:
        return blob_discovery(arguments, state, catalog, data_dir, session_engine=session_engine, session_id=session_id)

    blob_mutation = _BLOB_MUTATION_TOOLS.get(tool_name)
    if blob_mutation is not None:
        prior = prior_validation if prior_validation is not None else state.validate()
        blob_kwargs: dict[str, Any] = {
            "session_engine": session_engine,
            "session_id": session_id,
        }
        if tool_name in _BLOB_QUOTA_MUTATION_TOOLS:
            blob_kwargs["max_blob_storage_per_session_bytes"] = max_blob_storage_per_session_bytes
        # ``create_blob`` writes the blob row with a
        # ``created_from_message_id`` provenance pointer. Only tools that
        # actually persist a new blob need the kwarg; ``set_source_from_blob``,
        # ``delete_blob``, and ``update_blob`` operate on existing rows
        # whose provenance is fixed at create time.
        if tool_name in _BLOB_PROVENANCE_MUTATION_TOOLS:
            blob_kwargs["user_message_id"] = user_message_id
        result = blob_mutation(
            arguments,
            state,
            catalog,
            data_dir,
            **blob_kwargs,
        )
        return _inject_prior_validation(result, prior)

    # Check secret tools (extended signature with secret_service + user_id)
    secret_discovery = _SECRET_DISCOVERY_TOOLS.get(tool_name)
    if secret_discovery is not None:
        return secret_discovery(arguments, state, catalog, data_dir, secret_service=secret_service, user_id=user_id)

    secret_mutation = _SECRET_MUTATION_TOOLS.get(tool_name)
    if secret_mutation is not None:
        prior = prior_validation if prior_validation is not None else state.validate()
        result = secret_mutation(arguments, state, catalog, data_dir, secret_service=secret_service, user_id=user_id)
        return _inject_prior_validation(result, prior)

    return _failure_result(state, f"Unknown tool: {tool_name}")


# ---------------------------------------------------------------------------
# request_interpretation_review (session-aware async tool)
# ---------------------------------------------------------------------------
#
# This is the first session-aware async composer tool. Unlike the state-pure
# handlers above, ``_handle_request_interpretation_review`` AWAITS the
# session-service writer (``create_pending_interpretation_event``) and the
# session-service reader (``list_interpretation_events``) needed by the
# rate-limit check. It cannot live in ``_DISCOVERY_TOOLS`` / ``_MUTATION_TOOLS``
# because those registries hold synchronous handlers; ``execute_tool`` is
# called inside ``run_sync_in_worker(...)`` and cannot ``await`` anything.
#
# Dispatch contract (mirrors the ``request_advisor_hint`` precedent at
# ``web/composer/service.py`` around line 2820): the compose loop intercepts
# the tool name BEFORE ``run_sync_in_worker(execute_tool, ...)`` and awaits
# the handler in this registry directly. The audit envelope opened by
# ``begin_dispatch_or_arg_error`` covers the dispatch on both success and
# ARG_ERROR paths.


# Regex for detecting placeholders of the form ``{{interpretation:<term>}}``
# inside a prompt_template string. Used by both the per-tool boundary check
# (``_assert_affected_llm_node`` — confirms the LLM's draft is being staged
# against a transform that actually has the placeholder) and the runtime
# pre-execution detector (``_detect_unresolved_interpretation_placeholders``
# — F-17). The capture group is the term itself, used to surface which
# placeholder is unresolved.
#
# The pattern accepts whitespace inside the braces (``{{ interpretation : cool }}``)
# because LLM drafts vary. The captured term is trimmed by the caller before
# comparison so leading/trailing whitespace on the term does not cause a
# false negative.
def _assert_affected_llm_node(
    state: CompositionState,
    affected_node_id: str,
    user_term: str,
) -> None:
    """Tier-3 boundary check on the LLM-supplied ``affected_node_id``.

    Raises :class:`ToolArgumentError` with an actionable message when:

    * the node does not exist in ``state.nodes``;
    * the node's plugin kind is not ``llm``;
    * the node's ``prompt_template`` does not contain a placeholder of the
      form ``{{interpretation:<user_term>}}``.

    Each branch raises ARG_ERROR (not a Tier-1 crash) because the LLM is
    expected to recover by calling ``upsert_node`` to add the placeholder
    and re-staging the tool call. A crash here would conflate "LLM made
    a recoverable mistake" with "we have a bug in our own code".

    The placeholder check is lenient on whitespace inside the braces but
    strict on the term value: the term inside the placeholder must equal
    ``user_term`` after both sides are stripped. This avoids false
    positives where the LLM staged a review for ``"important"`` but the
    transform's placeholder is ``{{interpretation:cool}}``.
    """
    node = next((n for n in state.nodes if n.id == affected_node_id), None)
    if node is None:
        known = sorted(n.id for n in state.nodes)
        raise ToolArgumentError(
            argument="affected_node_id",
            expected=f"id of an existing LLM transform (known ids: {known!r})",
            actual_type=f"unknown id {affected_node_id!r}",
        )
    plugin = node.plugin
    if plugin != "llm":
        raise ToolArgumentError(
            argument="affected_node_id",
            expected="id of a node whose plugin is 'llm'",
            actual_type=f"node {affected_node_id!r} has plugin={plugin!r}",
        )
    options = node.options if node.options else {}
    prompt_template = options.get("prompt_template") if isinstance(options, Mapping) else None
    if not isinstance(prompt_template, str) or not prompt_template:
        raise ToolArgumentError(
            argument="affected_node_id",
            expected=f"node {affected_node_id!r} to declare options.prompt_template (str) containing {{{{interpretation:{user_term}}}}}",
            actual_type=f"options.prompt_template is {type(prompt_template).__name__}",
        )
    matched_terms = [match.group(1).strip() for match in INTERPRETATION_PLACEHOLDER_RE.finditer(prompt_template)]
    if user_term.strip() not in matched_terms:
        raise ToolArgumentError(
            argument="affected_node_id",
            expected=(
                f"node {affected_node_id!r} prompt_template to contain placeholder "
                f"{{{{interpretation:{user_term}}}}} (found placeholders for: {matched_terms!r})"
            ),
            actual_type="missing placeholder",
        )


def _detect_unresolved_interpretation_placeholders(nodes: Mapping[str, Any]) -> list[str]:
    """Return the list of terms with unresolved ``{{interpretation:…}}`` placeholders.

    F-17 runtime detector — invoked at the boundary between composition and
    execution. ``nodes`` is a mapping of node-id to a node dict whose
    ``options.prompt_template`` field is inspected. Non-LLM nodes are
    skipped. The return value is a list of placeholder terms (deduplicated
    by insertion order); an empty list means the pipeline is safe to
    execute.

    Standalone (no compose-loop state) so the helper is testable in
    isolation. Production callers (the executor / preview path) raise
    ``RuntimeError`` and emit the
    ``interpretation_placeholder_unresolved_at_runtime`` operational
    telemetry signal with each ``node_id`` and ``term`` (NOT the prompt
    template value — that may carry user content).
    """
    unresolved: dict[str, None] = {}  # ordered set
    for node in nodes.values():
        if not isinstance(node, Mapping):
            continue
        if node.get("kind") != "llm":
            continue
        options = node.get("options")
        prompt_template = options.get("prompt_template") if isinstance(options, Mapping) else None
        if not isinstance(prompt_template, str):
            continue
        for match in INTERPRETATION_PLACEHOLDER_RE.finditer(prompt_template):
            unresolved[match.group(1).strip()] = None
    return list(unresolved.keys())


def _detect_unresolved_interpretation_placeholders_typed(
    nodes: Sequence[NodeSpec],
) -> list[tuple[str, str]]:
    """Return (node_id, term) tuples for every unresolved ``{{interpretation:…}}`` placeholder.

    F-17 runtime detector — typed sibling of
    :func:`_detect_unresolved_interpretation_placeholders` that operates
    directly on ``CompositionState.nodes`` (a ``Sequence[NodeSpec]``).
    The dict-shaped helper above filters on ``node.get("kind") == "llm"``
    because it walks the runtime YAML/pipeline dict shape where transforms
    carry a ``kind`` discriminator; ``NodeSpec`` has no ``kind`` field —
    LLM transforms are identified by ``node.plugin == "llm"`` (mirrors the
    per-tool boundary check in :func:`_assert_affected_llm_node`).
    Substituting ``kind`` here would match nothing and silently fail open,
    which is why this sibling exists rather than a single polymorphic
    helper.

    Each unresolved placeholder produces exactly one tuple per
    ``(node_id, term)`` pair, deduplicated within a node by insertion
    order so a repeated placeholder in one prompt template does not
    inflate telemetry / error surface area.  Cross-node duplicates ARE
    preserved (the same ``term`` on two different nodes is two distinct
    unresolved sites).

    The return type is a ``list`` (not a ``tuple``) to match the
    dict-shaped sibling and the spec at
    ``docs/composer/ux-redesign-2026-05/18a-phase-5b-backend.md`` §F-17
    (``list[str]`` of terms, lifted to ``list[tuple[str, str]]`` here
    because the typed call site needs the ``node_id`` for both the
    telemetry attribute and the user-actionable error message).
    """
    unresolved: list[tuple[str, str]] = []
    for node in nodes:
        if node.plugin != "llm":
            continue
        if "prompt_template" not in node.options:
            continue
        prompt_template = node.options["prompt_template"]
        if not isinstance(prompt_template, str):
            raise ToolArgumentError(
                argument="nodes[].options.prompt_template",
                expected="a string",
                actual_type=type(prompt_template).__name__,
            )
        seen_in_node: dict[str, None] = {}
        for match in INTERPRETATION_PLACEHOLDER_RE.finditer(prompt_template):
            term = match.group(1).strip()
            if term in seen_in_node:
                continue
            seen_in_node[term] = None
            unresolved.append((node.id, term))
    return unresolved


# Rate-cap discriminant codes carried on ``ToolArgumentError.code`` so the
# compose loop can branch on the cap type without parsing the message. These
# are fixed string constants — never substituted with LLM/user-supplied
# content — so they are safe to read into telemetry attributes.
RATE_CAP_PER_TERM_CODE: Final[str] = "RATE_CAP_PER_TERM"
RATE_CAP_PER_SESSION_DAY_CODE: Final[str] = "RATE_CAP_PER_SESSION_DAY"

# Mapping from rate-cap discriminant code → telemetry ``cap_type`` attribute
# value (per the F-15 spec at docs/composer/ux-redesign-2026-05/18a-phase-5b-backend.md
# §"Telemetry posture (F-15)"). Both keys MUST appear; the absence of a
# mapping is caught by the ``test_rate_cap_codes_map_to_telemetry_cap_type``
# parity test in tests/unit/web/composer/test_request_interpretation_review_tool.py.
RATE_CAP_CODE_TO_TELEMETRY_CAP_TYPE: Final[Mapping[str, str]] = MappingProxyType(
    {
        RATE_CAP_PER_TERM_CODE: "per_term",
        RATE_CAP_PER_SESSION_DAY_CODE: "per_session_day",
    }
)


def _utc_day_start(now: datetime) -> datetime:
    """Return the UTC-midnight start of the calendar day containing ``now``.

    F-30 fixed-window helper. The rate-limit window is the *calendar day in
    UTC*, not a sliding 24-hour window — simpler for operators to reason
    about and produces predictable reset behaviour ("the counter resets at
    UTC midnight"). The caller compares ``event.created_at >= utc_day_start(now)``.
    """
    aware_now = now if now.tzinfo is not None else now.replace(tzinfo=UTC)
    aware_now_utc = aware_now.astimezone(UTC)
    return aware_now_utc.replace(hour=0, minute=0, second=0, microsecond=0)


async def _check_interpretation_rate_limits(
    *,
    session_id: UUID,
    user_term: str,
    composition_state_id: UUID,
    list_events_fn: Callable[..., Awaitable[list[InterpretationEventRecord]]],
    per_term_cap: int,
    per_session_day_cap: int,
    now: datetime,
) -> None:
    """Enforce the two per-session interpretation-review rate limits (F-30/F-31).

    Two structural limits apply, in order:

    1. **Per-term cap (default 3):** the same ``(session_id, user_term)`` pair,
       scoped to the active ``composition_state_id`` branch, may be surfaced
       at most ``per_term_cap`` times before the LLM must fall back to
       AUTO_INTERPRETED_NO_SURFACES. Counted against rows that are NOT
       AUTO_INTERPRETED_OPT_OUT shape (those have NULL ``user_term``) and
       whose ``composition_state_id`` matches.

    2. **Per-session-day cap (default 10):** the session may make at most
       ``per_session_day_cap`` ``request_interpretation_review`` invocations
       per UTC calendar day. The window starts at UTC midnight (NOT a
       sliding 24-hour window) so the reset behaviour is predictable.

    On cap exceeded: raises :class:`ToolArgumentError`. The compose loop
    catches this and is expected (per the composer skill) to fall back to
    a non-LLM interpretation with ``interpretation_source =
    'auto_interpreted_no_surfaces'``.

    Async because ``list_events_fn`` (the injected
    ``list_interpretation_events`` service method) is async — it reads the
    session DB. A sync helper would force a blocking ``asyncio.run(...)``
    inside an already-async caller, which deadlocks.

    The cap values are injected as kwargs (not read from settings inside
    the helper) so this function is testable without a live settings
    object. Production callers thread ``WebSettings.composer_interpretation_*``
    in.
    """
    events = await list_events_fn(session_id, status="all")
    # Per-term cap — count rows for this composition branch with matching user_term.
    per_term_count = sum(
        1
        for event in events
        if event.composition_state_id == composition_state_id and event.user_term is not None and event.user_term == user_term
    )
    if per_term_count >= per_term_cap:
        raise ToolArgumentError(
            argument="user_term",
            expected=f"at most {per_term_cap} interpretation requests per term in this composition",
            actual_type=(
                f"term {user_term!r} would be surfaced {per_term_count + 1} times — use a direct "
                f"interpretation in the prompt template instead"
            ),
            # Compose-loop discriminant (F-6): the rate-cap branch is the
            # trigger for the
            # AUTO_INTERPRETED_NO_SURFACES writer + F-15 telemetry. The
            # ``code`` field lets the loop distinguish this exception from a
            # generic ARG_ERROR without grepping the message string.
            code=RATE_CAP_PER_TERM_CODE,
        )
    # Per-session-day cap — UTC-midnight fixed window. Only rows with
    # populated ``user_term`` (i.e. not opt-out skeletons) count toward the
    # invocation budget; opt-out is a different action with its own row.
    day_start = _utc_day_start(now)
    per_day_count = sum(1 for event in events if event.user_term is not None and event.created_at >= day_start)
    if per_day_count >= per_session_day_cap:
        raise ToolArgumentError(
            argument="user_term",
            expected=f"at most {per_session_day_cap} interpretation requests per session per UTC day",
            actual_type=(
                f"session would record {per_day_count + 1} requests today — the compose loop should "
                f"fall back to auto-interpretation (AUTO_INTERPRETED_NO_SURFACES)"
            ),
            # See per-term cap above for the ``code`` field rationale.
            code=RATE_CAP_PER_SESSION_DAY_CODE,
        )


async def _handle_request_interpretation_review(
    arguments: object,
    state: CompositionState,
    *,
    session_id: UUID,
    composition_state_id: UUID,
    tool_call_id: str,
    now: datetime,
    per_term_cap: int,
    per_session_day_cap: int,
    model_identifier: str,
    model_version: str,
    provider: str,
    composer_skill_hash: str,
    create_pending_interpretation_event: Callable[..., Awaitable[InterpretationEventRecord]],
    list_interpretation_events: Callable[..., Awaitable[list[InterpretationEventRecord]]],
) -> ToolResult:
    """Stage a pending interpretation event for user review.

    Returns a SUCCESS :class:`ToolResult` whose ``data`` payload signals
    the frontend to surface the review affordance. Does NOT advance
    composition state version — state mutation happens at /resolve time
    when the user accepts or amends the draft.

    Async because it awaits the injected service methods. Registered in
    :data:`_SESSION_AWARE_TOOL_HANDLERS` (NOT the synchronous
    :data:`_MUTATION_TOOLS` registry — see the dual-registry invariant
    documented at the registry block above).
    """
    parsed = cast(
        _RequestInterpretationReviewArgumentsModel,
        _validate_mutation_arguments(
            _RequestInterpretationReviewArgumentsModel,
            arguments,
            "request_interpretation_review arguments",
        ),
    )
    # F-34 credential prefilter: Tier-3 boundary check before any DB write.
    # ``_reject_credential_shaped_content`` raises ``ValueError``; we wrap
    # as ToolArgumentError so the compose loop's ARG_ERROR routing catches
    # it (a bare ValueError would land in the plugin-crash catch-all and
    # mis-classify the failure as a Tier-1 plugin bug).
    for field_name, field_value in (("user_term", parsed.user_term), ("llm_draft", parsed.llm_draft)):
        try:
            _reject_credential_shaped_content(field_value)
        except ValueError as exc:
            raise ToolArgumentError(
                argument=field_name,
                expected="content that does not match a known credential shape",
                actual_type="credential-shaped content rejected at the tool boundary",
            ) from exc
    # F-2 prompt-injection guard on llm_draft. Applied BEFORE the DB write
    # so a poisoned draft (e.g. ``{{system:override}}``) cannot enter the
    # audit row and reach the accepted_as_drafted resolution path where it
    # would be embedded directly into the prompt template.
    try:
        _validate_accepted_value_content(parsed.llm_draft)
    except ValueError as exc:
        raise ToolArgumentError(
            argument="llm_draft",
            expected="content without template metacharacters, control characters, or credential patterns",
            actual_type="rejected by accepted-value content validator",
        ) from exc
    # Rate-limit gate. Cap-exceeded raises ToolArgumentError; the compose
    # loop is expected to react by writing an AUTO_INTERPRETED_NO_SURFACES
    # event (handled in service.py — see ``record_auto_interpreted_no_surfaces_event``).
    await _check_interpretation_rate_limits(
        session_id=session_id,
        user_term=parsed.user_term,
        composition_state_id=composition_state_id,
        list_events_fn=list_interpretation_events,
        per_term_cap=per_term_cap,
        per_session_day_cap=per_session_day_cap,
        now=now,
    )
    # Tier-3 boundary check on the LLM-supplied affected_node_id. Three
    # branches (missing node / wrong kind / missing placeholder) all raise
    # ARG_ERROR so the LLM can retry after fixing the prompt template.
    _assert_affected_llm_node(state, parsed.affected_node_id, parsed.user_term)
    # Persist the pending row. The service method re-validates affected_node_id
    # under the session write lock (defence in depth — the compose-state could
    # in principle race with another writer; the service is the authoritative
    # consistency gate).
    event = await create_pending_interpretation_event(
        session_id=session_id,
        composition_state_id=composition_state_id,
        affected_node_id=parsed.affected_node_id,
        tool_call_id=tool_call_id,
        user_term=parsed.user_term,
        llm_draft=parsed.llm_draft,
        model_identifier=model_identifier,
        model_version=model_version,
        provider=provider,
        composer_skill_hash=composer_skill_hash,
    )
    if event.interpretation_source is InterpretationSource.AUTO_INTERPRETED_OPT_OUT:
        return ToolResult(
            success=True,
            updated_state=state,
            validation=state.validate(),
            affected_nodes=(),
            data={
                "_kind": "interpretation_review_suppressed_by_opt_out",
                "event_id": str(event.id),
                "interpretation_review_disabled": True,
                "message": "Interpretation review suppressed because this session has opted out.",
            },
        )
    return ToolResult(
        success=True,
        updated_state=state,  # no state change yet — /resolve advances version
        validation=state.validate(),
        affected_nodes=(parsed.affected_node_id,),
        data={
            "_kind": "interpretation_review_pending",
            "event_id": str(event.id),
            "affected_node_id": parsed.affected_node_id,
            "user_term": parsed.user_term,
            "llm_draft": parsed.llm_draft,
            "message": (
                f"Interpretation review staged for '{parsed.user_term}'. "
                f"Waiting for user acceptance/amendment before the pipeline can finalise."
            ),
        },
    )


# Session-aware async tool handler registry.
# See the dual-registry invariant documented at the registry block above
# (search for "Dual-registry invariant (F-18)").
SessionAwareToolHandler = Callable[..., Awaitable[ToolResult]]

_SESSION_AWARE_TOOL_HANDLERS: dict[str, SessionAwareToolHandler] = {
    "request_interpretation_review": _handle_request_interpretation_review,
}


def is_session_aware_tool(name: str) -> bool:
    """Return True if the tool requires async dispatch with session context.

    Session-aware tools are intercepted in the compose loop BEFORE
    ``execute_tool`` is called — they cannot be dispatched through the
    synchronous worker because they await session-service methods.
    """
    return name in _SESSION_AWARE_TOOL_HANDLERS


# Dual-registry invariant assertions (F-18). These execute at module import,
# so a regression — for example, copy-pasting an async handler into a sync
# registry — fails the build before any compose() call could trigger silent
# "coroutine was never awaited" warnings.
_all_tools_v2 = (
    set(_DISCOVERY_TOOLS)
    | set(_MUTATION_TOOLS)
    | set(_BLOB_DISCOVERY_TOOLS)
    | set(_BLOB_MUTATION_TOOLS)
    | set(_SECRET_DISCOVERY_TOOLS)
    | set(_SECRET_MUTATION_TOOLS)
    | set(_SESSION_AWARE_TOOL_HANDLERS)
)
assert len(_all_tools_v2) == (
    len(_DISCOVERY_TOOLS)
    + len(_MUTATION_TOOLS)
    + len(_BLOB_DISCOVERY_TOOLS)
    + len(_BLOB_MUTATION_TOOLS)
    + len(_SECRET_DISCOVERY_TOOLS)
    + len(_SECRET_MUTATION_TOOLS)
    + len(_SESSION_AWARE_TOOL_HANDLERS)
), (
    "Tool registry overlap detected — a tool name appears in more than one of _DISCOVERY_TOOLS / _MUTATION_TOOLS / blob / secret / _SESSION_AWARE_TOOL_HANDLERS"
)

# Every session-aware handler must be a coroutine function. A sync function
# accidentally registered here would silently return a non-Awaitable; the
# compose-loop ``await`` would crash with TypeError at the worst time.
for _name, _handler in _SESSION_AWARE_TOOL_HANDLERS.items():
    assert asyncio.iscoroutinefunction(_handler), (
        f"_SESSION_AWARE_TOOL_HANDLERS[{_name!r}] is not async; sync handlers belong in _MUTATION_TOOLS / _DISCOVERY_TOOLS instead."
    )

# Every sync-registry handler must NOT be a coroutine. Catches the reverse
# regression: an async handler dropped into the sync dispatch path that
# would return a coroutine object as if it were a ToolResult.
#
# The six sync registries have heterogeneous handler value-types (the blob
# and secret registries carry handlers with extra session-context kwargs),
# so the local ``_sync_registry`` is typed broadly as
# ``Mapping[str, Callable[..., Any]]`` for the duration of this check.
_sync_registries_for_check: tuple[tuple[str, Mapping[str, Callable[..., Any]]], ...] = (
    ("_DISCOVERY_TOOLS", cast(Mapping[str, Callable[..., Any]], _DISCOVERY_TOOLS)),
    ("_MUTATION_TOOLS", cast(Mapping[str, Callable[..., Any]], _MUTATION_TOOLS)),
    ("_BLOB_DISCOVERY_TOOLS", cast(Mapping[str, Callable[..., Any]], _BLOB_DISCOVERY_TOOLS)),
    ("_BLOB_MUTATION_TOOLS", cast(Mapping[str, Callable[..., Any]], _BLOB_MUTATION_TOOLS)),
    ("_SECRET_DISCOVERY_TOOLS", cast(Mapping[str, Callable[..., Any]], _SECRET_DISCOVERY_TOOLS)),
    ("_SECRET_MUTATION_TOOLS", cast(Mapping[str, Callable[..., Any]], _SECRET_MUTATION_TOOLS)),
)
for _sync_registry_name, _sync_registry in _sync_registries_for_check:
    for _name, _handler in _sync_registry.items():
        assert not asyncio.iscoroutinefunction(_handler), (
            f"{_sync_registry_name}[{_name!r}] is async; async handlers belong in _SESSION_AWARE_TOOL_HANDLERS instead."
        )
