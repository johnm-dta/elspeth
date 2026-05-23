"""Shared toolkit for the composer-tool plane modules.

Centralises:

- ``ToolResult`` (the canonical response shape every handler returns) and the
  leaf response helpers (``_failure_result`` / ``_discovery_result`` /
  ``_mutation_result`` / ``_prepend_rejection_entry`` / ``_attach_post_call_hints``).
- Validation-delta and graph-repair-suggestion synthesis used by
  ``ToolResult.to_dict`` and the high-level ``diff_states`` reporter.
- The Pydantic mutation-argument validator and merge-patch helper used by every
  per-resource mutation handler.
- Base serialisation helpers for source/node/output/edge — leaves that the
  repair-suggestion generator and downstream pipeline-state serialiser both consume.
- The TypedDicts shared across pipeline-state, edge-contract, and repair payloads.

Layer: L3 (application). Imports from L0 contracts and the ``web.composer.state`` /
``web.composer.protocol`` / ``web.catalog.protocol`` / ``web.execution.schemas``
surfaces only — no sibling-plane imports.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Final, TypedDict

from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from elspeth.contracts.freeze import deep_thaw, freeze_fields
from elspeth.web.catalog.protocol import CatalogService, PluginKind
from elspeth.web.composer.protocol import ToolArgumentError
from elspeth.web.composer.state import (
    CompositionState,
    EdgeSpec,
    NodeSpec,
    OutputSpec,
    SourceSpec,
    ValidationEntry,
    ValidationSummary,
    _coalesce_branch_connections,
    _coalesce_branch_names,
    _serialize_branches,
)
from elspeth.web.execution.schemas import ValidationResult

_DATA_ERROR_KEY: Final[str] = "error"


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


def _validate_mutation_arguments(model: type[BaseModel], arguments: object, argument_name: str) -> BaseModel:
    try:
        return model.model_validate(arguments)
    except PydanticValidationError as exc:
        raise ToolArgumentError(
            argument=argument_name,
            expected=f"object conforming to {model.__name__}",
            actual_type=type(exc).__name__,
        ) from exc


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
