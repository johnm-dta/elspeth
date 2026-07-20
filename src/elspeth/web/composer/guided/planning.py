"""Pure authority, redaction, and projection helpers for guided planning.

This module has no persistence or provider authority.  It snapshots the exact
reviewed facts used by :class:`PipelineProposal`, builds the deliberately less
capable model context, and projects a private canonical pipeline into the
closed ``PROPOSE_PIPELINE`` wire contract.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, NotRequired, TypedDict, cast
from uuid import UUID, uuid4

from pydantic import JsonValue

from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.freeze import deep_thaw, freeze_fields
from elspeth.contracts.hashing import stable_hash
from elspeth.web.composer.guided.connection_consumers import canonical_connection_consumers
from elspeth.web.composer.guided.deferred_intents import DeferredIntentClaimError, evaluate_deferred_intent_coverage
from elspeth.web.composer.guided.protocol import (
    PROPOSAL_RATIONALE_TEMPLATE,
    PROPOSAL_SUMMARY_TEMPLATE,
    ProposePipelinePayload,
    TurnType,
    proposal_component_label,
    proposal_structural_label,
    validate_payload,
    validate_proposal_catalog_refs,
)
from elspeth.web.composer.guided.stage_subjects import (
    ComponentCountConstraint,
    EdgeRouteConstraint,
    FailureRouteConstraint,
    OptionValueConstraint,
    SubjectPresenceConstraint,
)
from elspeth.web.composer.guided.state_machine import ComponentTarget, DeferredStageIntent, GuidedSession
from elspeth.web.composer.pipeline_proposal import PipelineProposal
from elspeth.web.composer.state import CompositionState, NodeSpec


class GuidedBoundSource(TypedDict):
    """One reviewed source restored into a planner-authored topology."""

    plugin: str
    options: dict[str, JsonValue]
    on_success: str
    on_validation_failure: str


class GuidedBoundOutput(TypedDict):
    """One reviewed output restored into a planner-authored topology."""

    sink_name: str
    plugin: str
    options: dict[str, JsonValue]
    on_write_failure: str


class GuidedBoundPipeline(TypedDict):
    """Validated set-pipeline shape after guided authority rebinding."""

    sources: dict[str, GuidedBoundSource]
    nodes: list[dict[str, JsonValue]]
    edges: list[dict[str, JsonValue]]
    outputs: list[GuidedBoundOutput]
    metadata: NotRequired[dict[str, JsonValue] | None]


@dataclass(frozen=True, slots=True)
class GuidedCorrectionTarget:
    """One closed public selection plus its authoritative private owner."""

    requested: ComponentTarget
    owner_kind: Literal["source", "node", "output"]
    owner_key: str
    authority_key: str | None
    public_target: Mapping[str, Any]
    before_fingerprint: str

    def __post_init__(self) -> None:
        if (self.requested.kind == "edge") != (self.authority_key is None):
            raise ValueError("edge correction targets must not claim a private positional edge identity")
        freeze_fields(self, "public_target")

    def planner_context(self) -> dict[str, object]:
        return {
            "kind": self.requested.kind,
            "stable_id": self.requested.stable_id,
            "owner_kind": self.owner_kind,
            "owner_key": self.owner_key,
            "target": deep_thaw(self.public_target),
        }


def _wire_target_fingerprint(
    payload: Mapping[str, Any],
    *,
    collection: Literal["sources", "nodes", "connections", "outputs"],
    index: int,
    authority: CompositionState,
) -> str | None:
    """Fingerprint selected public semantics independent of regenerated IDs."""

    raw_collection = payload.get(collection)
    if type(raw_collection) is not list or index >= len(raw_collection) or type(raw_collection[index]) is not dict:
        return None
    component = cast(dict[str, Any], deep_thaw(raw_collection[index]))
    stable_id = component.pop("stable_id", None)

    authority_keys = {
        "source": tuple(authority.sources),
        "node": tuple(node.id for node in authority.nodes),
        "output": tuple(output.name for output in authority.outputs),
    }
    identities: dict[tuple[str, str], str] = {}
    for component_kind, collection_name in (("source", "sources"), ("node", "nodes"), ("output", "outputs")):
        values = payload.get(collection_name)
        keys = authority_keys[component_kind]
        if type(values) is not list or len(values) != len(keys):
            raise AuditIntegrityError("guided correction wire projection lost identity collections")
        for position, value in enumerate(values):
            if type(value) is not dict or type(value.get("stable_id")) is not str:
                raise AuditIntegrityError("guided correction wire projection has malformed stable identities")
            identities[(component_kind, value["stable_id"])] = keys[position]

    def normalize_endpoint(value: object) -> object:
        if type(value) is not dict:
            return value
        kind = value.get("kind")
        endpoint_id = value.get("stable_id")
        if kind == "discard":
            return {"kind": "discard"}
        if type(kind) is not str or type(endpoint_id) is not str or (kind, endpoint_id) not in identities:
            raise AuditIntegrityError("guided correction wire edge has an unbound endpoint")
        return {"kind": kind, "key": identities[(kind, endpoint_id)]}

    if collection == "connections":
        component["from_endpoint"] = normalize_endpoint(component.get("from_endpoint"))
        component["to_endpoint"] = normalize_endpoint(component.get("to_endpoint"))
    else:
        connections = payload.get("connections")
        if type(connections) is not list:
            raise AuditIntegrityError("guided correction wire projection lost connections")
        incident = []
        for connection in connections:
            if type(connection) is not dict:
                raise AuditIntegrityError("guided correction wire projection has a malformed connection")
            origin = connection.get("from_endpoint")
            destination = connection.get("to_endpoint")
            is_origin = type(origin) is dict and origin.get("stable_id") == stable_id
            is_destination = type(destination) is dict and destination.get("stable_id") == stable_id
            if not is_origin and not is_destination:
                continue
            normalized = cast(dict[str, Any], deep_thaw(connection))
            normalized.pop("stable_id", None)
            normalized["from_endpoint"] = normalize_endpoint(normalized.get("from_endpoint"))
            normalized["to_endpoint"] = normalize_endpoint(normalized.get("to_endpoint"))
            incident.append(normalized)
        component["incident_connections"] = incident
    return stable_hash(component)


def resolve_guided_correction_target(
    *,
    requested: ComponentTarget,
    wire_payload: Mapping[str, Any],
    predecessor: CompositionState,
) -> GuidedCorrectionTarget:
    """Resolve one exact public stable ID without inventing private edge identity."""

    def resolve_owner(kind: str, stable_id: str) -> tuple[Literal["source", "node", "output"], str, int]:
        collection_name = {"source": "sources", "node": "nodes", "output": "outputs"}.get(kind)
        if collection_name is None:
            raise AuditIntegrityError("guided correction edge has an unsupported origin kind")
        components = wire_payload.get(collection_name)
        if type(components) is not list:
            raise AuditIntegrityError("guided correction wire projection lost a component collection")
        positions = [index for index, item in enumerate(components) if type(item) is dict and item.get("stable_id") == stable_id]
        if len(positions) != 1:
            raise AuditIntegrityError("guided correction stable target does not resolve exactly once")
        index = positions[0]
        if kind == "source":
            names = list(predecessor.sources)
            if index >= len(names):
                raise AuditIntegrityError("guided correction source target differs from private authority")
            return "source", names[index], index
        if kind == "node":
            if index >= len(predecessor.nodes):
                raise AuditIntegrityError("guided correction node target differs from private authority")
            return "node", predecessor.nodes[index].id, index
        if index >= len(predecessor.outputs):
            raise AuditIntegrityError("guided correction output target differs from private authority")
        return "output", predecessor.outputs[index].name, index

    if requested.kind == "edge":
        connections = wire_payload.get("connections")
        if type(connections) is not list:
            raise AuditIntegrityError("guided correction wire projection lost connections")
        matches = [item for item in connections if type(item) is dict and item.get("stable_id") == requested.stable_id]
        if len(matches) != 1 or type(matches[0].get("from_endpoint")) is not dict:
            raise AuditIntegrityError("guided correction edge target differs from private authority")
        origin = matches[0]["from_endpoint"]
        owner_kind, owner_key, _owner_index = resolve_owner(origin.get("kind"), origin.get("stable_id"))
        collection_index = connections.index(matches[0])
        collection: Literal["sources", "nodes", "connections", "outputs"] = "connections"
        authority_key = None
        public_target = matches[0]
    else:
        owner_kind, owner_key, collection_index = resolve_owner(requested.kind, requested.stable_id)
        authority_key = owner_key
        if requested.kind == "source":
            collection = "sources"
        elif requested.kind == "node":
            collection = "nodes"
        else:
            collection = "outputs"
        components = wire_payload.get(collection)
        if type(components) is not list or type(components[collection_index]) is not dict:
            raise AuditIntegrityError("guided correction target differs from public wire authority")
        public_target = components[collection_index]
    before_fingerprint = _wire_target_fingerprint(
        wire_payload,
        collection=collection,
        index=collection_index,
        authority=predecessor,
    )
    if before_fingerprint is None:
        raise AuditIntegrityError("guided correction target owner is absent from wire authority")
    if requested.kind == "edge":
        edge_connections = wire_payload.get("connections")
        if type(edge_connections) is not list:
            raise AuditIntegrityError("guided correction wire projection lost connections")
        matching_fingerprints = sum(
            _wire_target_fingerprint(
                wire_payload,
                collection="connections",
                index=index,
                authority=predecessor,
            )
            == before_fingerprint
            for index in range(len(edge_connections))
        )
        if matching_fingerprints != 1:
            raise AuditIntegrityError("guided correction edge semantics do not resolve exactly once")
    return GuidedCorrectionTarget(
        requested=requested,
        owner_kind=owner_kind,
        owner_key=owner_key,
        authority_key=authority_key,
        public_target=public_target,
        before_fingerprint=before_fingerprint,
    )


def require_guided_correction_target_changed(
    wire_payload: Mapping[str, Any],
    target: GuidedCorrectionTarget,
    successor: CompositionState,
) -> None:
    """Reject a plan that edited elsewhere while leaving exact target semantics intact."""

    if target.requested.kind == "edge":
        connections = wire_payload.get("connections")
        if type(connections) is not list:
            raise AuditIntegrityError("guided correction successor lost public connections")
        fingerprints = tuple(
            _wire_target_fingerprint(
                wire_payload,
                collection="connections",
                index=index,
                authority=successor,
            )
            for index in range(len(connections))
        )
        if target.before_fingerprint in fingerprints:
            raise AuditIntegrityError("guided correction planner did not change the selected component")
        return

    if target.requested.kind == "source":
        collection: Literal["sources", "nodes", "outputs"] = "sources"
        successor_keys = tuple(successor.sources)
    elif target.requested.kind == "node":
        collection = "nodes"
        successor_keys = tuple(node.id for node in successor.nodes)
    else:
        collection = "outputs"
        successor_keys = tuple(output.name for output in successor.outputs)
    positions = [index for index, key in enumerate(successor_keys) if key == target.authority_key]
    if not positions:
        return
    if len(positions) != 1:
        raise AuditIntegrityError("guided correction successor duplicated the selected component")

    fingerprint = _wire_target_fingerprint(
        wire_payload,
        collection=collection,
        index=positions[0],
        authority=successor,
    )
    if fingerprint == target.before_fingerprint:
        raise AuditIntegrityError("guided correction planner did not change the selected component")


def guided_private_reviewed_facts(guided: GuidedSession) -> dict[str, object]:
    """Return the exact ordered facts whose hash is stored in a guided ref."""

    return {
        "source_order": list(guided.source_order),
        "reviewed_sources": {stable_id: guided.reviewed_sources[stable_id].to_dict() for stable_id in guided.source_order},
        "output_order": list(guided.output_order),
        "reviewed_outputs": {stable_id: guided.reviewed_outputs[stable_id].to_dict() for stable_id in guided.output_order},
    }


def _provider_safe_deferred_constraint(
    constraint: SubjectPresenceConstraint | OptionValueConstraint | ComponentCountConstraint | EdgeRouteConstraint | FailureRouteConstraint,
) -> dict[str, object]:
    """Project one private constraint without option paths or values."""

    if type(constraint) is SubjectPresenceConstraint:
        return {
            "kind": constraint.kind,
            "subject": constraint.subject.to_dict(),
            "present": constraint.present,
        }
    if type(constraint) is OptionValueConstraint:
        value_type = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            type(None): "null",
        }[type(constraint.value)]
        return {
            "kind": constraint.kind,
            "subject": constraint.subject.to_dict(),
            "operator": constraint.operator,
            "value_type": value_type,
            "value_present": constraint.value is not None,
        }
    if type(constraint) is ComponentCountConstraint:
        return {
            "kind": constraint.kind,
            "component_kind": constraint.component_kind,
            "plugin_kind": constraint.plugin_kind,
            "plugin_name": constraint.plugin_name,
            "operator": constraint.operator,
            "count": constraint.count,
        }
    if type(constraint) is EdgeRouteConstraint:
        return {
            "kind": constraint.kind,
            "from_subject": constraint.from_subject.to_dict(),
            "edge_type": constraint.edge_type,
            "to_subject": constraint.to_subject.to_dict(),
            "present": constraint.present,
        }
    if type(constraint) is FailureRouteConstraint:
        return {
            "kind": constraint.kind,
            "subject": constraint.subject.to_dict(),
            "failure_kind": constraint.failure_kind,
            "operator": constraint.operator,
            "target": constraint.target if constraint.target == "discard" else constraint.target.to_dict(),
        }
    raise AuditIntegrityError("guided deferred constraint is outside the provider-safe closed projection")


def guided_redacted_planner_context(guided: GuidedSession) -> dict[str, object]:
    """Build the closed provider-visible summary without option values or rows."""

    return {
        "schema": "guided.reviewed-planner-context.v1",
        "sources": [
            {
                "stable_id": stable_id,
                "plugin": source.plugin,
                "observed_columns": list(source.observed_columns),
                "option_keys": sorted(source.options),
                "on_validation_failure": source.on_validation_failure,
            }
            for stable_id in guided.source_order
            for source in (guided.reviewed_sources[stable_id],)
        ],
        "outputs": [
            {
                "stable_id": stable_id,
                "plugin": output.plugin,
                "required_fields": list(output.required_fields),
                "schema_mode": output.schema_mode,
                "option_keys": sorted(output.options),
                "on_write_failure": output.on_write_failure,
            }
            for stable_id in guided.output_order
            for output in (guided.reviewed_outputs[stable_id],)
        ],
        "deferred_intents": [
            {
                "intent_id": intent.intent_id,
                "target_stage": intent.target_stage,
                "catalog_kind": intent.catalog_kind,
                "catalog_name": intent.catalog_name,
                "redacted_summary": intent.redacted_summary,
                "constraints": [_provider_safe_deferred_constraint(constraint) for constraint in intent.constraints],
            }
            for intent in guided.deferred_intents
        ],
    }


def guided_redacted_current_state_context(state: CompositionState) -> dict[str, object]:
    """Return provider-visible topology without any open option values."""

    return {
        "schema": "guided.current-state-context.v1",
        "version": state.version,
        "sources": [
            {
                "name": name,
                "plugin": source.plugin,
                "option_keys": sorted(source.options),
                "on_success": source.on_success,
                "on_validation_failure": source.on_validation_failure,
            }
            for name, source in state.sources.items()
        ],
        "nodes": [
            {
                "id": node.id,
                "node_type": node.node_type,
                "plugin": node.plugin,
                "option_keys": sorted(node.options),
                "input": node.input,
                "on_success": node.on_success,
                "on_error": node.on_error,
            }
            for node in state.nodes
        ],
        "outputs": [
            {
                "name": output.name,
                "plugin": output.plugin,
                "option_keys": sorted(output.options),
                "on_write_failure": output.on_write_failure,
            }
            for output in state.outputs
        ],
    }


def bind_guided_reviewed_components(
    pipeline: Mapping[str, Any],
    guided: GuidedSession,
) -> GuidedBoundPipeline:
    """Replace provider-authored component configuration with reviewed authority.

    The planner remains responsible for topology.  Source and output plugin
    configuration was already reviewed by the operator, so those private
    values are restored server-side after the terminal model call and before
    candidate validation or proposal sealing.
    """

    bound = cast(dict[str, Any], deep_thaw(pipeline))
    raw_sources = bound.get("sources")
    if type(raw_sources) is not dict:
        singular = bound.get("source")
        if len(guided.source_order) != 1 or type(singular) is not dict:
            raise AuditIntegrityError("guided planner candidate does not identify reviewed sources")
        source_id = guided.source_order[0]
        source = guided.reviewed_sources[source_id]
        if singular.get("name", source.name) != source.name:
            raise AuditIntegrityError("guided planner candidate source name differs from reviewed authority")
        raw_sources = {source.name: singular}
        bound.pop("source", None)
    expected_source_names = [guided.reviewed_sources[stable_id].name for stable_id in guided.source_order]
    if list(raw_sources) != expected_source_names:
        raise AuditIntegrityError("guided planner candidate sources differ from reviewed authority")
    rebound_sources: dict[str, GuidedBoundSource] = {}
    for stable_id in guided.source_order:
        reviewed = guided.reviewed_sources[stable_id]
        candidate = raw_sources[reviewed.name]
        if type(candidate) is not dict or type(candidate.get("on_success")) is not str:
            raise AuditIntegrityError("guided planner candidate source topology is malformed")
        rebound_sources[reviewed.name] = {
            "plugin": reviewed.plugin,
            "options": deep_thaw(reviewed.options),
            "on_success": candidate["on_success"],
            "on_validation_failure": reviewed.on_validation_failure,
        }
    bound["sources"] = rebound_sources

    raw_outputs = bound.get("outputs")
    if type(raw_outputs) is not list:
        raise AuditIntegrityError("guided planner candidate outputs are malformed")
    expected_output_names = [guided.reviewed_outputs[stable_id].name for stable_id in guided.output_order]
    actual_output_names = [item.get("sink_name", item.get("name")) if type(item) is dict else None for item in raw_outputs]
    if actual_output_names != expected_output_names:
        raise AuditIntegrityError("guided planner candidate outputs differ from reviewed authority")
    rebound_outputs: list[GuidedBoundOutput] = []
    for index, stable_id in enumerate(guided.output_order):
        reviewed_output = guided.reviewed_outputs[stable_id]
        candidate = raw_outputs[index]
        assert type(candidate) is dict
        rebound_outputs.append(
            {
                "sink_name": reviewed_output.name,
                "plugin": reviewed_output.plugin,
                "options": deep_thaw(reviewed_output.options),
                "on_write_failure": reviewed_output.on_write_failure,
            }
        )
    bound["outputs"] = rebound_outputs
    return cast(GuidedBoundPipeline, bound)


def _state_from_proposal(proposal: PipelineProposal) -> CompositionState:
    raw = cast(dict[str, Any], deep_thaw(proposal.pipeline))
    if "source" in raw and "sources" not in raw:
        source = raw.pop("source")
        raw["sources"] = {"source": source} if source is not None else {}
    outputs = raw.get("outputs")
    if type(outputs) is list:
        for output in outputs:
            if type(output) is dict and "sink_name" in output:
                output["name"] = output.pop("sink_name")
    raw.setdefault("metadata", {"name": "Untitled Pipeline", "description": ""})
    raw["version"] = 1
    try:
        return CompositionState.from_dict(raw)
    except (KeyError, TypeError, ValueError) as exc:
        raise AuditIntegrityError("guided proposal private pipeline is not canonical") from exc


def guided_candidate_state(proposal: PipelineProposal) -> CompositionState:
    """Restore the immutable candidate named by a durable proposal.

    Wire review inspects this candidate, never the uncommitted composition
    checkpoint and never topology reconstructed from guided dialogue.
    """

    return _state_from_proposal(proposal)


def _component_target(kind: str, stable_id: str) -> dict[str, str]:
    return {"kind": kind, "stable_id": stable_id}


def _endpoint(kind: str, stable_id: str | None = None) -> dict[str, str]:
    result = {"kind": kind}
    if stable_id is not None:
        result["stable_id"] = stable_id
    return result


def _ordered_gate_routes(node: NodeSpec) -> tuple[tuple[str, str], ...]:
    """Return the protocol-canonical direct routes followed by fork routes."""

    assert node.node_type == "gate"
    routes = sorted((node.routes or {}).items(), key=lambda route: route[0])
    return (
        *((name, destination) for name, destination in routes if destination != "fork"),
        *((name, destination) for name, destination in routes if destination == "fork"),
    )


def _node_behavior(node: NodeSpec, *, route_aliases: Mapping[str, str], branch_aliases: Mapping[str, str]) -> dict[str, object]:
    if node.node_type == "transform":
        return {"kind": "transform"}
    if node.node_type == "queue":
        return {"kind": "queue"}
    if node.node_type == "aggregation":
        trigger = dict(deep_thaw(node.trigger or {}))
        # Preserve the executable scalar semantics, never free-form prose.
        trigger_kinds = [
            name for name in ("count", "timeout", "condition") if trigger.get(name if name != "timeout" else "timeout_seconds") is not None
        ]
        count = trigger.get("count")
        timeout_seconds = trigger.get("timeout_seconds")
        return {
            "kind": "aggregation",
            "trigger_kinds": trigger_kinds,
            "count": str(count) if count is not None else None,
            "timeout_seconds": float(timeout_seconds) if timeout_seconds is not None else None,
            "output_mode": node.output_mode,
            "expected_output_count": (str(node.expected_output_count) if node.expected_output_count is not None else None),
        }
    if node.node_type == "coalesce":
        branches = node.branches
        names = list(branches.values()) if isinstance(branches, Mapping) else list(branches or ())
        return {
            "kind": "coalesce",
            "branch_aliases": [branch_aliases[name] for name in names],
            "policy": node.policy,
            "merge": node.merge,
        }
    assert node.node_type == "gate"
    routes = _ordered_gate_routes(node)
    route_names = [name for name, _destination in routes]
    fork_routes = [name for name, destination in routes if destination == "fork"]
    fork_to = list(node.fork_to or ())
    return {
        "kind": "gate",
        "route_aliases": [route_aliases[name] for name in route_names],
        "fork_branches": [
            {
                "routes": [route_aliases[name] for name in fork_routes],
                "branch": branch_aliases[destination],
            }
            for destination in fork_to
        ],
    }


def _projection_ids_from_payload(payload: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    nodes = payload.get("nodes")
    graph = payload.get("graph")
    if type(nodes) is not list or type(graph) is not dict or type(graph.get("edges")) is not list:
        raise AuditIntegrityError("guided proposal projection has malformed stable-id containers")
    node_ids = [node.get("stable_id") for node in nodes if type(node) is dict]
    edge_ids = [edge.get("stable_id") for edge in graph["edges"] if type(edge) is dict]
    if len(node_ids) != len(nodes) or len(edge_ids) != len(graph["edges"]):
        raise AuditIntegrityError("guided proposal projection has malformed stable IDs")
    return cast(list[str], node_ids), cast(list[str], edge_ids)


def _build_projection(
    *,
    proposal_id: UUID,
    proposal: PipelineProposal,
    guided: GuidedSession,
    catalog_plugin_ids: Mapping[str, frozenset[str]],
    node_stable_ids: Sequence[str] | None,
    edge_stable_ids: Sequence[str] | None,
) -> ProposePipelinePayload:
    state = _state_from_proposal(proposal)
    if [state.sources[name].plugin for name in state.sources] != [
        guided.reviewed_sources[stable_id].plugin for stable_id in guided.source_order
    ]:
        raise AuditIntegrityError("guided proposal sources differ from reviewed authority")
    if [output.plugin for output in state.outputs] != [guided.reviewed_outputs[stable_id].plugin for stable_id in guided.output_order]:
        raise AuditIntegrityError("guided proposal outputs differ from reviewed authority")
    if list(state.sources) != [guided.reviewed_sources[stable_id].name for stable_id in guided.source_order]:
        raise AuditIntegrityError("guided proposal source names differ from reviewed authority")
    if [output.name for output in state.outputs] != [guided.reviewed_outputs[stable_id].name for stable_id in guided.output_order]:
        raise AuditIntegrityError("guided proposal output names differ from reviewed authority")

    resolved_node_ids = list(node_stable_ids or (str(uuid4()) for _ in state.nodes))
    if len(resolved_node_ids) != len(state.nodes):
        raise AuditIntegrityError("guided proposal projection node stable-id count mismatch")
    node_ids = {node.id: resolved_node_ids[index] for index, node in enumerate(state.nodes)}
    source_ids = {name: guided.source_order[index] for index, name in enumerate(state.sources)}
    output_ids = {output.name: guided.output_order[index] for index, output in enumerate(state.outputs)}

    route_keys: list[tuple[str, str]] = []
    branch_names: list[str] = []
    for node in state.nodes:
        routes = _ordered_gate_routes(node) if node.node_type == "gate" else ()
        for route, _destination in routes:
            route_keys.append((node_ids[node.id], route))
        for branch in node.fork_to or ():
            if branch not in branch_names:
                branch_names.append(branch)
        raw_branches = node.branches
        values = list(raw_branches.values()) if isinstance(raw_branches, Mapping) else list(raw_branches or ())
        for branch in values:
            if branch not in branch_names:
                branch_names.append(branch)
    route_aliases = {key: proposal_structural_label("route", index) for index, key in enumerate(route_keys)}
    branch_aliases = {name: proposal_structural_label("branch", index) for index, name in enumerate(branch_names)}

    def gate_route_aliases(node: NodeSpec) -> dict[str, str]:
        assert node.node_type == "gate"
        return {route: route_aliases[(node_ids[node.id], route)] for route, _destination in _ordered_gate_routes(node)}

    try:
        consumers = canonical_connection_consumers(
            state,
            node_identities=node_ids,
            output_identities=output_ids,
        )
    except ValueError as exc:  # pragma: no cover - validated state and exact IDs own this invariant
        raise AuditIntegrityError("guided proposal canonical consumer identities are malformed") from exc

    edge_specs: list[tuple[dict[str, str], dict[str, str], dict[str, object]]] = []

    # A queue's connection name is its own id (``queue_node_contract_error``
    # enforces ``input == id``), so ``canonical_connection_consumers`` lists both
    # the queue itself (input side) and its one ordinary downstream node
    # (republish side) as consumers of that connection. Those two sides must be
    # separated in the wire projection: an external producer publishing into the
    # connection reaches only the fan-in point, while the queue's own
    # ``queue_continue`` republishes to the downstream node — never back to
    # itself. Collapsing them would either self-loop the queue or fan a producer
    # straight past it (elspeth-a5b86149d4).
    queue_stable_by_connection = {node.id: node_ids[node.id] for node in state.nodes if node.node_type == "queue"}

    def add_targets(origin: dict[str, str], connection: str | None, flow: dict[str, object]) -> None:
        if connection is None:
            return
        if connection == "discard":
            edge_specs.append((origin, _endpoint("discard"), flow))
            return
        destinations = consumers.get(connection, ())
        queue_stable = queue_stable_by_connection.get(connection)
        if queue_stable is not None:
            if origin.get("stable_id") == queue_stable:
                destinations = tuple(dest for dest in destinations if dest != ("node", queue_stable))
            else:
                destinations = (("node", queue_stable),)
        if not destinations:
            raise AuditIntegrityError("guided proposal connection has no canonical consumer")
        for kind, stable_id in destinations:
            edge_specs.append((origin, _endpoint(kind, stable_id), flow))

    for source_name, source in state.sources.items():
        origin = _endpoint("source", source_ids[source_name])
        add_targets(origin, source.on_success, {"kind": "source_success", "branch": None})
        add_targets(origin, source.on_validation_failure, {"kind": "source_validation_failure"})

    for node in state.nodes:
        origin = _endpoint("node", node_ids[node.id])
        if node.node_type == "gate":
            routes = _ordered_gate_routes(node)
            node_route_aliases = gate_route_aliases(node)
            fork_routes = [name for name, destination in routes if destination == "fork"]
            for route, destination in routes:
                if destination == "fork":
                    continue
                add_targets(
                    origin,
                    destination,
                    {"kind": "gate_route", "route": node_route_aliases[route], "branch": None},
                )
            for destination in node.fork_to or ():
                add_targets(
                    origin,
                    destination,
                    {
                        "kind": "gate_fork",
                        "routes": [node_route_aliases[route] for route in fork_routes],
                        "branch": branch_aliases[destination],
                    },
                )
        elif node.node_type == "queue":
            add_targets(origin, node.id, {"kind": "queue_continue", "branch": None})
        elif node.node_type == "coalesce":
            add_targets(origin, node.on_success, {"kind": "coalesce_success", "branch": None})
        else:
            add_targets(origin, node.on_success, {"kind": "node_success", "branch": None})
            add_targets(origin, node.on_error, {"kind": "node_error"})

    for output in state.outputs:
        add_targets(
            _endpoint("output", output_ids[output.name]),
            output.on_write_failure,
            {"kind": "output_write_failure"},
        )

    resolved_edge_ids = list(edge_stable_ids or (str(uuid4()) for _ in edge_specs))
    if len(resolved_edge_ids) != len(edge_specs):
        raise AuditIntegrityError("guided proposal projection edge stable-id count mismatch")
    edges: list[dict[str, Any]] = [
        {
            "stable_id": resolved_edge_ids[index],
            "from_endpoint": origin,
            "to_endpoint": destination,
            "flow": flow,
        }
        for index, (origin, destination, flow) in enumerate(edge_specs)
    ]
    nodes: list[dict[str, Any]] = [
        {
            "stable_id": node_ids[node.id],
            "label": proposal_component_label("node", index),
            "node_type": node.node_type,
            "plugin": ({"kind": "transform", "id": node.plugin} if node.plugin is not None else None),
            "behavior": _node_behavior(
                node,
                route_aliases=gate_route_aliases(node) if node.node_type == "gate" else {},
                branch_aliases=branch_aliases,
            ),
        }
        for index, node in enumerate(state.nodes)
    ]
    sources: list[dict[str, Any]] = [
        {
            "stable_id": source_ids[name],
            "label": proposal_component_label("source", index),
            "plugin": {"kind": "source", "id": source.plugin},
        }
        for index, (name, source) in enumerate(state.sources.items())
    ]
    outputs: list[dict[str, Any]] = [
        {
            "stable_id": output_ids[output.name],
            "label": proposal_component_label("output", index),
            "plugin": {"kind": "sink", "id": output.plugin},
        }
        for index, output in enumerate(state.outputs)
    ]
    payload = cast(
        ProposePipelinePayload,
        {
            "proposal_id": str(proposal_id),
            "draft_hash": proposal.draft_hash,
            "summary": PROPOSAL_SUMMARY_TEMPLATE,
            "rationale": PROPOSAL_RATIONALE_TEMPLATE,
            "component_counts": {
                "sources": len(sources),
                "nodes": len(nodes),
                "edges": len(edges),
                "outputs": len(outputs),
            },
            "blockers": [],
            "graph": {"sources": sources, "edges": edges},
            "nodes": nodes,
            "outputs": outputs,
            "edit_targets": [
                *(_component_target("source", source["stable_id"]) for source in sources),
                *(_component_target("node", node["stable_id"]) for node in nodes),
                *(_component_target("edge", edge["stable_id"]) for edge in edges),
                *(_component_target("output", output["stable_id"]) for output in outputs),
            ],
        },
    )
    if (error := validate_payload(TurnType.PROPOSE_PIPELINE, payload)) is not None:
        raise AuditIntegrityError(f"guided proposal projection is invalid: {error}")
    if (error := validate_proposal_catalog_refs(payload, catalog_plugin_ids)) is not None:
        raise AuditIntegrityError(f"guided proposal catalog binding failed: {error}")
    return payload


def build_guided_proposal_projection(
    *,
    proposal_id: UUID,
    proposal: PipelineProposal,
    guided: GuidedSession,
    catalog_plugin_ids: Mapping[str, frozenset[str]],
) -> ProposePipelinePayload:
    """Allocate and return one safe immutable-proposal projection."""

    return _build_projection(
        proposal_id=proposal_id,
        proposal=proposal,
        guided=guided,
        catalog_plugin_ids=catalog_plugin_ids,
        node_stable_ids=None,
        edge_stable_ids=None,
    )


def verify_guided_proposal_projection(
    *,
    payload: Mapping[str, Any],
    proposal_id: UUID,
    proposal: PipelineProposal,
    guided: GuidedSession,
    catalog_plugin_ids: Mapping[str, frozenset[str]],
) -> None:
    """Recompute all safe semantics while retaining persisted stable IDs."""

    node_ids, edge_ids = _projection_ids_from_payload(payload)
    expected = _build_projection(
        proposal_id=proposal_id,
        proposal=proposal,
        guided=guided,
        catalog_plugin_ids=catalog_plugin_ids,
        node_stable_ids=node_ids,
        edge_stable_ids=edge_ids,
    )
    if deep_thaw(payload) != expected:
        raise AuditIntegrityError("guided proposal projection differs from private proposal authority")


def verified_remaining_deferred_intents(
    *,
    guided: GuidedSession,
    proposal: PipelineProposal,
) -> tuple[DeferredStageIntent, ...]:
    """Verify mechanically covered constraints and return the exact remainder."""

    state = _state_from_proposal(proposal)
    try:
        covered_ordered = evaluate_deferred_intent_coverage(
            candidate=state,
            reviewed_guided=guided,
            claimed_intent_ids=proposal.covered_deferred_intent_ids,
        )
    except DeferredIntentClaimError as exc:
        raise AuditIntegrityError("guided proposal does not mechanically satisfy a covered deferred constraint") from exc
    covered = set(covered_ordered)
    return tuple(intent for intent in guided.deferred_intents if intent.intent_id not in covered)


__all__ = [
    "GuidedCorrectionTarget",
    "bind_guided_reviewed_components",
    "build_guided_proposal_projection",
    "guided_candidate_state",
    "guided_private_reviewed_facts",
    "guided_redacted_current_state_context",
    "guided_redacted_planner_context",
    "require_guided_correction_target_changed",
    "resolve_guided_correction_target",
    "verified_remaining_deferred_intents",
    "verify_guided_proposal_projection",
]
