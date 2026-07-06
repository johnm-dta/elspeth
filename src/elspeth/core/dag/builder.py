"""DAG construction from plugin instances.

Extracts the graph-building logic from ExecutionGraph.from_plugin_instances()
into a module-level function. The classmethod facade on ExecutionGraph delegates
here via lazy import to avoid circular dependencies.

Dependency: models.py (leaf) — no import of graph.py at module level.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from elspeth.contracts import RouteDestination, RoutingMode, error_edge_label
from elspeth.contracts.enums import NodeType
from elspeth.contracts.errors import FrameworkBugError
from elspeth.contracts.schema import SchemaConfig, get_raw_schema_config
from elspeth.contracts.types import (
    AggregationName,
    BranchName,
    CoalesceName,
    GateName,
    NodeID,
    SinkName,
)
from elspeth.core.canonical import canonical_json
from elspeth.core.dag.coalesce_merge import merge_coalesce_schema
from elspeth.core.dag.models import (
    _NODE_ID_MAX_LENGTH,
    BranchInfo,
    GraphValidationError,
    _GateEntry,
    _suggest_similar,
)

if TYPE_CHECKING:
    from elspeth.contracts import SinkProtocol, SourceProtocol, TransformProtocol
    from elspeth.core.config import (
        AggregationSettings,
        CoalesceSettings,
        GateSettings,
        QueueSettings,
        SourceSettings,
    )
    from elspeth.core.dag.graph import ExecutionGraph
    from elspeth.core.dag.models import NodeConfig, WiredTransform


@dataclass(frozen=True, slots=True)
class _CoalesceBranchSpec:
    branch_name: BranchName
    coalesce_name: CoalesceName
    coalesce_node_id: NodeID
    input_connection: str
    uses_transform_chain: bool


@dataclass(frozen=True, slots=True)
class _CoalesceBranchPlan:
    branch_name: BranchName
    coalesce_name: CoalesceName
    coalesce_node_id: NodeID
    gate_name: GateName
    gate_node_id: NodeID
    input_connection: str
    uses_transform_chain: bool

    @classmethod
    def from_spec(cls, spec: _CoalesceBranchSpec, *, gate_name: GateName, gate_node_id: NodeID) -> _CoalesceBranchPlan:
        return cls(
            branch_name=spec.branch_name,
            coalesce_name=spec.coalesce_name,
            coalesce_node_id=spec.coalesce_node_id,
            gate_name=gate_name,
            gate_node_id=gate_node_id,
            input_connection=spec.input_connection,
            uses_transform_chain=spec.uses_transform_chain,
        )

    def to_branch_info(self) -> BranchInfo:
        return BranchInfo(
            coalesce_name=self.coalesce_name,
            gate_node_id=self.gate_node_id,
            input_connection=self.input_connection,
            uses_transform_chain=self.uses_transform_chain,
        )


@dataclass(frozen=True, slots=True)
class _CoalescePlan:
    name: CoalesceName
    node_id: NodeID
    branches: tuple[_CoalesceBranchSpec, ...]


def _validate_output_schema_contract(transform: Any) -> None:
    """Validate consistency between declared_output_fields and _output_schema_config.

    Two-directional check:
    1. Forward: declared_output_fields non-empty → _output_schema_config must exist.
    2. Containment: declared_output_fields ⊆ guaranteed_fields when both are set.

    Raises FrameworkBugError on any contract violation.
    """
    declared = transform.declared_output_fields
    config = transform._output_schema_config

    # Forward: declares fields but no schema contract → silent DAG validation gap.
    if declared and config is None:
        raise FrameworkBugError(
            f"Transform {transform.name!r} declares output fields "
            f"{sorted(declared)} but provides no "
            f"_output_schema_config for DAG contract validation. "
            f"Call self._output_schema_config = self._build_output_schema_config(schema_config) "
            f"in __init__ after setting declared_output_fields."
        )

    # Containment: declared fields must appear in effective guaranteed fields.
    # Uses get_effective_guaranteed_fields() rather than raw guaranteed_fields
    # to include implicit guarantees from fixed/flexible mode declared fields.
    # Without this, collision detection checks fields that the DAG contract
    # doesn't guarantee — downstream required_fields validation has a blind spot.
    if declared and config is not None and config.declares_guaranteed_fields:
        effective = config.get_effective_guaranteed_fields()
        missing = set(declared) - effective
        if missing:
            raise FrameworkBugError(
                f"Transform {transform.name!r} declares output fields "
                f"{sorted(missing)} not present in effective guaranteed fields "
                f"{sorted(effective)}. "
                f"declared_output_fields must be a subset of guaranteed_fields."
            )


def _parse_contract_schema_config(
    config: Mapping[str, Any],
    *,
    owner: str,
    component_id: str,
    component_type: str,
) -> SchemaConfig | None:
    """Parse a node schema config using the shared raw-option rules."""
    try:
        return get_raw_schema_config(config, owner=owner)
    except ValueError as exc:
        raise GraphValidationError(
            f"Invalid schema config: {exc}",
            component_id=component_id,
            component_type=component_type,
        ) from exc


def build_execution_graph(
    cls: type[ExecutionGraph],
    *,
    sources: Mapping[str, SourceProtocol],
    source_settings_map: Mapping[str, SourceSettings],
    transforms: Sequence[WiredTransform] = (),
    sinks: Mapping[str, SinkProtocol] | None = None,
    aggregations: Mapping[str, tuple[TransformProtocol, AggregationSettings]] | None = None,
    gates: Sequence[GateSettings] = (),
    coalesce_settings: Sequence[CoalesceSettings] | None = None,
    queues: Mapping[str, QueueSettings] | None = None,
) -> ExecutionGraph:
    """Build an ExecutionGraph from plugin instances.

    Called by ExecutionGraph.from_plugin_instances() facade. See that method
    for full documentation of parameters and semantics.

    Per ADR-025 §2 the source surface is plural-only — callers pass
    ``sources`` and ``source_settings_map`` keyed by source name. The
    pre-ADR singular ``source=`` / ``source_settings=`` keyword shim
    and its ``legacy_single_source_invocation`` branch are deleted.
    """
    if not sources:
        raise GraphValidationError("ExecutionGraph requires at least one source")
    if sinks is None:
        raise GraphValidationError("ExecutionGraph requires at least one sink")
    if aggregations is None:
        aggregations = {}
    if set(sources) != set(source_settings_map):
        raise GraphValidationError(
            f"Source plugin names and source settings names must match. plugins={sorted(sources)}, settings={sorted(source_settings_map)}"
        )

    queue_settings = queues or {}
    graph = cls()

    def node_id(prefix: str, name: str, config: NodeConfig, sequence: int | None = None) -> NodeID:
        """Generate deterministic node ID based on plugin type and config.

        Node IDs must be deterministic for checkpoint/resume compatibility.
        If a pipeline is checkpointed and later resumed, the node IDs must
        be identical so checkpoint state can be restored correctly.

        For nodes that can appear multiple times with identical configs
        (transforms, aggregations), include sequence number to ensure uniqueness.

        Args:
            prefix: Node type prefix (source_, transform_, sink_, etc.)
            name: Plugin name
            config: Plugin configuration dict
            sequence: Optional sequence number for duplicate configs (transforms, aggregations)

        Returns:
            Deterministic node ID
        """
        # Create stable hash of config using RFC 8785 canonical JSON
        # CRITICAL: Must use canonical_json() not json.dumps() for true determinism
        # (floats, nested dicts, datetime serialization must be consistent)
        config_str = canonical_json(config)
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:12]  # 48 bits

        # Include sequence number for nodes that can have duplicates
        if sequence is not None:
            generated = f"{prefix}_{name}_{config_hash}_{sequence}"
        else:
            generated = f"{prefix}_{name}_{config_hash}"

        if len(generated) > _NODE_ID_MAX_LENGTH:
            raise GraphValidationError(
                f"Generated node_id exceeds {_NODE_ID_MAX_LENGTH} characters: "
                f"'{generated}' (length={len(generated)}). "
                "Use shorter transform/gate/aggregation/source/sink names.",
                component_id=name,
                component_type=prefix,
            )

        return NodeID(generated)

    def _best_schema_config(nid: NodeID) -> SchemaConfig:
        """Get SchemaConfig from a node.

        All nodes have output_schema_config populated at construction time
        (sources, transforms, aggregations from config; gates and coalesce
        from upstream inheritance via _assign_schema).
        """
        info = graph.get_node_info(nid)
        if info.output_schema_config is None:
            raise FrameworkBugError(
                f"Node '{nid}' has no output_schema_config. "
                "All producer nodes must have output_schema_config populated "
                "at construction time."
            )
        return info.output_schema_config

    def _assign_schema(target_nid: NodeID, schema: SchemaConfig) -> None:
        """Set output_schema_config on a pass-through node (gate or coalesce).

        Pass-through nodes don't have their own schema — they inherit from
        upstream producers. This sets the typed SchemaConfig so all consumers
        can read it directly without fallback chains.
        """
        graph.set_node_output_schema(target_nid, schema)

    def _sink_name_set() -> set[str]:
        return {str(name) for name in sink_ids}

    # Add sources. Per ADR-025 §2 source node identity always includes the
    # configured source name in the config hash so two instances of the same
    # plugin remain distinct DAG roots and audit records. There is no
    # singular checkpoint-identity reservation; the prior "source" literal
    # name shortcut is gone with the legacy facade.
    source_ids: dict[str, NodeID] = {}
    for source_name, source_instance in sources.items():
        source_config = source_instance.config
        source_schema_config = _parse_contract_schema_config(
            source_config,
            owner=f"source:{source_name}",
            component_id=source_name,
            component_type="source",
        )
        source_node_config = dict(source_config)
        source_node_config["source_name"] = source_name
        source_id = node_id("source", source_name, source_node_config)
        source_ids[source_name] = source_id
        graph.add_node(
            source_id,
            node_type=NodeType.SOURCE,
            plugin_name=source_instance.name,
            config=source_node_config,
            output_schema=source_instance.output_schema,  # SourceProtocol requires this
            output_schema_config=source_schema_config,
        )

    # Add sinks
    sink_ids: dict[SinkName, NodeID] = {}
    for sink_name, sink in sinks.items():
        sink_config = sink.config
        sid = node_id("sink", sink_name, sink_config)
        sink_ids[SinkName(sink_name)] = sid
        sink_schema_config = _parse_contract_schema_config(
            sink_config,
            owner=f"sink:{sink_name}",
            component_id=sink_name,
            component_type="sink",
        )
        graph.add_node(
            sid,
            node_type=NodeType.SINK,
            plugin_name=sink.name,
            config=sink_config,
            input_schema=sink.input_schema,  # SinkProtocol requires this
            output_schema_config=sink_schema_config,
            declared_required_fields=sink.declared_required_fields,
        )

    graph.set_sink_id_map(sink_ids)

    # Build declared scheduling queues. V1 queue semantics are pass-through
    # coordination only: queues do not merge fields or synthesize guarantees
    # across sources, so their schema contract is deliberately observed.
    queue_ids: dict[str, NodeID] = {}
    observed_queue_schema = SchemaConfig(mode="observed", fields=None)
    for queue_name, queue_config in queue_settings.items():
        queue_node_config: NodeConfig = {"name": queue_name}
        if queue_config.description is not None:
            queue_node_config["description"] = queue_config.description
        qid = node_id("queue", queue_name, queue_node_config)
        queue_ids[queue_name] = qid
        graph.add_node(
            qid,
            node_type=NodeType.QUEUE,
            plugin_name=f"queue:{queue_name}",
            config=queue_node_config,
            output_schema_config=observed_queue_schema,
        )

    # Build transforms
    transform_ids_by_name: dict[str, NodeID] = {}
    transform_ids_by_seq: dict[int, NodeID] = {}
    gate_entries: list[_GateEntry] = []
    gate_route_connections: list[tuple[NodeID, str, str]] = []

    for seq, wired in enumerate(transforms):
        transform = wired.plugin
        transform_config = transform.config
        tid = node_id("transform", wired.settings.name, transform_config)
        transform_ids_by_name[wired.settings.name] = tid
        transform_ids_by_seq[seq] = tid

        node_config = dict(transform_config)
        node_type = NodeType.TRANSFORM

        # Validate output schema contract — crash if transform declares output
        # fields but provides no DAG contract.
        _validate_output_schema_contract(transform)
        output_schema_config = transform._output_schema_config

        # Shape-preserving transforms don't compute _output_schema_config.
        # Parse the raw schema config so every node has a typed schema.
        if output_schema_config is None:
            output_schema_config = _parse_contract_schema_config(
                transform_config,
                owner=f"transform:{wired.settings.name}",
                component_id=wired.settings.name,
                component_type="transform",
            )

        graph.add_node(
            tid,
            node_type=node_type,
            plugin_name=transform.name,
            config=node_config,
            input_schema=transform.input_schema,  # TransformProtocol requires this
            output_schema=transform.output_schema,  # TransformProtocol requires this
            output_schema_config=output_schema_config,
            passes_through_input=transform.passes_through_input,
        )

    graph.set_transform_id_map(transform_ids_by_seq)

    # Build aggregations
    aggregation_ids: dict[AggregationName, NodeID] = {}
    for agg_name, (transform, agg_config) in aggregations.items():
        transform_config = transform.config
        # Use "input_schema" (not "schema") so add_node() doesn't auto-populate
        # output_schema_config. Aggregations have dynamic output by design —
        # BatchStats produces count/sum/mean, not the input fields. The key is
        # preserved for audit/hashing but doesn't trigger output schema inference.
        # See elspeth-c3a98c358c.
        agg_node_config = {
            "trigger": agg_config.trigger.model_dump(),
            "output_mode": agg_config.output_mode,
            "options": dict(agg_config.options),
            "input_schema": transform_config["schema"],  # Input validation, not output
        }
        aid = node_id("aggregation", agg_name, agg_node_config)
        aggregation_ids[AggregationName(agg_name)] = aid

        # Aggregations have dynamic output by design — BatchStats produces
        # count/sum/mean, not the input fields. But _output_schema_config IS
        # correct: _build_output_schema_config() merges declared_output_fields
        # into guaranteed_fields and preserves required_fields (for derived
        # input requirements like group_by). Downstream pass-through nodes
        # (gates, coalesce branches) need output_schema_config for _best_schema_config().
        #
        # Fallback to the raw schema config for test fixtures that don't
        # compute _output_schema_config (same pattern as transforms above).
        agg_output_schema_config = transform._output_schema_config
        if agg_output_schema_config is None:
            agg_output_schema_config = _parse_contract_schema_config(
                transform_config,
                owner=f"aggregation:{agg_name}",
                component_id=agg_name,
                component_type="aggregation",
            )

        graph.add_node(
            aid,
            node_type=NodeType.AGGREGATION,
            plugin_name=agg_config.plugin,
            config=agg_node_config,
            input_schema=transform.input_schema,
            output_schema=transform.output_schema,
            output_schema_config=agg_output_schema_config,
            passes_through_input=transform.passes_through_input,
        )

    graph.set_aggregation_id_map(aggregation_ids)

    # Build config gates (no plugin instances)
    config_gate_ids: dict[GateName, NodeID] = {}
    config_gate_schema_inputs: list[tuple[NodeID, str, str]] = []

    for gate_config in gates:
        gate_node_config = {
            "condition": gate_config.condition,
            "routes": dict(gate_config.routes),
        }
        if gate_config.fork_to:
            gate_node_config["fork_to"] = list(gate_config.fork_to)

        gid = node_id("config_gate", gate_config.name, gate_node_config)
        config_gate_ids[GateName(gate_config.name)] = gid

        graph.add_node(
            gid,
            node_type=NodeType.GATE,
            plugin_name=f"config_gate:{gate_config.name}",
            config=gate_node_config,
        )

        config_gate_schema_inputs.append((gid, gate_config.name, gate_config.input))

        # Gate routes to fork/sinks immediately. Connection-name routes are
        # deferred until the consumer registry exists. A literal "discard"
        # route is also deferred unless a real sink by that name exists, so a
        # real connection named "discard" can win before the virtual-drop
        # sentinel fallback is applied.
        for route_label, target in gate_config.routes.items():
            if target == "fork":
                # Fork is a special routing mode - handled by fork_to branches
                graph.add_route_resolution_entry(gid, route_label, RouteDestination.fork())
            elif SinkName(target) in sink_ids:
                target_sink_id = sink_ids[SinkName(target)]
                graph.add_edge(gid, target_sink_id, label=route_label, mode=RoutingMode.MOVE)
                graph.add_route_label_entry(gid, SinkName(target), route_label)
                graph.add_route_resolution_entry(gid, route_label, RouteDestination.sink(SinkName(target)))
            else:
                gate_route_connections.append((gid, route_label, target))

        gate_entries.append(
            _GateEntry(
                node_id=gid,
                name=gate_config.name,
                fork_to=tuple(gate_config.fork_to) if gate_config.fork_to is not None else None,
                routes=dict(gate_config.routes),
            )
        )

    graph.set_config_gate_id_map(config_gate_ids)

    # ===== COALESCE IMPLEMENTATION (BUILD NODES AND MAPPINGS FIRST) =====
    # Build coalesce nodes BEFORE connecting gates (needed for branch routing)
    coalesce_ids: dict[CoalesceName, NodeID] = {}
    coalesce_branch_specs: dict[BranchName, _CoalesceBranchSpec] = {}
    coalesce_plans: dict[CoalesceName, _CoalescePlan] = {}
    if coalesce_settings:
        for coalesce_config in coalesce_settings:
            coalesce_name = CoalesceName(coalesce_config.name)
            # Coalesce merges - no schema transformation
            # Note: Pydantic validates min_length=2 for branches field
            config_dict: NodeConfig = {
                "branches": dict(coalesce_config.branches),
                "policy": coalesce_config.policy,
                "merge": coalesce_config.merge,
            }
            if coalesce_config.timeout_seconds is not None:
                config_dict["timeout_seconds"] = coalesce_config.timeout_seconds
            if coalesce_config.quorum_count is not None:
                config_dict["quorum_count"] = coalesce_config.quorum_count
            if coalesce_config.select_branch is not None:
                config_dict["select_branch"] = coalesce_config.select_branch

            cid = node_id("coalesce", coalesce_config.name, config_dict)
            coalesce_ids[coalesce_name] = cid

            # Map branches to this coalesce - check for duplicates
            branch_specs: list[_CoalesceBranchSpec] = []
            for branch_name, input_connection in coalesce_config.branches.items():
                branch_key = BranchName(branch_name)
                if branch_key in coalesce_branch_specs:
                    # Branch already mapped to another coalesce
                    existing_coalesce = coalesce_branch_specs[branch_key].coalesce_name
                    raise GraphValidationError(
                        f"Duplicate branch name '{branch_name}' found in coalesce settings.\n"
                        f"Branch '{branch_name}' is already mapped to coalesce '{existing_coalesce}', "
                        f"but coalesce '{coalesce_config.name}' also declares it.\n"
                        f"Each fork branch can only merge at one coalesce point.",
                        component_id=coalesce_config.name,
                        component_type="coalesce",
                    )
                spec = _CoalesceBranchSpec(
                    branch_name=branch_key,
                    coalesce_name=coalesce_name,
                    coalesce_node_id=cid,
                    input_connection=input_connection,
                    uses_transform_chain=input_connection != branch_name,
                )
                coalesce_branch_specs[branch_key] = spec
                branch_specs.append(spec)
            coalesce_plans[coalesce_name] = _CoalescePlan(
                name=coalesce_name,
                node_id=cid,
                branches=tuple(branch_specs),
            )

            graph.add_node(
                cid,
                node_type=NodeType.COALESCE,
                plugin_name=f"coalesce:{coalesce_config.name}",
                config=config_dict,
            )

        graph.set_coalesce_id_map(coalesce_ids)

    # ===== CONNECT FORK GATES - EXPLICIT DESTINATIONS ONLY =====
    # CRITICAL: No fallback behavior. All fork branches must have explicit destinations.
    # This prevents silent configuration bugs (typos, missing destinations).
    fork_branch_owner: dict[BranchName, GateName] = {}
    coalesce_branch_plans: dict[BranchName, _CoalesceBranchPlan] = {}
    for gate_entry in gate_entries:
        if gate_entry.fork_to:
            branch_counts = Counter(gate_entry.fork_to)
            duplicates = sorted([branch for branch, count in branch_counts.items() if count > 1])
            if duplicates:
                raise GraphValidationError(
                    f"Gate '{gate_entry.name}' has duplicate fork branches: {duplicates}. Each fork branch name must be unique.",
                    component_id=gate_entry.name,
                    component_type="gate",
                )
            for branch_name in gate_entry.fork_to:
                branch_key = BranchName(branch_name)
                if branch_key in fork_branch_owner:
                    raise GraphValidationError(
                        f"Fork branch '{branch_name}' is declared by multiple gates: "
                        f"'{fork_branch_owner[branch_key]}' and '{gate_entry.name}'. "
                        "Fork branch names must be globally unique across all gates.",
                        component_id=gate_entry.name,
                        component_type="gate",
                    )
                fork_branch_owner[branch_key] = GateName(gate_entry.name)
                if branch_key in coalesce_branch_specs:
                    plan = _CoalesceBranchPlan.from_spec(
                        coalesce_branch_specs[branch_key],
                        gate_name=GateName(gate_entry.name),
                        gate_node_id=gate_entry.node_id,
                    )
                    coalesce_branch_plans[branch_key] = plan
                    if not plan.uses_transform_chain:
                        # Identity branch: direct COPY edge (current behavior)
                        graph.add_edge(
                            gate_entry.node_id,
                            plan.coalesce_node_id,
                            label=branch_name,
                            mode=RoutingMode.COPY,
                        )
                elif SinkName(branch_name) in sink_ids:
                    # Explicit sink destination (branch name matches sink name)
                    graph.add_edge(
                        gate_entry.node_id,
                        sink_ids[SinkName(branch_name)],
                        label=branch_name,
                        mode=RoutingMode.COPY,
                    )
                else:
                    # NO FALLBACK - this is a configuration error
                    raise GraphValidationError(
                        f"Gate '{gate_entry.name}' has fork branch '{branch_name}' with no destination.\n"
                        f"Fork branches must either:\n"
                        f"  1. Be listed in a coalesce 'branches' dict/list, or\n"
                        f"  2. Match a sink name exactly\n"
                        f"\n"
                        f"Available coalesce branches: {sorted(coalesce_branch_specs.keys())}\n"
                        f"Available sinks: {sorted(sink_ids.keys())}",
                        component_id=gate_entry.name,
                        component_type="gate",
                    )

    # ===== VALIDATE COALESCE BRANCHES ARE PRODUCED BY GATES =====
    # All branches declared in coalesce settings must be produced by some fork gate
    if coalesce_branch_specs:
        for branch_name, spec in coalesce_branch_specs.items():
            if branch_name not in coalesce_branch_plans:
                raise GraphValidationError(
                    f"Coalesce '{spec.coalesce_name}' declares branch '{branch_name}', "
                    f"but no gate produces this branch.\n"
                    f"Branches must be listed in a gate's fork_to list to be valid.\n"
                    f"\n"
                    f"Branches produced by gates: {sorted(fork_branch_owner.keys()) if fork_branch_owner else '(none)'}\n"
                    f"Coalesce '{spec.coalesce_name}' expects branches: "
                    f"{sorted(branch.branch_name for branch in coalesce_plans[spec.coalesce_name].branches)}",
                    component_id=str(spec.coalesce_name),
                    component_type="coalesce",
                )

    # ===== BUILD PRODUCER REGISTRY =====
    producers: dict[str, tuple[NodeID, str]] = {}
    producer_desc: dict[str, str] = {}
    queue_input_edges: defaultdict[str, list[tuple[NodeID, str, str]]] = defaultdict(list)
    gate_connection_route_labels: defaultdict[tuple[NodeID, str], list[str]] = defaultdict(list)

    def register_producer(connection_name: str, node_id: NodeID, label: str, description: str) -> None:
        if connection_name in queue_ids:
            queue_input_edges[connection_name].append((node_id, label, description))
            return
        if connection_name in producers:
            existing_node, _existing_label = producers[connection_name]
            raise GraphValidationError(
                f"Duplicate producer for connection '{connection_name}': "
                f"{producer_desc[connection_name]} ({existing_node}) and {description} ({node_id}).",
                component_id=str(node_id),
            )
        producers[connection_name] = (node_id, label)
        producer_desc[connection_name] = description

    for source_name, source_settings_entry in source_settings_map.items():
        source_on_success = source_settings_entry.on_success
        if SinkName(source_on_success) not in sink_ids:
            register_producer(
                source_on_success,
                source_ids[source_name],
                "continue",
                f"source '{source_name}'",
            )

    for wired in transforms:
        tid = transform_ids_by_name[wired.settings.name]
        on_success = wired.settings.on_success
        if SinkName(on_success) not in sink_ids:
            register_producer(on_success, tid, "continue", f"transform '{wired.settings.name}'")

    for agg_name, (_transform, agg_settings) in aggregations.items():
        aid = aggregation_ids[AggregationName(agg_name)]
        if agg_settings.on_success is None:
            register_producer(agg_settings.name, aid, "continue", f"aggregation '{agg_settings.name}'")
        elif SinkName(agg_settings.on_success) not in sink_ids:
            register_producer(agg_settings.on_success, aid, "continue", f"aggregation '{agg_settings.name}'")

    if coalesce_settings:
        for coalesce_config in coalesce_settings:
            if coalesce_config.on_success is None:
                coalesce_id = coalesce_ids[CoalesceName(coalesce_config.name)]
                register_producer(
                    coalesce_config.name,
                    coalesce_id,
                    "continue",
                    f"coalesce '{coalesce_config.name}'",
                )

    for queue_name, queue_id in queue_ids.items():
        producers[queue_name] = (queue_id, "continue")
        producer_desc[queue_name] = f"queue '{queue_name}'"

    # Register fork branches as produced connections (only for branches with transforms).
    # Identity branches use direct COPY edges and don't need connection registration.
    for plan in coalesce_branch_plans.values():
        if not plan.uses_transform_chain:
            continue
        register_producer(
            plan.branch_name,
            plan.gate_node_id,
            plan.branch_name,
            f"fork branch '{plan.branch_name}' from gate '{plan.gate_name}'",
        )

    # ===== BUILD CONSUMER REGISTRY =====
    consumers: dict[str, NodeID] = {}
    consumer_claims: list[tuple[str, NodeID, str]] = []

    def register_consumer(connection_name: str, node_id: NodeID, description: str) -> None:
        consumer_claims.append((connection_name, node_id, description))
        if connection_name not in consumers:
            consumers[connection_name] = node_id

    for wired in transforms:
        register_consumer(
            wired.settings.input,
            transform_ids_by_name[wired.settings.name],
            f"transform '{wired.settings.name}'",
        )

    for agg_name, (_transform, agg_settings) in aggregations.items():
        register_consumer(
            agg_settings.input,
            aggregation_ids[AggregationName(agg_name)],
            f"aggregation '{agg_settings.name}'",
        )

    for gate_settings in gates:
        register_consumer(
            gate_settings.input,
            config_gate_ids[GateName(gate_settings.name)],
            f"gate '{gate_settings.name}'",
        )

    # Register coalesce nodes as consumers of transform branch input connections.
    # For transform branches, the coalesce consumes from the final transform's
    # output connection (not the branch name). The connection resolution system
    # will create MOVE edges through the transform chain automatically.
    for plan in coalesce_branch_plans.values():
        if not plan.uses_transform_chain:
            continue
        register_consumer(
            plan.input_connection,
            plan.coalesce_node_id,
            f"coalesce '{plan.coalesce_name}' branch '{plan.branch_name}'",
        )

    for gate_id, route_label, target in gate_route_connections:
        if target == "discard" and target not in consumers:
            # No real sink or consumer claimed this target. It remains the
            # virtual drop sentinel and must not create a dangling producer.
            continue

        gate_connection_key = (gate_id, target)
        gate_connection_route_labels[gate_connection_key].append(route_label)

        # Multiple routes from the same gate may converge to the same target
        # (e.g., {"true": "next_gate", "false": "next_gate"}). Only register
        # the producer once — the connection is the same regardless of which
        # route label was taken.
        if target in producers and producers[target][0] == gate_id:
            continue
        register_producer(target, gate_id, route_label, f"gate route '{route_label}' from '{gate_id}'")

    for queue_name, upstream_edges in queue_input_edges.items():
        if not upstream_edges:
            raise GraphValidationError(
                f"Queue '{queue_name}' has no upstream producers.",
                component_id=queue_name,
                component_type="queue",
            )
        queue_id = queue_ids[queue_name]
        for upstream_node_id, edge_label, _description in upstream_edges:
            graph.add_edge(upstream_node_id, queue_id, label=edge_label, mode=RoutingMode.MOVE)

    # ===== VALIDATE CONNECTION NAMESPACES =====
    cls._validate_connection_namespaces(
        producers=producers,
        consumers=consumers,
        consumer_claims=consumer_claims,
        sink_names=_sink_name_set(),
        check_dangling=False,
    )

    # Config gate schema resolution (pass 1): resolve gates whose upstream
    # producer already has a schema. Gates downstream of coalesce nodes are
    # deferred to pass 2 (after coalesce schema population).
    deferred_config_gate_schemas: list[tuple[NodeID, str, str]] = []
    for gate_id, gate_name, input_connection in config_gate_schema_inputs:
        if input_connection not in producers:
            suggestions = _suggest_similar(input_connection, sorted(producers.keys()))
            hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
            raise GraphValidationError(
                f"Gate '{gate_name}' input '{input_connection}' has no producer.{hint}\nAvailable connections: {', '.join(sorted(producers.keys()))}",
                component_id=gate_name,
                component_type="gate",
            )
        producer_id, _producer_label = producers[input_connection]
        upstream_info = graph.get_node_info(producer_id)
        if upstream_info.output_schema_config is not None:
            _assign_schema(gate_id, _best_schema_config(producer_id))
        else:
            deferred_config_gate_schemas.append((gate_id, gate_name, input_connection))

    # ===== MATCH PRODUCERS TO CONSUMERS =====
    gate_node_ids = {entry.node_id for entry in gate_entries}

    gate_default_continue_targets: dict[NodeID, NodeID] = {}
    ambiguous_continue_gates: set[NodeID] = set()

    for connection_name, consumer_id in consumers.items():
        producer_id, producer_label = producers[connection_name]
        if producer_id in gate_node_ids and producer_label != "continue":
            route_labels = gate_connection_route_labels[(producer_id, connection_name)]
            if route_labels:
                for route_label in route_labels:
                    graph.add_edge(producer_id, consumer_id, label=route_label, mode=RoutingMode.MOVE)
            else:
                graph.add_edge(producer_id, consumer_id, label=producer_label, mode=RoutingMode.MOVE)
            # Preserve gate fallthrough semantics for RoutingAction.continue_():
            # when a gate has a single downstream processing target, continue
            # should route there even if explicit route labels are present.
            if producer_id not in gate_default_continue_targets:
                gate_default_continue_targets[producer_id] = consumer_id
            elif gate_default_continue_targets[producer_id] != consumer_id:
                # Ambiguous continue fallthrough (multiple processing targets).
                # Leave unresolved; GateExecutor will fail closed if a gate
                # emits continue_() without a unique continuation edge.
                ambiguous_continue_gates.add(producer_id)
        else:
            graph.add_edge(producer_id, consumer_id, label="continue", mode=RoutingMode.MOVE)

    for gate_id, continue_target in gate_default_continue_targets.items():
        if gate_id in ambiguous_continue_gates:
            continue
        graph.add_edge(gate_id, continue_target, label="continue", mode=RoutingMode.MOVE)

    # ===== RESOLVE DEFERRED GATE ROUTES =====
    for gate_id, route_label, target in gate_route_connections:
        if target in consumers:
            graph.add_route_resolution_entry(gate_id, route_label, RouteDestination.processing_node(consumers[target]))
        elif target == "discard":
            graph.add_route_resolution_entry(gate_id, route_label, RouteDestination.discard())
        else:
            suggestions = _suggest_similar(target, sorted(consumers.keys()))
            hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
            raise GraphValidationError(
                f"Gate route target '{target}' is neither a sink nor a known connection name.{hint}",
                component_id=str(gate_id),
                component_type="gate",
            )

    # Ensure all declared gate route labels are resolvable before runtime.
    graph._validate_route_resolution_map_complete()

    # ===== TERMINAL ROUTING (on_success -> sinks) =====
    for wired in transforms:
        on_success = wired.settings.on_success
        tid = transform_ids_by_name[wired.settings.name]
        if SinkName(on_success) in sink_ids:
            graph.add_edge(tid, sink_ids[SinkName(on_success)], label="on_success", mode=RoutingMode.MOVE)
        elif on_success not in consumers:
            suggestions = _suggest_similar(on_success, sorted(consumers.keys()))
            hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
            raise GraphValidationError(
                f"Transform '{wired.settings.name}' on_success '{on_success}' is neither a sink nor a known connection.{hint}",
                component_id=wired.settings.name,
                component_type="transform",
            )

    for agg_name, (_transform, agg_settings) in aggregations.items():
        agg_on_success = agg_settings.on_success
        if agg_on_success is None:
            continue
        aid = aggregation_ids[AggregationName(agg_name)]
        if SinkName(agg_on_success) in sink_ids:
            graph.add_edge(aid, sink_ids[SinkName(agg_on_success)], label="on_success", mode=RoutingMode.MOVE)
        elif agg_on_success not in consumers:
            suggestions = _suggest_similar(agg_on_success, sorted(consumers.keys()))
            hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
            raise GraphValidationError(
                f"Aggregation '{agg_settings.name}' on_success '{agg_on_success}' is neither a sink nor a known connection.{hint}",
                component_id=agg_settings.name,
                component_type="aggregation",
            )

    if coalesce_settings:
        for coalesce_config in coalesce_settings:
            if coalesce_config.on_success is None:
                continue
            if coalesce_config.on_success in consumers:
                raise GraphValidationError(
                    f"Coalesce '{coalesce_config.name}' has on_success='{coalesce_config.on_success}'. "
                    "Coalesce on_success must point to a sink when configured.",
                    component_id=coalesce_config.name,
                    component_type="coalesce",
                )
            on_success_sink = SinkName(coalesce_config.on_success)
            if on_success_sink not in sink_ids:
                raise GraphValidationError(
                    f"Coalesce '{coalesce_config.name}' on_success references unknown sink "
                    f"'{coalesce_config.on_success}'. Available sinks: {sorted(sink_ids.keys())}",
                    component_id=coalesce_config.name,
                    component_type="coalesce",
                )
            graph.add_edge(
                coalesce_ids[CoalesceName(coalesce_config.name)],
                sink_ids[on_success_sink],
                label="on_success",
                mode=RoutingMode.MOVE,
            )

    for source_name, source_settings_entry in source_settings_map.items():
        source_on_success = source_settings_entry.on_success
        source_display_name = sources[source_name].name if len(sources) == 1 and source_name == "source" else source_name
        if SinkName(source_on_success) in sink_ids:
            graph.add_edge(
                source_ids[source_name],
                sink_ids[SinkName(source_on_success)],
                label="on_success",
                mode=RoutingMode.MOVE,
            )
        elif source_on_success in queue_ids:
            if source_on_success not in consumers:
                raise GraphValidationError(
                    f"Source '{source_display_name}' on_success '{source_on_success}' "
                    f"references queue '{source_on_success}' with no downstream consumer.",
                    component_id=source_name,
                    component_type="source",
                )
        elif source_on_success not in consumers and source_on_success not in queue_ids:
            suggestions = _suggest_similar(source_on_success, sorted(str(s) for s in sink_ids))
            hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
            raise GraphValidationError(
                f"Source '{source_display_name}' on_success '{source_on_success}' is neither a sink nor a known connection.{hint}",
                component_id=source_name,
                component_type="source",
            )

    # Re-run namespace validation with dangling-output checks enabled now
    # that terminal on_success sink/connection validation has completed.
    cls._validate_connection_namespaces(
        producers=producers,
        consumers=consumers,
        consumer_claims=consumer_claims,
        sink_names=_sink_name_set(),
        check_dangling=True,
    )

    # ===== ADD DIVERT EDGES (quarantine/error sinks) =====
    # Divert edges represent error/quarantine data flows that bypass the
    # normal DAG execution path. They make quarantine/error sinks reachable
    # in the graph (required for node_ids and audit trail).
    #
    # These are STRUCTURAL markers, not execution paths. Rows reach these
    # sinks via exception handling (processor.py) or source validation
    # failures (orchestrator.py), not by traversing the edge during
    # normal processing.

    # Source quarantine edges
    # _on_validation_failure is defined on SourceProtocol (protocols.py:78)
    for source_name, source_instance in sources.items():
        quarantine_dest = source_instance._on_validation_failure
        if quarantine_dest != "discard" and SinkName(quarantine_dest) in sink_ids:
            graph.add_edge(
                source_ids[source_name],
                sink_ids[SinkName(quarantine_dest)],
                label="__quarantine__",
                mode=RoutingMode.DIVERT,
            )

    # Transform error edges
    for wired in transforms:
        on_error = wired.settings.on_error
        if on_error != "discard":
            if SinkName(on_error) not in sink_ids:
                suggestions = _suggest_similar(on_error, sorted(str(s) for s in sink_ids))
                hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
                raise GraphValidationError(
                    f"Transform '{wired.settings.name}' on_error '{on_error}' references unknown sink.{hint} "
                    f"Available sinks: {', '.join(sorted(str(s) for s in sink_ids))}",
                    component_id=wired.settings.name,
                    component_type="transform",
                )
            graph.add_edge(
                transform_ids_by_name[wired.settings.name],
                sink_ids[SinkName(on_error)],
                label=error_edge_label(wired.settings.name),
                mode=RoutingMode.DIVERT,
            )

    # Sink failsink edges
    for sink_name_key, sink_node_id in sink_ids.items():
        sink_instance = sinks[str(sink_name_key)]
        on_write_failure = sink_instance._on_write_failure
        if on_write_failure is not None and on_write_failure != "discard":
            failsink_name = SinkName(on_write_failure)
            if failsink_name not in sink_ids:
                raise GraphValidationError(
                    f"Sink '{sink_name_key}' on_write_failure references '{on_write_failure}' "
                    f"which is not in sink_ids. Available: {sorted(str(s) for s in sink_ids)}.",
                    component_id=str(sink_name_key),
                    component_type="sink",
                )
            graph.add_edge(
                sink_node_id,
                sink_ids[failsink_name],
                label="__failsink__",
                mode=RoutingMode.DIVERT,
            )

    # ===== PIPELINE ORDERING (TOPOLOGICAL) =====
    processing_node_ids: set[NodeID] = set()
    processing_node_ids.update(queue_ids.values())
    processing_node_ids.update(transform_ids_by_name.values())
    processing_node_ids.update(aggregation_ids.values())
    processing_node_ids.update(config_gate_ids.values())
    processing_node_ids.update(coalesce_ids.values())

    pipeline_nodes = graph.topological_processing_order(processing_node_ids)

    branch_info: dict[BranchName, BranchInfo] = {branch_name: plan.to_branch_info() for branch_name, plan in coalesce_branch_plans.items()}
    graph.set_branch_info(branch_info)

    # ===== POPULATE COALESCE SCHEMA CONFIG =====
    # Coalesce nodes are structural pass-throughs; record the upstream schema
    # so audit logs reflect the actual data contract at the merge point.
    # Schema validation is strategy-aware:
    #   union:  require compatible types on overlapping fields
    #   nested: no cross-branch constraint (each branch keyed separately)
    #   select: no cross-branch constraint (only selected branch matters)
    coalesce_id_to_config: dict[NodeID, CoalesceSettings] = {}
    if coalesce_settings:
        for coalesce_config in coalesce_settings:
            cid = coalesce_ids[CoalesceName(coalesce_config.name)]
            coalesce_id_to_config[cid] = coalesce_config

    for coalesce_id in coalesce_ids.values():
        incoming_edges = graph.get_incoming_edges(coalesce_id)
        if not incoming_edges:
            raise GraphValidationError(
                f"Coalesce node '{coalesce_id}' has no incoming branches; cannot determine schema for audit.",
                component_id=str(coalesce_id),
                component_type="coalesce",
            )

        coal_config = coalesce_id_to_config[coalesce_id]

        # Build a branch_name → schema mapping from the branch plan created
        # during fork/coalesce wiring. Identity branches use the producing
        # gate schema; transform branches use their configured input
        # connection's producer.
        branch_to_schema: dict[str, SchemaConfig] = {}

        coalesce_plan = coalesce_plans[CoalesceName(coal_config.name)]
        for branch_spec in coalesce_plan.branches:
            if branch_spec.branch_name not in coalesce_branch_plans:
                continue
            branch_plan = coalesce_branch_plans[branch_spec.branch_name]
            if branch_plan.uses_transform_chain:
                producer_node, _producer_label = producers[branch_plan.input_connection]
            else:
                producer_node = branch_plan.gate_node_id
            branch_to_schema[str(branch_plan.branch_name)] = _best_schema_config(producer_node)

        # Update branch_info with schema information for runtime tracking of
        # lost branch fields. When a branch is diverted at runtime, the coalesce
        # executor can report which fields were expected from that lost branch.
        for branch_name_str, schema in branch_to_schema.items():
            branch_key = BranchName(branch_name_str)
            if branch_key in branch_info:
                # Use replace() to preserve any future BranchInfo fields automatically
                branch_info[branch_key] = replace(branch_info[branch_key], schema=schema)

        merged_schema = merge_coalesce_schema(
            branch_to_schema,
            merge_strategy=coal_config.merge,
            require_all=coal_config.has_all_branch_semantics,
            collision_policy=coal_config.union_collision_policy,
            branch_order=tuple(coal_config.branches.keys()),
            select_branch=coal_config.select_branch,
            coalesce_id=str(coalesce_id),
        )
        _assign_schema(coalesce_id, merged_schema)

    # Update branch_info on the graph now that schemas are populated.
    # The initial set_branch_info (line ~821) stored entries without schemas.
    # This call overwrites with schema-enriched entries for runtime lost-branch
    # field tracking.
    if branch_info:
        graph.set_branch_info(branch_info)

    # Config gate schema resolution (pass 2): resolve gates that were deferred
    # because their upstream producer (e.g., coalesce) didn't have schema yet.
    for gate_id, _gate_name, input_connection in deferred_config_gate_schemas:
        producer_id, _producer_label = producers[input_connection]
        _assign_schema(gate_id, _best_schema_config(producer_id))

    # PHASE 2 VALIDATION: Validate schema compatibility AFTER graph is built
    graph.validate_edge_compatibility()

    # Warn about DIVERT edges feeding require_all coalesces (non-fatal).
    if coalesce_id_to_config:
        graph.set_validation_warnings(graph.warn_divert_coalesce_interactions(coalesce_id_to_config))

    # Deep-freeze all NodeInfo configs now that schema resolution is complete.
    # NodeInfo.__post_init__ cannot freeze config because graph construction
    # replaces NodeInfo payloads during multi-step schema propagation.
    # deep_freeze converts nested dicts/lists to MappingProxyType/tuple recursively.
    graph.finalize_node_configs()

    # Step maps and node sequence support node_id-based processor traversal.
    graph.set_pipeline_nodes(pipeline_nodes)
    graph.set_node_step_map(graph.build_step_map())
    graph._freeze_build_metadata()

    return graph
