"""Build-time schema/contract validation for the execution graph.

PHASE 2 (cross-plugin) validation: per-edge contract + type compatibility,
coalesce branch compatibility, sink required-field satisfaction, and the
effective-producer-schema resolution they share. Extracted from graph.py
(elspeth-b2c6ab6db8): validation policy is contract logic, not graph
topology. ExecutionGraph keeps thin delegators with the historical
signatures.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from elspeth.contracts import PluginSchema, RoutingMode, check_compatibility
from elspeth.contracts.enums import NodeType
from elspeth.contracts.types import NodeID
from elspeth.core.dag.guarantees import (
    EffectiveGuaranteeVote,
    get_effective_guaranteed_fields,
    get_required_fields,
    walk_effective_guarantee_vote,
)
from elspeth.core.dag.models import EdgeContractError, GraphValidationError
from elspeth.core.dag.schema_factory import build_coalesce_schema

if TYPE_CHECKING:
    from elspeth.core.dag.graph import ExecutionGraph


def validate_edge_compatibility(graph: ExecutionGraph) -> None:
    """Validate schema compatibility for all edges in the graph.

    Called AFTER graph construction is complete. Validates that each edge
    connects compatible schemas.

    Raises:
        ValueError: If any edge has incompatible schemas

    Note:
        This is PHASE 2 validation (cross-plugin compatibility). Plugin
        SELF-validation happens in PHASE 1 during plugin construction.
    """
    # Schema resolution cache shared across all edge validations.
    # Eliminates redundant recursion through long gate chains (O(N^2) → O(N)).
    schema_cache: dict[str, type[PluginSchema] | None] = {}

    # Validate each edge (skip divert edges — quarantine/error data doesn't
    # conform to producer schemas because it failed validation or errored)
    for from_id, to_id, edge_data in graph._graph.edges(data=True):
        if edge_data["mode"] == RoutingMode.DIVERT:
            continue
        validate_single_edge(graph, from_id, to_id, _schema_cache=schema_cache)

    # Validate all coalesce nodes (must have compatible schemas from all branches)
    coalesce_nodes = [node_id for node_id, data in graph._graph.nodes(data=True) if data["info"].node_type == NodeType.COALESCE]
    for coalesce_id in coalesce_nodes:
        validate_coalesce_compatibility(graph, coalesce_id, _schema_cache=schema_cache)

    # Validate sink required-field satisfaction against direct predecessors.
    # Catches mismatches between sink.declared_required_fields and upstream
    # output optionality at build time, rather than at runtime with a
    # generic PluginContractViolation. Walking sinks backward (rather than
    # coalesces forward) handles COALESCE → TRANSFORM → SINK topologies
    # and any future multi-hop shape.
    validate_sink_required_fields(graph)


def validate_single_edge(
    graph: ExecutionGraph,
    from_node_id: str,
    to_node_id: str,
    *,
    _schema_cache: dict[str, type[PluginSchema] | None] | None = None,
) -> None:
    """Validate schema compatibility for a single edge.

    Validation is performed in two phases:
    1. CONTRACT VALIDATION: Check required/guaranteed field names
    2. TYPE VALIDATION: Check field type compatibility

    Contract validation catches missing fields even for dynamic schemas,
    which is critical for template-based transforms (e.g., LLM) that
    reference specific row fields.

    Args:
        graph: The execution graph to validate against
        from_node_id: Source node ID
        to_node_id: Destination node ID
        _schema_cache: Shared memoization dict for schema resolution.

    Raises:
        GraphValidationError: If schemas are incompatible or contracts violated
    """
    to_info = graph.get_node_info(to_node_id)

    # Skip edge validation for coalesce nodes - they have special validation
    # that checks all incoming branches together
    if to_info.node_type == NodeType.COALESCE:
        return

    # Rule 0: Gates must preserve schema (input == output)
    if (
        to_info.node_type == NodeType.GATE
        and to_info.input_schema is not None
        and to_info.output_schema is not None
        and to_info.input_schema != to_info.output_schema
    ):
        raise GraphValidationError(
            f"Gate '{to_node_id}' must preserve schema: "
            f"input_schema={to_info.input_schema.__name__}, "
            f"output_schema={to_info.output_schema.__name__}",
            component_id=str(to_node_id),
            component_type="gate",
        )

    # ===== PHASE 1: CONTRACT VALIDATION (field name requirements) =====
    # This catches missing fields even for dynamic schemas.
    #
    # SINK consumers are deliberately excluded: their required fields are
    # validated by validate_sink_required_fields, whose EffectiveGuaranteeVote
    # walk distinguishes an ABSTAINING upstream (no fields, no participation —
    # defer to SinkExecutor's per-row declared_required_fields enforcement)
    # from a participating upstream that genuinely misses fields (reject at
    # build). Running the name check here too would pre-empt that abstention
    # with a hard failure and make the dedicated sink check unreachable for
    # its target scenario (elspeth-3283f2eaec).
    consumer_required = frozenset() if to_info.node_type == NodeType.SINK else get_required_fields(graph, to_node_id)

    if consumer_required:
        # Get effective guaranteed fields (walks through pass-through nodes)
        producer_guaranteed = get_effective_guaranteed_fields(graph, from_node_id)

        missing = consumer_required - producer_guaranteed
        if missing:
            # Build actionable error message
            from_info = graph.get_node_info(from_node_id)
            raise GraphValidationError(
                f"Schema contract violation: edge '{from_node_id}' → '{to_node_id}'\n"
                f"  Consumer ({to_info.plugin_name}) requires fields: {sorted(consumer_required)}\n"
                f"  Producer ({from_info.plugin_name}) guarantees: "
                f"{sorted(producer_guaranteed) if producer_guaranteed else '(none - dynamic schema)'}\n"
                f"  Missing fields: {sorted(missing)}\n"
                f"\n"
                f"Fix: Either:\n"
                f"  1. Add missing fields to producer's schema or guaranteed_fields, or\n"
                f"  2. Remove from consumer's required_input_fields if truly optional",
                component_id=str(to_node_id),
                component_type=to_info.node_type.value,
            )

    # ===== PHASE 2: TYPE VALIDATION (schema compatibility) =====
    # Get EFFECTIVE producer schema (walks through gates if needed)
    producer_schema = get_effective_producer_schema(graph, from_node_id, _cache=_schema_cache)
    consumer_schema = to_info.input_schema

    # Rule 1: Dynamic schemas (None) bypass type validation
    if producer_schema is None or consumer_schema is None:
        return  # Observed schema - compatible with anything

    # Handle observed schemas (no explicit fields + extra='allow')
    # These are created by _create_dynamic_schema and accept anything
    # NOTE: We control all schemas via PluginSchema base class which sets model_config["extra"].
    # Direct access is correct per Tier 1 trust model - missing key would be our bug.
    producer_is_observed = len(producer_schema.model_fields) == 0 and producer_schema.model_config["extra"] == "allow"
    consumer_is_observed = len(consumer_schema.model_fields) == 0 and consumer_schema.model_config["extra"] == "allow"
    if producer_is_observed or consumer_is_observed:
        return  # Observed schemas bypass static type validation

    # Rule 2: Full compatibility check (missing fields, type mismatches, extra fields)
    result = check_compatibility(producer_schema, consumer_schema)
    if not result.compatible:
        # Raise the structured subclass so downstream layers (composer
        # runtime preflight error formatter at
        # web/execution/validation.py) can build LLM-actionable
        # suggestions without re-parsing the prose form. The prose
        # message remains backwards-compatible for legacy str(exc)
        # consumers.
        raise EdgeContractError(
            f"Edge from '{from_node_id}' to '{to_node_id}' invalid: "
            f"producer schema '{producer_schema.__name__}' incompatible with "
            f"consumer schema '{consumer_schema.__name__}': {result.error_message}",
            from_node_id=str(from_node_id),
            to_node_id=str(to_node_id),
            producer_schema_name=producer_schema.__name__,
            consumer_schema_name=consumer_schema.__name__,
            compatibility_result=result,
            component_type=to_info.node_type.value,
        )


def get_effective_producer_schema(
    graph: ExecutionGraph,
    node_id: str,
    _cache: dict[str, type[PluginSchema] | None] | None = None,
) -> type[PluginSchema] | None:
    """Get effective output schema, walking through pass-through nodes (gates, coalesce).

    Gates and coalesce nodes don't transform data - they inherit schema from their
    upstream producers. This method walks backwards through the graph to find the
    nearest schema-carrying producer.

    Results are memoized in ``_cache`` to avoid redundant recursion through
    long gate chains (O(N) depth per gate → O(1) with cache).

    Args:
        graph: The execution graph to resolve against
        node_id: Node to get effective schema for
        _cache: Internal memoization dict, created on first call.

    Returns:
        Output schema type, or None if dynamic

    Raises:
        GraphValidationError: If pass-through node has no incoming edges (graph construction bug)
    """
    if _cache is None:
        _cache = {}

    if node_id in _cache:
        return _cache[node_id]

    node_info = graph.get_node_info(node_id)

    # If node has output_schema, return it directly
    if node_info.output_schema is not None:
        _cache[node_id] = node_info.output_schema
        return node_info.output_schema

    # Coalesce nodes are NOT pass-throughs — they transform data via merge
    # strategy (nested wraps in {branch: data}, union merges fields, select
    # picks a branch).  Strategy-aware handling:
    #   - select: passes through one branch unchanged → trace that branch's schema
    #   - union:  builder synthesizes typed field schema → build PluginSchema class
    #   - nested: all fields are "any" type → return None (nothing useful to validate)
    if node_info.node_type == NodeType.COALESCE:
        merge_strategy = node_info.config["merge"]
        if merge_strategy == "select":
            # Select merge passes through the selected branch's data unchanged.
            # Trace back to that branch's producer schema for type validation.
            if "select_branch" not in node_info.config:
                raise GraphValidationError(
                    f"Coalesce node '{node_id}' has merge strategy 'select' but "
                    "no 'select_branch' in config. This indicates a graph construction bug.",
                    component_id=str(node_id),
                    component_type="coalesce",
                )
            select_branch = node_info.config["select_branch"]
            # Identity branch: COPY edge from gate to coalesce with label == select_branch
            for from_id, _, _key, edge_data in graph._graph.in_edges(node_id, keys=True, data=True):
                if edge_data["mode"] == RoutingMode.COPY and edge_data["label"] == select_branch:
                    result = get_effective_producer_schema(graph, from_id, _cache)
                    _cache[node_id] = result
                    return result
            # Transform branch: last transform's edge has label "continue", not
            # the branch name. Trace backward to find the last transform node.
            _first, last = graph._trace_branch_endpoints(NodeID(node_id), select_branch)
            result = get_effective_producer_schema(graph, last, _cache)
            _cache[node_id] = result
            return result
        if merge_strategy == "union" and node_info.output_schema_config is not None and not node_info.output_schema_config.is_observed:
            # Union merge: the builder sets output_schema_config with concrete
            # field types.  Build a PluginSchema class from it so PHASE 2 edge
            # validation can type-check downstream.
            result = build_coalesce_schema(node_info.output_schema_config, coalesce_id=str(node_id))
            _cache[node_id] = result
            return result
        # nested merge, observed union, or no synthesized schema: return None (dynamic)
        _cache[node_id] = None
        return None

    # Gates are true pass-throughs — inherit schema from upstream producers
    if node_info.node_type == NodeType.GATE:
        incoming = list(graph._graph.in_edges(node_id, keys=True, data=True))

        if not incoming:
            # Pass-through node with no inputs is a graph construction bug - CRASH
            raise GraphValidationError(
                f"{node_info.node_type.capitalize()} node '{node_id}' has no incoming edges - this indicates a bug in graph construction",
                component_id=str(node_id),
                component_type=node_info.node_type.value,
            )

        # Gather all input schemas for validation
        all_schemas: list[tuple[str, type[PluginSchema] | None]] = []
        for from_id, _, _key, _ in incoming:
            schema = get_effective_producer_schema(graph, from_id, _cache)
            all_schemas.append((from_id, schema))

        # For multi-input nodes, check for mixed observed/explicit schemas first
        # Mixed observed/explicit branches create semantic mismatches that cause runtime failures
        if len(all_schemas) > 1:
            observed_branches = [(nid, s) for nid, s in all_schemas if is_observed_schema(s)]
            explicit_branches = [(nid, s) for nid, s in all_schemas if not is_observed_schema(s)]

            if observed_branches and explicit_branches:
                # Mixed observed/explicit - reject with clear error
                observed_names = [nid for nid, _ in observed_branches]
                # Schema is guaranteed non-None here (explicit_branches filtered out observed/None)
                explicit_names = [f"{nid} ({s.__name__})" for nid, s in explicit_branches if s is not None]
                raise GraphValidationError(
                    f"{node_info.node_type.capitalize()} '{node_id}' has mixed observed/explicit schemas - "
                    f"this is not allowed because observed branches may produce rows missing fields "
                    f"expected by downstream consumers. "
                    f"Observed branches: {observed_names}, explicit branches: {explicit_names}. "
                    f"Fix: ensure all branches produce explicit schemas with compatible fields, "
                    f"or all branches produce observed schemas.",
                    component_id=str(node_id),
                    component_type=node_info.node_type.value,
                )

            # All explicit - verify structural compatibility
            if len(explicit_branches) > 1:
                _first_id, first_schema = explicit_branches[0]
                for _other_id, other_schema in explicit_branches[1:]:
                    compatible, error_msg = schemas_structurally_compatible(first_schema, other_schema)
                    if not compatible:
                        # Schemas are guaranteed non-None here (explicit_branches filtered out observed/None)
                        first_name = first_schema.__name__ if first_schema is not None else "observed"
                        other_name = other_schema.__name__ if other_schema is not None else "observed"
                        raise GraphValidationError(
                            f"{node_info.node_type.capitalize()} '{node_id}' receives incompatible schemas from "
                            f"multiple inputs - this is a graph construction bug. "
                            f"First input: {first_name}, other input: {other_name}. {error_msg}",
                            component_id=str(node_id),
                            component_type=node_info.node_type.value,
                        )

        # Return first schema (all are now either all-observed or all-explicit-compatible)
        result = all_schemas[0][1]
        _cache[node_id] = result
        return result

    # Not a pass-through node and no schema - return None (observed)
    _cache[node_id] = None
    return None


def is_observed_schema(schema: type[PluginSchema] | None) -> bool:
    """Check if a schema is observed (accepts any fields, types inferred from data).

    A schema is observed if:
    - It is None (unspecified output_schema)
    - It has no fields and allows extra fields (structural observed)

    Args:
        schema: Schema class or None

    Returns:
        True if schema is observed, False if explicit (fixed/flexible)
    """
    if schema is None:
        return True

    # Structural observed: no fields + extra="allow"
    # NOTE: We control all schemas via PluginSchema base class which sets model_config["extra"].
    # Direct access is correct per Tier 1 trust model - missing key would be our bug.
    return len(schema.model_fields) == 0 and schema.model_config["extra"] == "allow"


def schemas_structurally_compatible(schema_a: type[PluginSchema] | None, schema_b: type[PluginSchema] | None) -> tuple[bool, str]:
    """Check if two schemas are structurally compatible (not by class identity).

    Uses check_compatibility() for structural comparison. Handles observed schemas
    which are compatible with anything.

    Args:
        schema_a: First schema (or None for observed)
        schema_b: Second schema (or None for observed)

    Returns:
        Tuple of (is_compatible, error_message). If compatible, error_message is empty.
    """
    # Both observed - compatible
    if is_observed_schema(schema_a) and is_observed_schema(schema_b):
        return True, ""

    # One observed, one explicit - for general compatibility, allow this
    # (Pass-through nodes use stricter checking via _check_passthrough_schema_homogeneity)
    if is_observed_schema(schema_a) or is_observed_schema(schema_b):
        return True, ""

    # Both explicit schemas - same class is trivially compatible
    if schema_a is schema_b:
        return True, ""

    # At this point both schemas are explicit (not None, not dynamic)
    # Type narrowing for mypy: we've already returned if either is dynamic/None
    assert schema_a is not None and schema_b is not None

    # Both explicit schemas - use bidirectional structural comparison
    # For coalesce/pass-through nodes, schemas must be mutually compatible
    result_ab = check_compatibility(schema_a, schema_b)
    result_ba = check_compatibility(schema_b, schema_a)

    if result_ab.compatible and result_ba.compatible:
        return True, ""

    # Build error message showing what's incompatible
    errors = []
    if not result_ab.compatible:
        errors.append(f"{schema_a.__name__} -> {schema_b.__name__}: {result_ab.error_message}")
    if not result_ba.compatible:
        errors.append(f"{schema_b.__name__} -> {schema_a.__name__}: {result_ba.error_message}")
    return False, "; ".join(errors)


def validate_coalesce_compatibility(
    graph: ExecutionGraph,
    coalesce_id: str,
    *,
    _schema_cache: dict[str, type[PluginSchema] | None] | None = None,
) -> None:
    """Validate all inputs to coalesce node have compatible schemas.

    Strategy-aware: only ``union`` requires cross-branch schema compatibility.
    ``nested`` and ``select`` strategies have no cross-branch constraint because
    branches are keyed separately (nested) or only one branch is used (select).

    Args:
        graph: The execution graph to validate against
        coalesce_id: Coalesce node ID
        _schema_cache: Shared memoization dict for schema resolution.

    Raises:
        GraphValidationError: If branches have incompatible schemas
    """
    incoming = list(graph._graph.in_edges(coalesce_id, keys=True, data=True))

    if not incoming:
        raise GraphValidationError(
            f"Coalesce '{coalesce_id}' has no incoming edges — this is a graph construction bug",
            component_id=str(coalesce_id),
            component_type="coalesce",
        )
    if len(incoming) < 2:
        return  # Degenerate case (1 branch) - always compatible

    # Determine merge strategy from node config.
    # Config is populated by the builder — direct access is correct (Tier 1).
    node_info = graph.get_node_info(coalesce_id)
    merge_strategy = node_info.config["merge"]

    # nested/select strategies have no cross-branch schema constraint
    if merge_strategy in ("nested", "select"):
        return

    # union strategy: gather all branch schemas and validate
    all_schemas: list[tuple[str, type[PluginSchema] | None]] = []
    for from_id, _, _key, _ in incoming:
        schema = get_effective_producer_schema(graph, from_id, _cache=_schema_cache)
        all_schemas.append((from_id, schema))

    # Reject mixed observed/explicit schemas
    observed_branches = [(nid, s) for nid, s in all_schemas if is_observed_schema(s)]
    explicit_branches = [(nid, s) for nid, s in all_schemas if not is_observed_schema(s)]

    if observed_branches and explicit_branches:
        observed_names = [nid for nid, _ in observed_branches]
        explicit_names = [f"{nid} ({s.__name__})" for nid, s in explicit_branches if s is not None]
        raise GraphValidationError(
            f"Coalesce '{coalesce_id}' has mixed observed/explicit schemas - "
            f"this is not allowed because observed branches may produce rows missing fields "
            f"expected by downstream consumers. "
            f"Observed branches: {observed_names}, explicit branches: {explicit_names}. "
            f"Fix: ensure all branches produce explicit schemas with compatible fields, "
            f"or all branches produce observed schemas.",
            component_id=str(coalesce_id),
            component_type="coalesce",
        )

    # All explicit: verify structural compatibility across branches
    if len(explicit_branches) > 1:
        _first_id, first_schema = explicit_branches[0]
        for other_id, other_schema in explicit_branches[1:]:
            compatible, error_msg = schemas_structurally_compatible(first_schema, other_schema)
            if not compatible:
                first_name = first_schema.__name__ if first_schema else "observed"
                other_name = other_schema.__name__ if other_schema else "observed"
                raise GraphValidationError(
                    f"Coalesce '{coalesce_id}' receives incompatible schemas from "
                    f"multiple branches: first branch has {first_name}, "
                    f"branch from '{other_id}' has {other_name}. {error_msg}",
                    component_id=str(coalesce_id),
                    component_type="coalesce",
                )


def validate_sink_required_fields(graph: ExecutionGraph) -> None:
    """Validate each sink's declared_required_fields against upstream guarantees.

    For every SINK node with a non-empty declared_required_fields, check
    every direct predecessor's effective guaranteed fields (via the existing
    get_effective_guaranteed_fields() API). A sink's required fields must
    be a subset of its upstream guaranteed fields.

    This catches cases where:
    - A coalesce marks a field optional (branch-exclusive or AND-downgraded)
      and feeds a sink that requires it (direct or through a transform)
    - A transform's declared_output_fields don't include a field the sink requires
    - A source doesn't guarantee a field the sink requires

    Uses get_effective_guaranteed_fields() rather than reading
    output_schema_config.fields directly. The fields tuple is unreliable
    for shape-changing transforms that use the base _build_output_schema_config()
    — that base method copies INPUT fields into the output config, with
    only guaranteed_fields recomputed correctly. The effective-guarantees
    API is the single source of truth: it handles aggregations (dynamic
    output → no guarantees), coalesce strategies (union intersection,
    nested branch keys, select pass-through), and the base-class
    guaranteed_fields recomputation in one place.

    Sink predecessors are visited via .predecessors() which yields each
    unique source node once, even when the underlying MultiDiGraph has
    parallel edges (e.g., a gate routing two labels to the same sink).

    Runs at build time rather than failing at runtime with a generic
    PluginContractViolation.

    Raises:
        GraphValidationError: if a sink's direct predecessor does not
            guarantee a field the sink requires.
    """
    # Shared cache across all sinks' predecessor walks. Mirrors the
    # schema_cache pattern at validate_edge_compatibility — avoids
    # re-walking common upstream nodes in fan-in topologies.
    effective_fields_cache: dict[str, EffectiveGuaranteeVote] = {}

    for node_id, data in graph._graph.nodes(data=True):
        info = data["info"]
        if info.node_type != NodeType.SINK:
            continue

        sink_required = info.declared_required_fields
        if not sink_required:
            continue

        for predecessor_id in graph._graph.predecessors(node_id):
            vote = walk_effective_guarantee_vote(graph, predecessor_id, effective_fields_cache)
            guaranteed = vote.fields
            if not guaranteed and not vote.participated:
                continue

            missing = sink_required - guaranteed
            if not missing:
                continue

            raise GraphValidationError(
                f"Sink '{info.plugin_name}' requires fields {sorted(missing)} "
                f"but its upstream '{predecessor_id}' does not guarantee them. "
                f"Likely causes: a coalesce union marked these fields optional "
                f"(branch-exclusive or AND-downgraded), or an upstream transform "
                f"did not declare them as guaranteed output. "
                f"Fix: ensure the upstream node guarantees these fields, "
                f"or remove them from the sink's declared_required_fields.",
                component_id=str(node_id),
                component_type="sink",
            )
