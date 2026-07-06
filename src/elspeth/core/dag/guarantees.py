"""Guarantee propagation and contract-field reads over the execution graph.

Implements the ADR-007 propagation-aware effective-guarantee walk (with the
ADR-009 §Clause 1 ``compose_propagation`` aggregation step) plus the raw
per-node contract reads it builds on. Extracted from graph.py
(elspeth-b2c6ab6db8): guarantee policy is schema/contract logic, not graph
topology. ExecutionGraph keeps thin delegators with the historical
signatures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from elspeth.contracts.errors import FrameworkBugError
from elspeth.contracts.guarantee_propagation import compose_propagation
from elspeth.contracts.schema import get_raw_node_required_fields
from elspeth.core.dag.models import GraphValidationError

if TYPE_CHECKING:
    from elspeth.contracts.schema import SchemaConfig
    from elspeth.core.dag.graph import ExecutionGraph


@dataclass(frozen=True, slots=True)
class EffectiveGuaranteeVote:
    """Propagation result that preserves fields AND participation state.

    ``fields`` alone is not enough for sink validation, because an empty
    effective guarantee set can mean either:

    - no node participated in the guarantee vote (abstain), or
    - one or more nodes participated and the result collapsed to empty

    Carrying ``participated`` through the recursive walk keeps that contract
    mechanical for downstream validators.
    """

    fields: frozenset[str]
    participated: bool


def get_schema_config_from_node(graph: ExecutionGraph, node_id: str) -> SchemaConfig | None:
    """Extract SchemaConfig from node.

    Returns output_schema_config directly — all nodes with schemas have
    this populated at construction time by the builder.

    Args:
        graph: The execution graph to read from
        node_id: Node ID to get schema config from

    Returns:
        SchemaConfig if available, None if node has no schema
    """
    node_info = graph.get_node_info(node_id)
    return node_info.output_schema_config


def get_guaranteed_fields(graph: ExecutionGraph, node_id: str) -> frozenset[str]:
    """Get fields that a node guarantees in its output.

    Priority:
    1. Explicit guaranteed_fields in schema config
    2. Declared fields in flexible/fixed mode schemas
    3. Empty set for observed schemas

    Args:
        graph: The execution graph to read from
        node_id: Node ID to get guarantees from

    Returns:
        Frozenset of field names the node guarantees to output
    """
    schema_config = get_schema_config_from_node(graph, node_id)

    if schema_config is None:
        return frozenset()

    return schema_config.get_effective_guaranteed_fields()


def get_required_fields(graph: ExecutionGraph, node_id: str) -> frozenset[str]:
    """Get fields that a node EXPLICITLY requires in its input.

    This returns only explicit contract declarations, not implicit
    requirements from typed schemas. The existing type validation
    handles typed schema compatibility separately.

    Priority:
    1. Explicit required_input_fields from plugin config (TransformDataConfig)
    2. Explicit required_fields in schema config

    Note: This deliberately does NOT include implicit requirements from
    strict/free mode schemas. Those are handled by type validation, which
    correctly skips validation when either side is dynamic.

    Args:
        graph: The execution graph to read from
        node_id: Node ID to get requirements from

    Returns:
        Frozenset of field names explicitly required
    """
    node_info = graph.get_node_info(node_id)
    try:
        return get_raw_node_required_fields(
            node_info.config,
            owner=f"node:{node_id}",
            node_type=node_info.node_type.value,
        )
    except ValueError as exc:
        raise GraphValidationError(
            f"Invalid contract config: {exc}",
            component_id=str(node_id),
            component_type=node_info.node_type.value,
        ) from exc


def get_effective_guaranteed_fields(graph: ExecutionGraph, node_id: str) -> frozenset[str]:
    """Get effective output guarantees for a node (propagation-aware).

    Per ADR-007, this method is the propagation-aware implementation.
    For a TRANSFORM node whose plugin declared ``passes_through_input=True``,
    the effective guarantees are the intersection of its participating
    predecessors' guarantees unioned with the node's own declared fields.
    For all other nodes, returns the node's own declarations (same as
    ``get_guaranteed_fields``).

    Callers that want the raw per-node declarations (without propagation)
    must call ``get_guaranteed_fields`` instead.

    For coalesce nodes, builder.py pre-computes strategy-aware guarantees:
    - **union** with require_all: union of branch guarantees (all branches arrive)
    - **union** with other policies: intersection (only fields in ALL branches)
    - **nested**: the node's own guarantees (branch names, not inner fields)
    - **select**: the node's own guarantees (selected branch's schema)

    Tests that construct graphs directly must set output_schema_config
    on nodes to match what the builder would compute.

    Args:
        graph: The execution graph to read from
        node_id: Node to get effective guarantees for

    Returns:
        Frozenset of field names effectively guaranteed at this point
    """
    return walk_effective_guaranteed_fields(graph, node_id, {})


def walk_effective_guarantee_vote(
    graph: ExecutionGraph,
    node_id: str,
    cache: dict[str, EffectiveGuaranteeVote],
    field_cache: dict[str, frozenset[str]] | None = None,
) -> EffectiveGuaranteeVote:
    """Recursive implementation that preserves participation state.

    Pass-through propagation needs more than the final field set. A sink
    validator must be able to distinguish "nobody voted" from "the vote
    participated and collapsed to empty", so this helper carries both.
    """
    if node_id in cache:
        return cache[node_id]

    node_info = graph.get_node_info(node_id)
    own_participates = node_info.output_schema_config is not None and node_info.output_schema_config.participates_in_propagation
    own_fields = (
        node_info.output_schema_config.get_effective_guaranteed_fields() if node_info.output_schema_config is not None else frozenset()
    )

    if node_info.passes_through_input:
        predecessors = list(graph._graph.predecessors(node_id))
        if not predecessors:
            raise FrameworkBugError(
                f"Pass-through transform {node_id!r} has no predecessors. Builder must wire transforms with at least one upstream edge."
            )
        predecessor_votes = [walk_effective_guarantee_vote(graph, pred_id, cache, field_cache) for pred_id in predecessors]
        result_fields = compose_propagation(
            own_fields,
            [vote.fields if vote.participated else None for vote in predecessor_votes],
        )
        result = EffectiveGuaranteeVote(
            fields=result_fields,
            participated=own_participates or any(vote.participated for vote in predecessor_votes),
        )
    else:
        result = EffectiveGuaranteeVote(
            fields=own_fields,
            participated=own_participates,
        )

    cache[node_id] = result
    if field_cache is not None:
        field_cache[node_id] = result.fields
    return result


def walk_effective_guaranteed_fields(
    graph: ExecutionGraph,
    node_id: str,
    cache: dict[str, frozenset[str]],
) -> frozenset[str]:
    """Recursive implementation of get_effective_guaranteed_fields.

    Explicit (non-optional) cache parameter — internal callers with
    bulk-validation scope (e.g., ``validate_sink_required_fields``)
    pass a shared cache across their loop to avoid per-node re-allocation
    and re-walking. Public callers go through ``get_effective_guaranteed_fields``
    which allocates a fresh per-call cache.

    For pass-through transforms, delegates the aggregation step to
    ``compose_propagation`` (ADR-009 §Clause 1). Predecessor
    participation is checked via ``SchemaConfig.participates_in_propagation``
    — the canonical predicate that both this walker and the composer's
    preview walker consult.

    Raises ``FrameworkBugError`` if a pass-through transform has no
    predecessors — per the NodeInfo guard, pass-through nodes are
    transforms, and transforms in a built DAG always have at least one
    upstream edge.
    """
    if node_id in cache:
        return cache[node_id]

    result = walk_effective_guarantee_vote(graph, node_id, {}, cache)
    cache[node_id] = result.fields
    return result.fields
