# tests/helpers/coalesce.py
"""Coalesce schema computation helpers for DAG tests.

These helpers use production merge functions from elspeth.core.dag.coalesce_merge
to ensure tests exercise the same logic as builder.py.
"""

from __future__ import annotations

from typing import Any

from elspeth.contracts import NodeType
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.dag import ExecutionGraph, merge_guaranteed_fields


def _compute_coalesce_schema(
    branch_schemas: dict[str, SchemaConfig | None],
    *,
    policy: str = "best_effort",
) -> SchemaConfig:
    """Compute merged schema for a coalesce node from branch schemas.

    Uses the production merge_guaranteed_fields() function from builder.py
    to ensure tests exercise the same logic as runtime.

    Args:
        branch_schemas: Map of branch_name -> SchemaConfig for each branch
        policy: Coalesce policy ("require_all" uses union, others use intersection)

    Returns:
        SchemaConfig with computed guaranteed_fields for the coalesce node
    """
    # Filter out None values before calling production function
    valid_schemas = {k: v for k, v in branch_schemas.items() if v is not None}

    merged_guaranteed_tuple = merge_guaranteed_fields(
        valid_schemas,
        require_all=(policy == "require_all"),
    )

    return SchemaConfig(mode="observed", fields=None, guaranteed_fields=merged_guaranteed_tuple)


def _add_coalesce_with_computed_schema(
    graph: ExecutionGraph,
    coalesce_node_id: str,
    branch_node_ids: list[str],
    *,
    policy: str = "best_effort",
    extra_config: dict[str, Any] | None = None,
) -> None:
    """Add a coalesce node with computed schema from its predecessor branches.

    This is a convenience helper for tests that construct graphs directly.
    It collects branch schemas, computes the merged schema, and adds the
    coalesce node with the correct output_schema_config.

    Args:
        graph: The ExecutionGraph to add the node to
        coalesce_node_id: ID for the coalesce node
        branch_node_ids: List of branch node IDs to compute guarantees from
        policy: Coalesce policy ("require_all" uses union, others use intersection)
        extra_config: Additional config dict entries for the coalesce node
    """
    # Collect branch schemas
    branch_schemas = {node_id: graph.get_schema_config_from_node(node_id) for node_id in branch_node_ids}

    # Compute merged schema
    coalesce_schema = _compute_coalesce_schema(branch_schemas, policy=policy)

    # Build config
    config: dict[str, Any] = {"merge": "union", "branches": {}, "policy": policy}
    if extra_config:
        config.update(extra_config)

    # Add coalesce node with computed schema
    graph.add_node(
        coalesce_node_id,
        node_type=NodeType.COALESCE,
        plugin_name="coalesce",
        config=config,
        output_schema_config=coalesce_schema,
    )
