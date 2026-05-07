"""NFR gate for the propagation-aware validator walk (ADR-007).

Measures the cost of ``_walk_effective_guaranteed_fields`` on a reference
diamond topology (16 nodes — source + 4 levels of branching factor 2 +
coalesce + sink). The gate enforces P99 ≤ 5 ms on this topology.
"""

from __future__ import annotations

import pytest

from elspeth.contracts import NodeType, RoutingMode
from elspeth.contracts.schema import FieldDefinition, SchemaConfig
from elspeth.core.dag import ExecutionGraph


def _build_reference_diamond() -> ExecutionGraph:
    """16-node diamond: source → gate → 4 levels → coalesce → sink."""
    leaf_schema = SchemaConfig(
        mode="flexible",
        fields=(
            FieldDefinition("a", "str", required=True),
            FieldDefinition("b", "str", required=True),
        ),
        guaranteed_fields=("a", "b"),
    )
    coalesce_schema = SchemaConfig(
        mode="flexible",
        fields=(
            FieldDefinition("a", "str", required=True),
            FieldDefinition("b", "str", required=True),
        ),
        guaranteed_fields=("a", "b"),
    )

    graph = ExecutionGraph()
    graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv", output_schema_config=leaf_schema)
    graph.add_node("gate", node_type=NodeType.GATE, plugin_name="fork")
    graph.add_edge("source", "gate", label="continue", mode=RoutingMode.MOVE)
    # 4 parallel pass-through branches.
    for i in range(4):
        graph.add_node(
            f"pt_{i}",
            node_type=NodeType.TRANSFORM,
            plugin_name="passthrough",
            config={"schema": {"mode": "observed"}},
            output_schema_config=None,
            passes_through_input=True,
        )
        graph.add_edge("gate", f"pt_{i}", label=f"branch_{i}", mode=RoutingMode.COPY)
    graph.add_node(
        "coalesce",
        node_type=NodeType.COALESCE,
        plugin_name="coalesce:merge",
        config={
            "branches": {f"pt_{i}": f"pt_{i}" for i in range(4)},
            "policy": "require_all",
            "merge": "union",
        },
        output_schema_config=coalesce_schema,
    )
    for i in range(4):
        graph.add_edge(f"pt_{i}", "coalesce", label=f"pt_{i}", mode=RoutingMode.MOVE)
    graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv_sink")
    graph.add_edge("coalesce", "sink", label="continue")
    return graph


@pytest.mark.performance
def test_diamond_walk_p99_within_budget(benchmark: pytest.FixtureRequest) -> None:
    """Walk on a 16-node reference diamond completes with P99 ≤ 5 ms."""
    graph = _build_reference_diamond()

    def walk() -> frozenset[str]:
        return graph.get_effective_guaranteed_fields("coalesce")

    result = benchmark(walk)
    assert result == frozenset({"a", "b"})

    stats = benchmark.stats
    median_sec = stats["median"]
    mean_sec = stats["mean"]
    stddev_sec = stats["stddev"]
    p99_bound = mean_sec + 3 * stddev_sec
    assert p99_bound < 5e-3, (
        f"Mean+3*stddev {p99_bound * 1e3:.3f}ms exceeds 5ms budget (mean={mean_sec * 1e3:.3f}ms, stddev={stddev_sec * 1e3:.3f}ms)"
    )
    assert median_sec < 5e-3
