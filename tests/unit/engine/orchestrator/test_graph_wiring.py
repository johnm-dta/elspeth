# tests/unit/engine/orchestrator/test_graph_wiring.py
"""Unit tests for graph_wiring.build_dag_traversal_context structural classification.

Regression tests for elspeth-c522931bd1: structural node IDs must be an
explicit allowlist (source/queue/coalesce node types), not the complement
"every node_to_next entry without a plugin". Under the complement derivation
a transform/gate node missing from node_to_plugin (node-id drift, config/graph
inconsistency) was silently classified structural and skipped at traversal —
fail-open. Construction must fail closed instead.
"""

from __future__ import annotations

import pytest

from elspeth.contracts.errors import OrchestrationInvariantError
from elspeth.core.dag import ExecutionGraph
from elspeth.engine.orchestrator import PipelineConfig
from elspeth.engine.orchestrator.graph_wiring import (
    assign_plugin_node_ids,
    build_dag_traversal_context,
)
from tests.fixtures.base_classes import as_sink, as_source, as_transform
from tests.fixtures.pipeline import build_production_graph
from tests.fixtures.plugins import CollectSink, ListSource, PassTransform


def _build_wired_pipeline() -> tuple[PipelineConfig, ExecutionGraph]:
    """Source -> transform -> sink pipeline with node_ids assigned."""
    config = PipelineConfig(
        sources={"primary": as_source(ListSource([{"value": 1}]))},
        transforms=[as_transform(PassTransform())],
        sinks={"default": as_sink(CollectSink())},
    )
    graph = build_production_graph(config)
    assign_plugin_node_ids(
        sources=config.sources,
        transforms=config.transforms,
        sinks=config.sinks,
        source_id_map={"primary": graph.get_sources()[0]},
        transform_id_map=graph.get_transform_id_map(),
        sink_id_map=graph.get_sink_id_map(),
    )
    return config, graph


class TestStructuralNodeClassification:
    def test_structural_node_ids_are_the_explicit_allowlist(self) -> None:
        """The traversal context carries source nodes as its structural set."""
        config, graph = _build_wired_pipeline()

        traversal = build_dag_traversal_context(
            graph=graph,
            config=config,
            config_gate_id_map=graph.get_config_gate_id_map(),
        )

        assert traversal.structural_node_ids == frozenset(graph.get_sources())

    def test_unaccounted_traversal_node_fails_closed(self) -> None:
        """A transform in node_to_next with no plugin mapping raises at construction.

        Simulates node-id drift: the plugin claims a node id the graph does not
        know, so the graph's real transform node has no plugin mapping. The old
        complement derivation classified it structural and the processor
        silently skipped it — bypassing whatever the transform enforced.
        """
        config, graph = _build_wired_pipeline()
        config.transforms[0].node_id = "transform-drifted"

        with pytest.raises(OrchestrationInvariantError, match="structural"):
            build_dag_traversal_context(
                graph=graph,
                config=config,
                config_gate_id_map=graph.get_config_gate_id_map(),
            )
