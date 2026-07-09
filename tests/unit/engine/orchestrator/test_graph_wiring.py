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
from elspeth.contracts.types import NodeID
from elspeth.core.dag import ExecutionGraph
from elspeth.engine.orchestrator import PipelineConfig
from elspeth.engine.orchestrator.graph_wiring import (
    assign_plugin_node_ids,
    build_dag_traversal_context,
    build_source_id_map,
    load_edge_map,
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
        aggregation_node_ids=frozenset(graph.get_aggregation_id_map().values()),
    )
    return config, graph


class TestAssignPluginNodeIds:
    def test_pre_set_regular_transform_node_id_must_match_graph_map(self) -> None:
        """A stale transform node_id is not an aggregation marker."""
        transform = as_transform(PassTransform())
        transform.node_id = "stale-transform"

        with pytest.raises(OrchestrationInvariantError, match="stale-transform"):
            assign_plugin_node_ids(
                sources={},
                transforms=[transform],
                sinks={},
                source_id_map={},
                transform_id_map={0: NodeID("transform-0")},
                sink_id_map={},
                aggregation_node_ids=frozenset(),
            )

    def test_pre_set_aggregation_transform_node_id_is_allowed_when_in_aggregation_set(self) -> None:
        """Aggregation transforms may keep the graph aggregation node id."""
        transform = as_transform(PassTransform())
        transform.node_id = "agg-transform-0"

        assign_plugin_node_ids(
            sources={},
            transforms=[transform],
            sinks={},
            source_id_map={},
            transform_id_map={},
            sink_id_map={},
            aggregation_node_ids=frozenset({NodeID("agg-transform-0")}),
        )

        assert transform.node_id == "agg-transform-0"


class TestSharedGraphArtifactLoaders:
    """build_source_id_map / load_edge_map — one loader for leader, resume, follower.

    Regression for elspeth-07b2031e41: the source-name map loop and the
    edge-map load+rekey were copy-pasted across core.py (leader), resume.py,
    and follower.py, so a multi-source or edge-map fix to one seam silently
    missed the others.
    """

    def test_build_source_id_map_maps_source_names(self) -> None:
        _config, graph = _build_wired_pipeline()

        source_id_map = build_source_id_map(graph)

        assert source_id_map == {"primary": graph.get_sources()[0]}

    def test_build_source_id_map_refuses_missing_source_name(self) -> None:
        """A source node without source_name is an ADR-025 §2 construction bug."""
        from elspeth.contracts.enums import NodeType

        graph = ExecutionGraph()
        graph.add_node("src", node_type=NodeType.SOURCE, plugin_name="null", config={})

        with pytest.raises(OrchestrationInvariantError, match="ADR-025"):
            build_source_id_map(graph)

    def test_load_edge_map_rekeys_for_row_processor(self) -> None:
        """DB edge keys (str, str) are rekeyed to (NodeID, label)."""

        class _EdgeMapDataFlow:
            def get_edge_map(self, run_id: str) -> dict[tuple[str, str], str]:
                return {("node-1", "continue"): "edge-1", ("node-2", "__error_x__"): "edge-2"}

        edge_map = load_edge_map(_EdgeMapDataFlow(), "run-1")

        assert edge_map == {
            (NodeID("node-1"), "continue"): "edge-1",
            (NodeID("node-2"), "__error_x__"): "edge-2",
        }


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
