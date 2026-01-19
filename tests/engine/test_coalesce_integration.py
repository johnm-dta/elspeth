"""Integration tests for fork/coalesce pipelines.

These tests verify the complete flow:
source -> fork gate -> parallel paths -> coalesce -> sink

Unlike the unit tests in test_coalesce_executor.py, these tests use:
- Real source/sink plugins (inline test fixtures)
- Real Orchestrator
- Real ExecutionGraph.from_config()
- Real LandscapeDB (in-memory)
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from elspeth.contracts import Determinism, PluginSchema, RoutingMode, SourceRow
from elspeth.core.config import (
    CoalesceSettings,
    DatasourceSettings,
    ElspethSettings,
    GateSettings,
    SinkSettings,
)
from elspeth.core.dag import ExecutionGraph
from elspeth.core.landscape import LandscapeDB
from elspeth.engine.artifacts import ArtifactDescriptor
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from elspeth.plugins.base import BaseTransform

# =============================================================================
# Test Fixtures - Inline Plugin Implementations
# =============================================================================


class _TestSchema(PluginSchema):
    """Minimal schema for test fixtures."""

    pass


class _TestSourceBase:
    """Base class providing SourceProtocol required attributes."""

    node_id: str | None = None
    determinism = Determinism.DETERMINISTIC
    plugin_version = "1.0.0"


class _TestSinkBase:
    """Base class providing SinkProtocol required attributes."""

    input_schema = _TestSchema
    idempotent = True
    node_id: str | None = None
    determinism = Determinism.DETERMINISTIC
    plugin_version = "1.0.0"

    def flush(self) -> None:
        pass


class ListSource(_TestSourceBase):
    """Reusable test source that yields rows from a list."""

    name = "list_source"
    output_schema = _TestSchema

    def __init__(self, data: list[dict[str, Any]]) -> None:
        self._data = data

    def on_start(self, ctx: Any) -> None:
        pass

    def on_complete(self, ctx: Any) -> None:
        pass

    def load(self, ctx: Any) -> Iterator[SourceRow]:
        for row in self._data:
            yield SourceRow.valid(row)

    def close(self) -> None:
        pass


class CollectSink(_TestSinkBase):
    """Reusable test sink that collects rows into a list."""

    name = "collect_sink"

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    @property
    def config(self) -> dict[str, Any]:
        return {}

    def on_start(self, ctx: Any) -> None:
        pass

    def on_complete(self, ctx: Any) -> None:
        pass

    def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
        for row in rows:
            self.rows.append(row)
        return ArtifactDescriptor.for_file(
            path="memory://test",
            size_bytes=0,
            content_hash="test",
        )

    def close(self) -> None:
        pass


def _build_fork_coalesce_graph(
    config: PipelineConfig,
    settings: ElspethSettings,
) -> ExecutionGraph:
    """Build a test graph that supports fork and coalesce operations.

    This manually builds the graph because ExecutionGraph.from_config() requires
    plugins to be registered, which we can't do with inline test fixtures.

    Args:
        config: Pipeline configuration with plugins
        settings: Full settings with gates and coalesce config
    """
    graph = ExecutionGraph()

    # Add source
    source_id = "source_test"
    graph.add_node(
        source_id,
        node_type="source",
        plugin_name=config.source.name,
    )

    # Add transforms
    transform_ids: dict[int, str] = {}
    prev = source_id
    for i, t in enumerate(config.transforms):
        node_id = f"transform_{i}"
        transform_ids[i] = node_id
        graph.add_node(
            node_id,
            node_type="transform",
            plugin_name=t.name,
        )
        graph.add_edge(prev, node_id, label="continue", mode=RoutingMode.MOVE)
        prev = node_id

    # Add sinks
    sink_ids: dict[str, str] = {}
    for sink_name, sink in config.sinks.items():
        node_id = f"sink_{sink_name}"
        sink_ids[sink_name] = node_id
        graph.add_node(node_id, node_type="sink", plugin_name=sink.name)

    # Add config gates (from settings.gates)
    config_gate_ids: dict[str, str] = {}
    route_resolution_map: dict[tuple[str, str], str] = {}

    for gate_config in settings.gates:
        gate_id = f"config_gate_{gate_config.name}"
        config_gate_ids[gate_config.name] = gate_id

        gate_node_config = {
            "condition": gate_config.condition,
            "routes": dict(gate_config.routes),
        }
        if gate_config.fork_to:
            gate_node_config["fork_to"] = list(gate_config.fork_to)

        graph.add_node(
            gate_id,
            node_type="gate",
            plugin_name=f"config_gate:{gate_config.name}",
            config=gate_node_config,
        )

        # Edge from previous node
        graph.add_edge(prev, gate_id, label="continue", mode=RoutingMode.MOVE)

        # Config gate routes
        for route_label, target in gate_config.routes.items():
            route_resolution_map[(gate_id, route_label)] = target

        prev = gate_id

    # Build coalesce nodes
    coalesce_ids: dict[str, str] = {}
    branch_to_coalesce: dict[str, str] = {}

    for coalesce_config in settings.coalesce:
        cid = f"coalesce_{coalesce_config.name}"
        coalesce_ids[coalesce_config.name] = cid

        for branch in coalesce_config.branches:
            branch_to_coalesce[branch] = coalesce_config.name

        coalesce_node_config = {
            "branches": list(coalesce_config.branches),
            "policy": coalesce_config.policy,
            "merge": coalesce_config.merge,
            "timeout_seconds": coalesce_config.timeout_seconds,
            "quorum_count": coalesce_config.quorum_count,
            "select_branch": coalesce_config.select_branch,
        }

        graph.add_node(
            cid,
            node_type="coalesce",
            plugin_name=f"coalesce:{coalesce_config.name}",
            config=coalesce_node_config,
        )

    # Create edges from fork gates to coalesce nodes (for branches in coalesce)
    output_sink_id = sink_ids[settings.output_sink]

    for gate_config in settings.gates:
        if gate_config.fork_to:
            gate_id = config_gate_ids[gate_config.name]
            for branch in gate_config.fork_to:
                if branch in branch_to_coalesce:
                    coalesce_name = branch_to_coalesce[branch]
                    coalesce_id = coalesce_ids[coalesce_name]
                    graph.add_edge(
                        gate_id,
                        coalesce_id,
                        label=branch,
                        mode=RoutingMode.COPY,
                    )
                else:
                    # Branch not in any coalesce - route to output sink
                    graph.add_edge(
                        gate_id,
                        output_sink_id,
                        label=branch,
                        mode=RoutingMode.COPY,
                    )

    # Create edges from coalesce nodes to output sink
    for _coalesce_name, cid in coalesce_ids.items():
        graph.add_edge(
            cid,
            output_sink_id,
            label="continue",
            mode=RoutingMode.MOVE,
        )

    # Edge from last node to output sink (for non-fork paths)
    # Only add if no fork gates (fork gates handle their own routing)
    if not settings.gates or not any(g.fork_to for g in settings.gates):
        graph.add_edge(prev, output_sink_id, label="continue", mode=RoutingMode.MOVE)

    # Populate internal ID maps
    graph._sink_id_map = sink_ids
    graph._transform_id_map = transform_ids
    graph._config_gate_id_map = config_gate_ids
    graph._coalesce_id_map = coalesce_ids
    graph._branch_to_coalesce = branch_to_coalesce
    graph._route_resolution_map = route_resolution_map
    graph._output_sink = settings.output_sink

    return graph


# =============================================================================
# Test Classes
# =============================================================================


class TestForkCoalescePipeline:
    """Test complete fork -> process -> coalesce -> sink flow."""

    @pytest.fixture
    def db(self) -> LandscapeDB:
        return LandscapeDB.in_memory()

    def test_fork_coalesce_pipeline_produces_merged_output(
        self,
        db: LandscapeDB,
    ) -> None:
        """Complete fork/join pipeline should produce merged output."""
        settings = ElspethSettings(
            datasource=DatasourceSettings(
                plugin="list_source",
                options={},
            ),
            sinks={
                "output": SinkSettings(plugin="collect_sink", options={}),
            },
            output_sink="output",
            gates=[
                GateSettings(
                    name="forker",
                    condition="True",
                    routes={"true": "fork", "false": "continue"},
                    fork_to=["path_a", "path_b"],
                ),
            ],
            coalesce=[
                CoalesceSettings(
                    name="merge_results",
                    branches=["path_a", "path_b"],
                    policy="require_all",
                    merge="union",
                ),
            ],
        )

        source = ListSource([{"id": 1, "value": 100}])
        sink = CollectSink()

        config = PipelineConfig(
            source=source,
            transforms=[],
            sinks={"output": sink},
            gates=settings.gates,
            coalesce_settings=settings.coalesce,
            aggregation_settings={},
            config={},
        )

        graph = _build_fork_coalesce_graph(config, settings)

        orchestrator = Orchestrator(db=db)
        result = orchestrator.run(config, graph=graph, settings=settings)

        # Should have processed rows
        assert result.rows_processed == 1
        # The row was forked (parent gets FORKED outcome)
        assert result.rows_forked == 1
        # The fork children were coalesced
        assert result.rows_coalesced == 1

        # Sink should have received merged output
        assert len(sink.rows) >= 1
        merged = sink.rows[0]
        assert merged["id"] == 1
        assert merged["value"] == 100

    def test_partial_branch_coverage_non_coalesced_branches_reach_sink(
        self,
        db: LandscapeDB,
    ) -> None:
        """Branches not in coalesce should still reach output sink."""
        settings = ElspethSettings(
            datasource=DatasourceSettings(
                plugin="list_source",
                options={},
            ),
            sinks={"output": SinkSettings(plugin="collect_sink", options={})},
            output_sink="output",
            gates=[
                GateSettings(
                    name="forker",
                    condition="True",
                    routes={"true": "fork", "false": "continue"},
                    fork_to=["path_a", "path_b", "path_c"],  # 3 branches
                ),
            ],
            coalesce=[
                CoalesceSettings(
                    name="merge_ab",
                    branches=["path_a", "path_b"],  # Only 2 coalesce
                    policy="require_all",
                    merge="union",
                ),
            ],
        )

        source = ListSource([{"id": 1}])
        sink = CollectSink()

        config = PipelineConfig(
            source=source,
            transforms=[],
            sinks={"output": sink},
            gates=settings.gates,
            coalesce_settings=settings.coalesce,
            aggregation_settings={},
            config={},
        )

        graph = _build_fork_coalesce_graph(config, settings)

        orchestrator = Orchestrator(db=db)
        result = orchestrator.run(config, graph=graph, settings=settings)

        # Should have:
        # - 1 row processed (1 source row)
        # - 1 forked (parent row was forked)
        # - 1 merged token from path_a + path_b coalesce
        # - 1 direct token from path_c (not in coalesce)
        assert result.rows_processed == 1
        assert result.rows_forked == 1
        assert result.rows_coalesced == 1

        # Sink should have 2 rows:
        # - 1 merged token from path_a + path_b
        # - 1 direct token from path_c
        assert len(sink.rows) == 2

    def test_fork_coalesce_with_transform(
        self,
        db: LandscapeDB,
    ) -> None:
        """Fork/coalesce with a transform before the fork gate."""
        from elspeth.plugins.results import TransformResult

        class EnrichedSchema(_TestSchema):
            id: int
            value: int
            enriched: bool

        class EnrichTransform(BaseTransform):
            name = "enrich"
            input_schema = _TestSchema
            output_schema = EnrichedSchema

            def __init__(self) -> None:
                super().__init__({})

            def process(self, row: Any, ctx: Any) -> TransformResult:
                return TransformResult.success(
                    {
                        **row,
                        "enriched": True,
                    }
                )

        settings = ElspethSettings(
            datasource=DatasourceSettings(
                plugin="list_source",
                options={},
            ),
            sinks={
                "output": SinkSettings(plugin="collect_sink", options={}),
            },
            output_sink="output",
            gates=[
                GateSettings(
                    name="forker",
                    condition="True",
                    routes={"true": "fork", "false": "continue"},
                    fork_to=["path_a", "path_b"],
                ),
            ],
            coalesce=[
                CoalesceSettings(
                    name="merge_results",
                    branches=["path_a", "path_b"],
                    policy="require_all",
                    merge="union",
                ),
            ],
        )

        source = ListSource([{"id": 1, "value": 42}])
        transform = EnrichTransform()
        sink = CollectSink()

        config = PipelineConfig(
            source=source,
            transforms=[transform],
            sinks={"output": sink},
            gates=settings.gates,
            coalesce_settings=settings.coalesce,
            aggregation_settings={},
            config={},
        )

        graph = _build_fork_coalesce_graph(config, settings)

        orchestrator = Orchestrator(db=db)
        result = orchestrator.run(config, graph=graph, settings=settings)

        # Verify processing worked
        assert result.rows_processed == 1
        assert result.rows_forked == 1
        assert result.rows_coalesced == 1

        # Merged output should have enriched=True from transform
        assert len(sink.rows) >= 1
        merged = sink.rows[0]
        assert merged["id"] == 1
        assert merged["value"] == 42
        assert merged["enriched"] is True

    def test_multiple_source_rows_fork_coalesce(
        self,
        db: LandscapeDB,
    ) -> None:
        """Multiple source rows each fork and coalesce independently."""
        settings = ElspethSettings(
            datasource=DatasourceSettings(
                plugin="list_source",
                options={},
            ),
            sinks={
                "output": SinkSettings(plugin="collect_sink", options={}),
            },
            output_sink="output",
            gates=[
                GateSettings(
                    name="forker",
                    condition="True",
                    routes={"true": "fork", "false": "continue"},
                    fork_to=["path_a", "path_b"],
                ),
            ],
            coalesce=[
                CoalesceSettings(
                    name="merge_results",
                    branches=["path_a", "path_b"],
                    policy="require_all",
                    merge="union",
                ),
            ],
        )

        # 3 source rows
        source = ListSource([{"id": 1}, {"id": 2}, {"id": 3}])
        sink = CollectSink()

        config = PipelineConfig(
            source=source,
            transforms=[],
            sinks={"output": sink},
            gates=settings.gates,
            coalesce_settings=settings.coalesce,
            aggregation_settings={},
            config={},
        )

        graph = _build_fork_coalesce_graph(config, settings)

        orchestrator = Orchestrator(db=db)
        result = orchestrator.run(config, graph=graph, settings=settings)

        # Each source row forks and coalesces
        assert result.rows_processed == 3
        assert result.rows_forked == 3
        assert result.rows_coalesced == 3

        # 3 merged outputs (one per source row)
        assert len(sink.rows) == 3

        # Verify all IDs are present
        ids = {row["id"] for row in sink.rows}
        assert ids == {1, 2, 3}


class TestCoalesceAuditTrail:
    """Test that coalesce operations are properly recorded in audit trail."""

    @pytest.fixture
    def db(self) -> LandscapeDB:
        return LandscapeDB.in_memory()

    def test_coalesce_records_node_states(
        self,
        db: LandscapeDB,
    ) -> None:
        """Coalesce should record node states for consumed tokens."""
        settings = ElspethSettings(
            datasource=DatasourceSettings(
                plugin="list_source",
                options={},
            ),
            sinks={
                "output": SinkSettings(plugin="collect_sink", options={}),
            },
            output_sink="output",
            gates=[
                GateSettings(
                    name="forker",
                    condition="True",
                    routes={"true": "fork", "false": "continue"},
                    fork_to=["path_a", "path_b"],
                ),
            ],
            coalesce=[
                CoalesceSettings(
                    name="merge_results",
                    branches=["path_a", "path_b"],
                    policy="require_all",
                    merge="union",
                ),
            ],
        )

        source = ListSource([{"id": 1}])
        sink = CollectSink()

        config = PipelineConfig(
            source=source,
            transforms=[],
            sinks={"output": sink},
            gates=settings.gates,
            coalesce_settings=settings.coalesce,
            aggregation_settings={},
            config={},
        )

        graph = _build_fork_coalesce_graph(config, settings)

        orchestrator = Orchestrator(db=db)
        result = orchestrator.run(config, graph=graph, settings=settings)

        # Verify run completed
        assert result.status == "completed"
        assert result.rows_coalesced == 1

        # Query the audit trail for node states at the coalesce node
        from elspeth.core.landscape.schema import node_states_table, nodes_table

        with db.connection() as conn:
            # Find coalesce node
            nodes_result = conn.execute(
                nodes_table.select().where(nodes_table.c.node_type == "coalesce")
            ).fetchall()

            assert len(nodes_result) == 1
            coalesce_node = nodes_result[0]
            assert "merge_results" in coalesce_node.plugin_name

            # Find node states for coalesce
            states_result = conn.execute(
                node_states_table.select().where(
                    node_states_table.c.node_id == coalesce_node.node_id
                )
            ).fetchall()

            # Should have 2 node states (one for each consumed token from path_a and path_b)
            assert len(states_result) == 2
            for state in states_result:
                assert state.status == "completed"
