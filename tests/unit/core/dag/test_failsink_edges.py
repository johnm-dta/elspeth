"""Test __failsink__ DIVERT edge creation in DAG builder."""

from __future__ import annotations

from typing import Any, cast

from elspeth.contracts import SinkProtocol, SourceProtocol
from elspeth.contracts.enums import RoutingMode
from elspeth.contracts.types import SinkName
from elspeth.core.config import SourceSettings
from elspeth.core.dag.graph import ExecutionGraph
from tests.fixtures.plugins import CollectSink, ListSource


def _build_graph(
    *,
    primary_on_write_failure: str,
    failsink_on_write_failure: str = "discard",
    include_failsink: bool = True,
) -> ExecutionGraph:
    """Build a minimal graph with a primary sink and optional failsink."""
    source = ListSource([{"id": 1}], name="source", on_success="primary")
    source_settings = SourceSettings(plugin="list_source", on_success="primary", options={})

    primary = CollectSink("primary")
    primary._on_write_failure = primary_on_write_failure

    sinks: dict[str, Any] = {"primary": primary}

    if include_failsink:
        failsink = CollectSink("csv_failsink")
        failsink._on_write_failure = failsink_on_write_failure
        sinks["csv_failsink"] = failsink

    return ExecutionGraph.from_plugin_instances(
        source=cast(SourceProtocol, source),
        source_settings=source_settings,
        transforms=[],
        sinks=cast("dict[str, SinkProtocol]", sinks),
        aggregations={},
        gates=[],
    )


class TestFailsinkEdges:
    def test_failsink_divert_edge_created(self) -> None:
        """A sink with on_write_failure pointing to another sink creates a __failsink__ DIVERT edge."""
        graph = _build_graph(primary_on_write_failure="csv_failsink")
        edges = graph.get_edges()
        failsink_edges = [e for e in edges if e.label == "__failsink__"]
        assert len(failsink_edges) == 1
        assert failsink_edges[0].mode == RoutingMode.DIVERT

    def test_discard_no_failsink_edge(self) -> None:
        """A sink with on_write_failure='discard' creates no __failsink__ edge."""
        graph = _build_graph(primary_on_write_failure="discard", include_failsink=False)
        edges = graph.get_edges()
        failsink_edges = [e for e in edges if e.label == "__failsink__"]
        assert len(failsink_edges) == 0

    def test_failsink_edge_connects_correct_nodes(self) -> None:
        """The __failsink__ edge connects from the primary sink to the failsink."""
        graph = _build_graph(primary_on_write_failure="csv_failsink")
        edges = graph.get_edges()
        failsink_edges = [e for e in edges if e.label == "__failsink__"]
        assert len(failsink_edges) == 1
        edge = failsink_edges[0]
        # from_node should be the primary sink, to_node should be the failsink
        sink_id_map = graph.get_sink_id_map()
        assert edge.from_node == sink_id_map[SinkName("primary")]
        assert edge.to_node == sink_id_map[SinkName("csv_failsink")]
