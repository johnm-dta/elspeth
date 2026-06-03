"""Property tests for ExecutionGraph step-map numbering.

Step numbering is audit-critical. These tests verify that node-based traversal
preserves the legacy positional numbering contract:
source=0, transforms=1..N, aggregations next, config gates last.
"""

from __future__ import annotations

from typing import Any, cast

from hypothesis import given, settings
from hypothesis import strategies as st

from elspeth.contracts import Determinism, SinkProtocol, SourceProtocol, TransformProtocol
from elspeth.contracts.types import AggregationName, GateName
from elspeth.core.config import AggregationSettings, CoalesceSettings, GateSettings, SourceSettings
from elspeth.core.dag import ExecutionGraph
from elspeth.plugins.infrastructure.results import TransformResult
from tests.fixtures.base_classes import _TestTransformBase
from tests.fixtures.factories import wire_transforms
from tests.fixtures.plugins import CollectSink, ListSource


class _NoopTransform(_TestTransformBase):
    """Minimal transform used only to construct production-path graphs."""

    determinism = Determinism.DETERMINISTIC

    def __init__(self, *, name: str) -> None:
        super().__init__()
        self.name = name

    def process(self, row: Any, ctx: Any) -> TransformResult:
        return TransformResult.success(row, success_reason={"action": "noop"})


@st.composite
def _topologies(draw: st.DrawFn) -> tuple[int, int, int, bool]:
    """Generate valid topology parameters for step-map verification."""
    transform_count = draw(st.integers(min_value=1, max_value=10))
    aggregation_count = draw(st.integers(min_value=0, max_value=1))
    gate_count = draw(st.integers(min_value=0, max_value=3))
    with_fork_coalesce = draw(st.booleans())
    if with_fork_coalesce:
        gate_count = 1
    return transform_count, aggregation_count, gate_count, with_fork_coalesce


def _build_graph(
    *,
    transform_count: int,
    aggregation_count: int,
    gate_count: int,
    with_fork_coalesce: bool,
) -> tuple[ExecutionGraph, list[str], list[str]]:
    """Build graph through from_plugin_instances() with valid explicit routing."""
    source_on_success = "t0_in"
    source = ListSource([{"id": 1}], name="source", on_success=source_on_success)
    source_settings = SourceSettings(plugin="list_source", on_success=source_on_success, options={})
    sinks = {"results": CollectSink("results")}

    first_post_transform_connection = "results"
    if aggregation_count > 0:
        first_post_transform_connection = "agg0_in"
    elif gate_count > 0:
        first_post_transform_connection = "g0_in"

    transforms: list[_NoopTransform] = []
    transform_names: list[str] = []
    for i in range(transform_count):
        name = f"t{i}"
        transform_names.append(name)
        transforms.append(
            _NoopTransform(
                name=name,
            )
        )
    wired_transforms = wire_transforms(
        cast(list[TransformProtocol], transforms),
        source_connection="t0_in",
        final_sink=first_post_transform_connection,
        names=transform_names,
    )

    aggregation_names: list[str] = []
    aggregations: dict[str, tuple[_NoopTransform, AggregationSettings]] = {}
    for i in range(aggregation_count):
        agg_name = f"agg{i}"
        aggregation_names.append(agg_name)
        agg_input = "agg0_in" if i == 0 else f"agg{i}_in"
        agg_on_success = "results" if gate_count == 0 else "g0_in"
        aggregations[agg_name] = (
            _NoopTransform(
                name=f"{agg_name}_transform",
            ),
            AggregationSettings(
                name=agg_name,
                plugin=f"{agg_name}_plugin",
                input=agg_input,
                on_success=agg_on_success,
                on_error="discard",
                trigger={"count": 1},
            ),
        )

    gate_names: list[str] = []
    gates: list[GateSettings] = []
    for i in range(gate_count):
        gate_name = f"g{i}"
        gate_names.append(gate_name)
        is_terminal = i == gate_count - 1
        is_fork_gate = with_fork_coalesce and i == 0
        gate_input = "g0_in" if i == 0 else ("merge_paths" if with_fork_coalesce and i == 1 else f"g{i}_in")

        if is_fork_gate:
            routes: dict[str, str] = {"split": "fork", "done": "results"}
            gates.append(
                GateSettings(
                    name=gate_name,
                    input=gate_input,
                    condition="row['route']",
                    routes=routes,
                    fork_to=["path_a", "path_b"],
                )
            )
        else:
            routes = {"done": "results"} if is_terminal else {"cont": f"g{i + 1}_in"}
            gates.append(
                GateSettings(
                    name=gate_name,
                    input=gate_input,
                    condition="row['route']",
                    routes=routes,
                )
            )

    coalesce_settings: list[CoalesceSettings] | None = None
    if with_fork_coalesce:
        coalesce_settings = [
            CoalesceSettings(
                name="merge_paths",
                branches=["path_a", "path_b"],
                policy="require_all",
                merge="union",
                on_success="results" if gate_count == 1 else None,
            )
        ]

    graph = ExecutionGraph.from_plugin_instances(
        source=cast(SourceProtocol, source),
        source_settings=source_settings,
        transforms=wired_transforms,
        sinks=cast("dict[str, SinkProtocol]", sinks),
        aggregations=cast("dict[str, tuple[TransformProtocol, AggregationSettings]]", aggregations),
        gates=gates,
        coalesce_settings=coalesce_settings,
    )

    return graph, aggregation_names, gate_names


class TestStepMapSchemaContract:
    """Property checks that build_step_map satisfies Landscape DB step numbering contract."""

    @given(topology=_topologies())
    @settings(max_examples=120, deadline=None)
    def test_build_step_map_matches_schema_contract(
        self,
        topology: tuple[int, int, int, bool],
    ) -> None:
        transform_count, aggregation_count, gate_count, with_fork_coalesce = topology
        graph, aggregation_names, gate_names = _build_graph(
            transform_count=transform_count,
            aggregation_count=aggregation_count,
            gate_count=gate_count,
            with_fork_coalesce=with_fork_coalesce,
        )

        actual = graph.build_step_map()

        source_id = graph.get_source()
        assert source_id is not None
        assert actual[source_id] == 0

        transform_map = graph.get_transform_id_map()
        for i in range(transform_count):
            assert actual[transform_map[i]] == i + 1

        aggregation_map = graph.get_aggregation_id_map()
        for i, agg_name in enumerate(aggregation_names):
            assert actual[aggregation_map[AggregationName(agg_name)]] == transform_count + i + 1

        gate_map = graph.get_config_gate_id_map()
        gate_steps: list[int] = []
        for i, gate_name in enumerate(gate_names):
            gate_step = actual[gate_map[GateName(gate_name)]]
            gate_steps.append(gate_step)
            assert gate_step > transform_count + aggregation_count
            if i > 0:
                assert gate_steps[i - 1] < gate_step

        coalesce_map = graph.get_coalesce_id_map()
        if with_fork_coalesce:
            assert coalesce_map
            for coalesce_id in coalesce_map.values():
                assert actual[coalesce_id] > 0
