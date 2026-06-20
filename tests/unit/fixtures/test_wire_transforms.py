# tests/unit/fixtures/test_wire_transforms.py
"""Regression tests for fixture wiring helpers used by integration/e2e suites."""

from __future__ import annotations

from typing import cast

from tests.fixtures.factories import wire_transforms
from tests.fixtures.pipeline import build_aggregation_pipeline, build_fork_pipeline, build_linear_pipeline
from tests.fixtures.plugins import CollectSink, PassTransform

from elspeth.contracts import TransformProtocol
from elspeth.contracts.routing import RouteDestinationKind
from elspeth.core.config import AggregationSettings, CoalesceSettings, GateSettings, TriggerConfig


def test_wire_transforms_single_transform() -> None:
    transform = PassTransform()

    wired = wire_transforms(cast("list[TransformProtocol]", [transform]), source_connection="source_out", final_sink="default")

    assert len(wired) == 1
    assert wired[0].settings.name == "pass_transform_0"
    assert wired[0].settings.input == "source_out"
    assert wired[0].settings.on_success == "default"
    assert transform.on_success == "default"


def test_wire_transforms_linear_chain() -> None:
    transforms = [PassTransform() for _ in range(3)]

    wired = wire_transforms(cast("list[TransformProtocol]", transforms), source_connection="source_out", final_sink="output")

    assert [w.settings.input for w in wired] == ["source_out", "conn_0_1", "conn_1_2"]
    assert [w.settings.on_success for w in wired] == ["conn_0_1", "conn_1_2", "output"]
    assert [w.settings.name for w in wired] == ["pass_transform_0", "pass_transform_1", "pass_transform_2"]


def test_wire_transforms_five_transform_stress() -> None:
    transforms = [PassTransform() for _ in range(5)]

    wired = wire_transforms(cast("list[TransformProtocol]", transforms), source_connection="src", final_sink="sink")

    assert len(wired) == 5
    assert len({w.settings.name for w in wired}) == 5
    assert wired[0].settings.input == "src"
    assert wired[-1].settings.on_success == "sink"


def test_wire_transforms_allows_explicit_names() -> None:
    transforms = [PassTransform(), PassTransform()]

    wired = wire_transforms(
        cast("list[TransformProtocol]", transforms),
        source_connection="inbox",
        final_sink="outbox",
        names=["normalize", "classify"],
    )

    assert [w.settings.name for w in wired] == ["normalize", "classify"]
    assert wired[1].settings.on_success == "outbox"


def test_build_linear_pipeline_uses_wired_transforms() -> None:
    source_data = [{"value": 1}, {"value": 2}]
    transforms = [PassTransform(), PassTransform()]

    source, _transforms, sinks, graph = build_linear_pipeline(source_data, transforms=transforms, sink_name="default")

    assert source.on_success == "primary_out"
    assert list(sinks.keys()) == ["default"]
    assert len(graph.get_pipeline_node_sequence()) == 2
    assert list(graph.get_terminal_sink_map().values()) == ["default"]


def test_build_fork_pipeline_routes_to_branch_sinks() -> None:
    gate = GateSettings(
        name="router",
        input="primary_out",
        condition="row['value'] > 0",
        routes={"true": "sink_a", "false": "sink_b"},
    )

    _source, _transforms, _sinks, graph = build_fork_pipeline(
        [{"value": 1}],
        gate=gate,
        branch_transforms={},
        sinks={"sink_a": CollectSink("sink_a"), "sink_b": CollectSink("sink_b")},
        sink_name="sink_a",
    )
    route_map = graph.get_route_resolution_map()

    assert len(route_map) == 2
    assert all(dest.kind == RouteDestinationKind.SINK for dest in route_map.values())


def test_build_fork_pipeline_supports_join_via_coalesce() -> None:
    gate = GateSettings(
        name="fork_gate",
        input="primary_out",
        condition="True",
        routes={"true": "fork", "false": "output"},
        fork_to=["path_a", "path_b"],
    )
    coalesce = CoalesceSettings(
        name="merge_paths",
        branches=["path_a", "path_b"],
        policy="require_all",
        merge="union",
        on_success="output",
    )

    _source, _transforms, _sinks, graph = build_fork_pipeline(
        [{"value": 1}],
        gate=gate,
        branch_transforms={"path_a": [], "path_b": []},
        sinks={"output": CollectSink("output")},
        sink_name="output",
        coalesce_settings=[coalesce],
    )
    route_map = graph.get_route_resolution_map()

    assert len(route_map) == 2
    assert route_map[next(key for key in route_map if key[1] == "true")].kind == RouteDestinationKind.FORK
    assert any(node.node_type == "coalesce" for node in graph.get_nodes())


def test_build_aggregation_pipeline_sets_source_input_and_terminal_sink() -> None:
    agg_settings = AggregationSettings(
        name="batch",
        plugin="batch_transform",
        input="batch_in",
        on_success="output",
        on_error="discard",
        trigger=TriggerConfig(count=2),
        output_mode="transform",
        options={},
    )

    source, aggregations, _sinks, graph = build_aggregation_pipeline(
        [{"value": 1}, {"value": 2}],
        aggregation_transform=PassTransform(),
        aggregation_settings=agg_settings,
        sink_name="output",
    )

    assert source.on_success == "batch_in"
    assert "batch" in aggregations
    assert len(graph.get_pipeline_node_sequence()) == 1
    assert list(graph.get_terminal_sink_map().values()) == ["output"]
