"""Integrity guard tests for DAG edge/route metadata."""

import pytest

from elspeth.contracts import RouteDestination, RoutingMode
from elspeth.contracts.types import NodeID, SinkName
from elspeth.core.dag import ExecutionGraph, GraphValidationError


class TestDAGEdgeDataIntegrity:
    """Tier-1 graph metadata must fail fast when malformed."""

    def test_get_branch_to_sink_map_raises_on_missing_mode(self) -> None:
        graph = ExecutionGraph()
        graph.add_node("gate_1", node_type="gate", plugin_name="gate", config={"routes": {"branch_a": "out"}})
        graph.add_node("sink_1", node_type="sink", plugin_name="json")
        graph.set_sink_id_map({SinkName("out"): NodeID("sink_1")})
        graph.add_edge("gate_1", "sink_1", label="branch_a", mode=RoutingMode.COPY)

        # Mutate internal graph directly — get_nx_graph() returns a frozen copy
        edge_data = graph._graph.get_edge_data("gate_1", "sink_1", "branch_a")
        assert edge_data is not None
        del edge_data["mode"]

        with pytest.raises(KeyError, match="mode"):
            graph.get_branch_to_sink_map()

    def test_get_terminal_sink_map_raises_on_missing_label(self) -> None:
        graph = ExecutionGraph()
        graph.add_node("transform_1", node_type="transform", plugin_name="passthrough")
        graph.add_node("sink_1", node_type="sink", plugin_name="json")
        graph.set_sink_id_map({SinkName("out"): NodeID("sink_1")})
        graph.add_edge("transform_1", "sink_1", label="on_success", mode=RoutingMode.MOVE)

        # Mutate internal graph directly — get_nx_graph() returns a frozen copy
        edge_data = graph._graph.get_edge_data("transform_1", "sink_1", "on_success")
        assert edge_data is not None
        del edge_data["label"]

        with pytest.raises(KeyError, match="label"):
            graph.get_terminal_sink_map()

    def test_is_sink_node_raises_on_missing_node(self) -> None:
        graph = ExecutionGraph()
        graph.add_node("sink_1", node_type="sink", plugin_name="json")

        with pytest.raises(KeyError, match="Node not found"):
            graph.is_sink_node(NodeID("missing_node"))


class TestRouteResolutionIntegrity:
    """Route-resolution map must be complete before runtime."""

    def test_route_resolution_validation_rejects_missing_label(self) -> None:
        graph = ExecutionGraph()
        graph.add_node(
            "gate_1",
            node_type="gate",
            plugin_name="config_gate:router",
            config={"routes": {"true": "output", "false": "output"}},
        )
        graph.add_route_resolution_entry(NodeID("gate_1"), "true", RouteDestination.sink(SinkName("output")))

        with pytest.raises(GraphValidationError, match="has no destination in route resolution map"):
            graph._validate_route_resolution_map_complete()

    def test_from_plugin_instances_invokes_route_resolution_validation(self, plugin_manager, monkeypatch) -> None:
        from elspeth.cli_helpers import instantiate_plugins_from_config
        from elspeth.core.config import ElspethSettings, GateSettings, SinkSettings, SourceSettings

        called = {"value": False}

        def fail_validator(self: ExecutionGraph) -> None:
            called["value"] = True
            raise GraphValidationError("sentinel route-resolution validation")

        monkeypatch.setattr(ExecutionGraph, "_validate_route_resolution_map_complete", fail_validator)

        settings = ElspethSettings(
            source=SourceSettings(
                plugin="csv",
                on_success="to_gate",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"mode": "observed"},
                },
            ),
            gates=[
                GateSettings(
                    name="router",
                    input="to_gate",
                    condition="row['score'] > 0.5",
                    routes={"true": "output", "false": "output"},
                )
            ],
            sinks={"output": SinkSettings(plugin="json", options={"path": "out.json", "schema": {"mode": "observed"}})},
        )

        plugins = instantiate_plugins_from_config(settings)
        with pytest.raises(GraphValidationError, match="sentinel route-resolution validation"):
            ExecutionGraph.from_plugin_instances(
                source=plugins["source"],
                source_settings=plugins["source_settings"],
                transforms=plugins["transforms"],
                sinks=plugins["sinks"],
                aggregations=plugins["aggregations"],
                gates=list(settings.gates),
            )

        assert called["value"] is True


class TestForkBranchIntegrity:
    """Fork branch names must be globally unique across all gates."""

    def test_fork_branch_names_must_be_unique_across_gates(self, plugin_manager) -> None:
        from elspeth.cli_helpers import instantiate_plugins_from_config
        from elspeth.core.config import ElspethSettings, GateSettings, SinkSettings, SourceSettings

        settings = ElspethSettings(
            source=SourceSettings(
                plugin="csv",
                on_success="source_out",
                options={
                    "path": "test.csv",
                    "on_validation_failure": "discard",
                    "schema": {"mode": "observed"},
                },
            ),
            sinks={
                "output": SinkSettings(plugin="json", options={"path": "output.json", "schema": {"mode": "observed"}}),
                "path_a": SinkSettings(plugin="json", options={"path": "path_a.json", "schema": {"mode": "observed"}}),
            },
            gates=[
                GateSettings(
                    name="forker_a",
                    input="source_out",
                    condition="True",
                    routes={"true": "fork", "false": "gate_b_in"},
                    fork_to=["path_a"],
                ),
                GateSettings(
                    name="forker_b",
                    input="gate_b_in",
                    condition="True",
                    routes={"true": "fork", "false": "output"},
                    fork_to=["path_a"],
                ),
            ],
        )

        plugins = instantiate_plugins_from_config(settings)
        with pytest.raises(GraphValidationError, match=r"globally unique across all gates"):
            ExecutionGraph.from_plugin_instances(
                source=plugins["source"],
                source_settings=plugins["source_settings"],
                transforms=plugins["transforms"],
                sinks=plugins["sinks"],
                aggregations=plugins["aggregations"],
                gates=list(settings.gates),
                coalesce_settings=settings.coalesce,
            )


class TestWiredTransformIntegrity:
    """WiredTransform must match plugin instance and settings metadata."""

    def test_wired_transform_mismatched_plugin_name_raises(self) -> None:
        from elspeth.core.config import TransformSettings
        from elspeth.core.dag import WiredTransform

        class DummyTransform:
            name = "actual_plugin"

        with pytest.raises(ValueError, match="WiredTransform mismatch"):
            WiredTransform(
                plugin=DummyTransform(),  # type: ignore[arg-type]
                settings=TransformSettings(
                    name="t0",
                    plugin="expected_plugin",
                    input="source_out",
                    on_success="output",
                    on_error="discard",
                    options={},
                ),
            )
