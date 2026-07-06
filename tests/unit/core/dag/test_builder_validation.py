"""Tests for DAG builder validation guards — rejection paths.

The builder constructs execution graphs from plugin instances and validates
graph topology. These tests exercise the REJECTION paths: ambiguous continue
fallthrough, coalesce routing errors, and observed-mode short-circuits in
union merge.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest

from elspeth.contracts import RouteDestination
from elspeth.contracts.schema import FieldDefinition, SchemaConfig
from elspeth.contracts.types import BranchName, CoalesceName, NodeID
from elspeth.core.config import CoalesceSettings, GateSettings, SourceSettings, TransformSettings
from elspeth.core.dag import ExecutionGraph
from elspeth.core.dag.models import GraphValidationError
from elspeth.core.dag.wiring import WiredTransform


class _BuilderValidationMockSource:
    name = "mock_source"
    output_schema = None
    config: ClassVar[dict[str, Any]] = {"schema": {"mode": "observed"}}
    _on_validation_failure = "discard"
    on_success = "source_out"


class _BuilderValidationMockSink:
    name = "mock_sink"
    input_schema = None
    config: ClassVar[dict[str, Any]] = {}
    _on_write_failure = "discard"
    declared_required_fields: ClassVar[frozenset[str]] = frozenset()

    def _reset_diversion_log(self) -> None:
        pass


class _BuilderValidationTransform:
    input_schema = None
    output_schema = None
    on_error: str | None = None
    on_success: str | None = "output"
    declared_output_fields: ClassVar[frozenset[str]] = frozenset()
    passes_through_input = False

    def __init__(self, *, name: str, output_schema_config: SchemaConfig) -> None:
        self.name = name
        self.config = {"schema": {"mode": "observed"}}
        self._output_schema_config = output_schema_config


class TestCoalesceBranchPlanning:
    """Builder should store one coherent plan per coalesce branch."""

    def test_builder_freezes_graph_metadata_after_construction(self) -> None:
        source = _BuilderValidationMockSource()

        graph = ExecutionGraph.from_plugin_instances(
            sources={"primary": source},  # type: ignore[arg-type]
            source_settings_map={"primary": SourceSettings(plugin=source.name, on_success="output", options={})},
            transforms=[],
            sinks={"output": _BuilderValidationMockSink()},  # type: ignore[dict-item]
            aggregations={},
            gates=[],
            coalesce_settings=[],
        )

        source_node = graph.get_sources()[0]
        sink_node = graph.get_sinks()[0]
        node_step_map = graph.get_node_step_map()

        assert node_step_map[source_node] == 0
        assert graph.build_step_map() == node_step_map
        assert graph.get_node_step_map() == node_step_map

        with pytest.raises(GraphValidationError, match="build metadata is frozen"):
            graph.add_edge(source_node, sink_node, label="late_route")
        with pytest.raises(GraphValidationError, match="build metadata is frozen"):
            graph.set_node_output_schema(source_node, SchemaConfig(mode="observed", fields=None))
        with pytest.raises(GraphValidationError, match="build metadata is frozen"):
            graph.set_sink_id_map({})
        with pytest.raises(GraphValidationError, match="build metadata is frozen"):
            graph.add_route_resolution_entry(NodeID("gate"), "true", RouteDestination.discard())

    def test_branch_info_carries_identity_and_transform_branch_plan(self) -> None:
        source = _BuilderValidationMockSource()
        transform = _BuilderValidationTransform(
            name="slow_branch_transform",
            output_schema_config=SchemaConfig(mode="observed", fields=None),
        )

        graph = ExecutionGraph.from_plugin_instances(
            sources={"primary": source},  # type: ignore[arg-type]
            source_settings_map={"primary": SourceSettings(plugin=source.name, on_success="source_out", options={})},
            transforms=[
                WiredTransform(
                    plugin=transform,  # type: ignore[arg-type]
                    settings=TransformSettings(
                        name="slow_branch_transform",
                        plugin=transform.name,
                        input="slow_branch",
                        on_success="slow_out",
                        on_error="discard",
                        options={},
                    ),
                )
            ],
            sinks={"output": _BuilderValidationMockSink()},  # type: ignore[dict-item]
            aggregations={},
            gates=[
                GateSettings(
                    name="splitter",
                    input="source_out",
                    condition="True",
                    routes={"true": "fork", "false": "output"},
                    fork_to=["fast_branch", "slow_branch"],
                )
            ],
            coalesce_settings=[
                CoalesceSettings(
                    name="merge_results",
                    branches={"fast_branch": "fast_branch", "slow_branch": "slow_out"},
                    policy="require_all",
                    merge="union",
                    on_success="output",
                )
            ],
        )

        branch_info = graph.get_branch_info_map()

        fast_branch = branch_info[BranchName("fast_branch")]
        assert fast_branch.input_connection == "fast_branch"
        assert fast_branch.uses_transform_chain is False

        slow_branch = branch_info[BranchName("slow_branch")]
        assert slow_branch.input_connection == "slow_out"
        assert slow_branch.uses_transform_chain is True

        branch_schemas = graph.get_coalesce_branch_schemas(CoalesceName("merge_results"))
        assert set(branch_schemas) == {"fast_branch", "slow_branch"}
        assert fast_branch.schema == branch_schemas["fast_branch"]
        assert slow_branch.schema == branch_schemas["slow_branch"]


class TestCoalesceUnionMergeMixedObservedExplicit:
    """Builder union merge rejects mixed observed and typed explicit branches."""

    def _build_mixed_branch_graph(self, *, observed_first: bool) -> ExecutionGraph:
        source = _BuilderValidationMockSource()
        observed = _BuilderValidationTransform(
            name="observed_branch_transform",
            output_schema_config=SchemaConfig(mode="observed", fields=None),
        )
        explicit = _BuilderValidationTransform(
            name="explicit_branch_transform",
            output_schema_config=SchemaConfig(
                mode="flexible",
                fields=(FieldDefinition("id", "int"), FieldDefinition("score", "float")),
            ),
        )

        if observed_first:
            branch_order = ("observed_branch", "explicit_branch")
            branch_plugins = {"observed_branch": observed, "explicit_branch": explicit}
        else:
            branch_order = ("explicit_branch", "observed_branch")
            branch_plugins = {"explicit_branch": explicit, "observed_branch": observed}

        transforms = [
            WiredTransform(
                plugin=plugin,  # type: ignore[arg-type]
                settings=TransformSettings(
                    name=branch_name,
                    plugin=plugin.name,
                    input=branch_name,
                    on_success=f"{branch_name}_out",
                    on_error="discard",
                    options={},
                ),
            )
            for branch_name, plugin in branch_plugins.items()
        ]

        return ExecutionGraph.from_plugin_instances(
            sources={"primary": source},  # type: ignore[arg-type]
            source_settings_map={"primary": SourceSettings(plugin=source.name, on_success="source_out", options={})},
            transforms=transforms,
            sinks={"output": _BuilderValidationMockSink()},  # type: ignore[dict-item]
            aggregations={},
            gates=[
                GateSettings(
                    name="splitter",
                    input="source_out",
                    condition="True",
                    routes={"true": "fork", "false": "output"},
                    fork_to=list(branch_order),
                )
            ],
            coalesce_settings=[
                CoalesceSettings(
                    name="merge_results",
                    branches={branch_name: f"{branch_name}_out" for branch_name in branch_order},
                    policy="require_all",
                    merge="union",
                    on_success="output",
                )
            ],
        )

    def test_builder_rejects_observed_first_mixed_union_coalesce(self) -> None:
        """Builder reaches merge_union_fields and rejects observed-first mixed policy."""
        with pytest.raises(GraphValidationError, match="mixed observed/explicit schemas"):
            self._build_mixed_branch_graph(observed_first=True)

    def test_builder_rejects_explicit_first_mixed_union_coalesce(self) -> None:
        """Builder rejection is independent of branch declaration order."""
        with pytest.raises(GraphValidationError, match="mixed observed/explicit schemas"):
            self._build_mixed_branch_graph(observed_first=False)


class TestAmbiguousContinueFallthrough:
    """Multi-route gates with 2+ different processing targets suppress continue edges.

    When a gate routes to multiple different transforms, the builder cannot
    determine which one a continue_() action should target. It must NOT add
    a 'continue' edge for that gate.
    """

    def test_multi_route_gate_suppresses_continue_edge(self, plugin_manager: Any) -> None:
        """Gate routing to two different transforms must not get a continue edge."""
        from elspeth.cli_helpers import instantiate_plugins_from_config
        from elspeth.contracts.types import GateName
        from elspeth.core.config import (
            ElspethSettings,
            GateSettings,
            SinkSettings,
            SourceSettings,
            TransformSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        settings = ElspethSettings(
            sources={
                "primary": SourceSettings(
                    plugin="csv",
                    on_success="source_out",
                    options={
                        "path": "test.csv",
                        "on_validation_failure": "discard",
                        "schema": {"mode": "observed"},
                    },
                )
            },
            gates=[
                GateSettings(
                    name="router",
                    input="source_out",
                    condition="True",
                    routes={"true": "conn_a", "false": "conn_b"},
                ),
            ],
            transforms=[
                TransformSettings(
                    name="transform_a",
                    plugin="passthrough",
                    input="conn_a",
                    on_success="output",
                    on_error="discard",
                    options={"schema": {"mode": "observed"}},
                ),
                TransformSettings(
                    name="transform_b",
                    plugin="passthrough",
                    input="conn_b",
                    on_success="output",
                    on_error="discard",
                    options={"schema": {"mode": "observed"}},
                ),
            ],
            sinks={
                "output": SinkSettings(
                    plugin="json",
                    on_write_failure="discard",
                    options={"path": "output.json", "schema": {"mode": "observed"}},
                ),
            },
        )

        plugins = instantiate_plugins_from_config(settings)
        graph = ExecutionGraph.from_plugin_instances(
            sources=plugins.sources,
            source_settings_map=plugins.source_settings_map,
            transforms=plugins.transforms,
            sinks=plugins.sinks,
            aggregations=plugins.aggregations,
            gates=list(settings.gates),
        )

        router_id = graph.get_config_gate_id_map()[GateName("router")]
        edges = graph.get_edges()
        continue_edges = [e for e in edges if str(e.from_node) == str(router_id) and e.label == "continue"]
        assert continue_edges == [], f"Gate with ambiguous multi-route should have no continue edge, but found: {continue_edges}"

    def test_gate_discard_route_builds_virtual_route_destination(self, plugin_manager: Any) -> None:
        """Gate route destination 'discard' is terminal and must not need a sink node."""
        from elspeth.cli_helpers import instantiate_plugins_from_config
        from elspeth.contracts import RouteDestination
        from elspeth.contracts.types import GateName, NodeID
        from elspeth.core.config import (
            ElspethSettings,
            GateSettings,
            SinkSettings,
            SourceSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        settings = ElspethSettings(
            sources={
                "primary": SourceSettings(
                    plugin="csv",
                    on_success="source_out",
                    options={
                        "path": "test.csv",
                        "on_validation_failure": "discard",
                        "schema": {"mode": "observed"},
                    },
                )
            },
            gates=[
                GateSettings(
                    name="drop_nonmatches",
                    input="source_out",
                    condition="True",
                    routes={"true": "output", "false": "discard"},
                ),
            ],
            sinks={
                "output": SinkSettings(
                    plugin="json",
                    on_write_failure="discard",
                    options={"path": "output.json", "schema": {"mode": "observed"}},
                ),
            },
        )

        plugins = instantiate_plugins_from_config(settings)
        graph = ExecutionGraph.from_plugin_instances(
            sources=plugins.sources,
            source_settings_map=plugins.source_settings_map,
            transforms=plugins.transforms,
            sinks=plugins.sinks,
            aggregations=plugins.aggregations,
            gates=list(settings.gates),
        )

        gate_id = graph.get_config_gate_id_map()[GateName("drop_nonmatches")]
        assert graph.get_route_resolution_map()[(gate_id, "false")] == RouteDestination.discard()
        discard_edges = [edge for edge in graph.get_edges() if edge.label == "false" and edge.to_node == NodeID("discard")]
        assert discard_edges == []

    def test_gate_discard_route_prefers_real_sink_named_discard(self, plugin_manager: Any) -> None:
        """A real sink named 'discard' must not be shadowed by the virtual drop sentinel."""
        from elspeth.cli_helpers import instantiate_plugins_from_config
        from elspeth.contracts import RouteDestination
        from elspeth.contracts.types import GateName, SinkName
        from elspeth.core.config import (
            ElspethSettings,
            GateSettings,
            SinkSettings,
            SourceSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        settings = ElspethSettings(
            sources={
                "primary": SourceSettings(
                    plugin="csv",
                    on_success="source_out",
                    options={
                        "path": "test.csv",
                        "on_validation_failure": "discard",
                        "schema": {"mode": "observed"},
                    },
                )
            },
            gates=[
                GateSettings(
                    name="router",
                    input="source_out",
                    condition="True",
                    routes={"true": "discard", "false": "output"},
                ),
            ],
            sinks={
                "discard": SinkSettings(
                    plugin="json",
                    on_write_failure="discard",
                    options={"path": "discard.json", "schema": {"mode": "observed"}},
                ),
                "output": SinkSettings(
                    plugin="json",
                    on_write_failure="discard",
                    options={"path": "output.json", "schema": {"mode": "observed"}},
                ),
            },
        )

        plugins = instantiate_plugins_from_config(settings)
        graph = ExecutionGraph.from_plugin_instances(
            sources=plugins.sources,
            source_settings_map=plugins.source_settings_map,
            transforms=plugins.transforms,
            sinks=plugins.sinks,
            aggregations=plugins.aggregations,
            gates=list(settings.gates),
        )

        gate_id = graph.get_config_gate_id_map()[GateName("router")]
        assert graph.get_route_resolution_map()[(gate_id, "true")] == RouteDestination.sink(SinkName("discard"))
        assert graph.get_route_label(str(gate_id), SinkName("discard")) == "true"

    def test_gate_discard_route_prefers_real_connection_named_discard(self, plugin_manager: Any) -> None:
        """A real connection named 'discard' must route to its consumer before virtual drop fallback."""
        from elspeth.cli_helpers import instantiate_plugins_from_config
        from elspeth.contracts import RouteDestination
        from elspeth.contracts.types import GateName
        from elspeth.core.config import (
            ElspethSettings,
            GateSettings,
            SinkSettings,
            SourceSettings,
            TransformSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        settings = ElspethSettings(
            sources={
                "primary": SourceSettings(
                    plugin="csv",
                    on_success="source_out",
                    options={
                        "path": "test.csv",
                        "on_validation_failure": "discard",
                        "schema": {"mode": "observed"},
                    },
                )
            },
            gates=[
                GateSettings(
                    name="router",
                    input="source_out",
                    condition="True",
                    routes={"true": "discard", "false": "output"},
                ),
            ],
            transforms=[
                TransformSettings(
                    name="after_discard_route",
                    plugin="passthrough",
                    input="discard",
                    on_success="output",
                    on_error="discard",
                    options={"schema": {"mode": "observed"}},
                ),
            ],
            sinks={
                "output": SinkSettings(
                    plugin="json",
                    on_write_failure="discard",
                    options={"path": "output.json", "schema": {"mode": "observed"}},
                ),
            },
        )

        plugins = instantiate_plugins_from_config(settings)
        graph = ExecutionGraph.from_plugin_instances(
            sources=plugins.sources,
            source_settings_map=plugins.source_settings_map,
            transforms=plugins.transforms,
            sinks=plugins.sinks,
            aggregations=plugins.aggregations,
            gates=list(settings.gates),
        )

        gate_id = graph.get_config_gate_id_map()[GateName("router")]
        true_edges = [edge for edge in graph.get_edges() if edge.from_node == gate_id and edge.label == "true"]
        assert len(true_edges) == 1
        consumer_id = true_edges[0].to_node
        assert graph.get_route_resolution_map()[(gate_id, "true")] == RouteDestination.processing_node(consumer_id)


class TestCoalesceOnSuccessRejectsConnection:
    """Coalesce on_success must point to a sink, not a connection name.

    If on_success names a connection consumed by a transform, the builder
    must raise GraphValidationError — coalesce output cannot feed back
    into the processing graph via on_success.
    """

    def test_coalesce_on_success_to_connection_raises(self, plugin_manager: Any) -> None:
        """Coalesce on_success pointing to a transform connection raises."""
        from elspeth.cli_helpers import instantiate_plugins_from_config
        from elspeth.core.config import (
            CoalesceSettings,
            ElspethSettings,
            GateSettings,
            SinkSettings,
            SourceSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        # The coalesce on_success must name a connection that IS in the
        # consumers dict to hit the guard at builder.py line 683.
        # "source_out" is produced by the source and consumed by the gate,
        # so it is a valid consumer connection — but not a sink.
        settings = ElspethSettings(
            sources={
                "primary": SourceSettings(
                    plugin="csv",
                    on_success="source_out",
                    options={
                        "path": "test.csv",
                        "on_validation_failure": "discard",
                        "schema": {"mode": "observed"},
                    },
                )
            },
            gates=[
                GateSettings(
                    name="forker",
                    input="source_out",
                    condition="True",
                    routes={"true": "fork", "false": "output"},
                    fork_to=["path_a", "path_b"],
                ),
            ],
            coalesce=[
                CoalesceSettings(
                    name="merge_results",
                    branches=["path_a", "path_b"],
                    policy="require_all",
                    merge="union",
                    on_success="source_out",
                ),
            ],
            sinks={
                "output": SinkSettings(
                    plugin="json",
                    on_write_failure="discard",
                    options={"path": "output.json", "schema": {"mode": "observed"}},
                ),
            },
        )

        plugins = instantiate_plugins_from_config(settings)
        with pytest.raises(GraphValidationError, match="must point to a sink"):
            ExecutionGraph.from_plugin_instances(
                sources=plugins.sources,
                source_settings_map=plugins.source_settings_map,
                transforms=plugins.transforms,
                sinks=plugins.sinks,
                aggregations=plugins.aggregations,
                gates=list(settings.gates),
                coalesce_settings=settings.coalesce,
            )
