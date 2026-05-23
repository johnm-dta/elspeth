"""Tests for DAG builder validation guards — rejection paths.

The builder constructs execution graphs from plugin instances and validates
graph topology. These tests exercise the REJECTION paths: ambiguous continue
fallthrough, coalesce routing errors, and observed-mode short-circuits in
union merge.
"""

from __future__ import annotations

from typing import Any, Literal

import pytest

from elspeth.contracts.schema import FieldDefinition, SchemaConfig
from elspeth.core.dag.models import GraphValidationError


class TestCoalesceUnionMergeObservedShortCircuit:
    """Union merge must collapse to observed mode when ANY branch is observed.

    When a coalesce does union merge and at least one branch has mode=observed,
    the entire merged schema must become observed — the declared branch's fields
    cannot be guaranteed because the observed branch's output is unknown.
    """

    def _build_coalesce_with_schemas(
        self,
        branch_schemas: dict[str, SchemaConfig],
    ) -> SchemaConfig:
        """Simulate the builder's union merge logic with SchemaConfig objects."""
        seen_types: dict[str, tuple[Literal["str", "int", "float", "bool", "any"], bool, str]] = {}
        all_observed = False
        for branch_name, schema_cfg in branch_schemas.items():
            if schema_cfg.is_observed:
                all_observed = True
                break
            if schema_cfg.fields is None:
                continue
            for fd in schema_cfg.fields:
                if fd.name in seen_types:
                    prior_type, _prior_req, prior_branch = seen_types[fd.name]
                    if prior_type != fd.field_type:
                        raise GraphValidationError(f"Type mismatch for {fd.name}")
                    if not fd.required:
                        seen_types[fd.name] = (prior_type, False, prior_branch)
                else:
                    seen_types[fd.name] = (fd.field_type, fd.required, branch_name)

        if all_observed or not seen_types:
            return SchemaConfig(mode="observed", fields=None)
        merged_fields = tuple(FieldDefinition(name=name, field_type=ftype, required=req) for name, (ftype, req, _) in seen_types.items())
        return SchemaConfig(mode="flexible", fields=merged_fields)

    def test_observed_branch_collapses_union_to_observed(self) -> None:
        """One declared + one observed branch → entire merge is observed."""
        result = self._build_coalesce_with_schemas(
            {
                "branch_a": SchemaConfig(
                    mode="flexible",
                    fields=(FieldDefinition("id", "int"), FieldDefinition("score", "float")),
                ),
                "branch_b": SchemaConfig(mode="observed", fields=None),
            }
        )
        assert result.is_observed
        assert result.fields is None

    def test_observed_first_branch_collapses_union_to_observed(self) -> None:
        """Observed branch FIRST still collapses — order should not matter."""
        result = self._build_coalesce_with_schemas(
            {
                "branch_a": SchemaConfig(mode="observed", fields=None),
                "branch_b": SchemaConfig(
                    mode="flexible",
                    fields=(FieldDefinition("id", "int"), FieldDefinition("label", "str")),
                ),
            }
        )
        assert result.is_observed
        assert result.fields is None


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
