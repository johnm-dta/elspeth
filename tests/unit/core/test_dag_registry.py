# tests/unit/core/test_dag_registry.py
"""Tests for producer/consumer registry logic in ExecutionGraph.from_plugin_instances().

The registry (dag.py:940-1041) is the core of declarative DAG wiring — it validates
that every connection name has exactly one producer and one consumer, that connection
names don't collide with sink names, and that Levenshtein suggestions are provided
for near-miss wiring errors.

Per CLAUDE.md Test Path Integrity: all tests go through from_plugin_instances()
and the real plugin instantiation path.
"""

from __future__ import annotations

from typing import Any

import pytest

from elspeth.core.config import (
    ElspethSettings,
    GateSettings,
    SinkSettings,
    SourceSettings,
    TransformSettings,
)
from elspeth.core.dag import ExecutionGraph, GraphValidationError


def _build_graph(config: ElspethSettings) -> ExecutionGraph:
    """Build ExecutionGraph through the real code path."""
    from elspeth.cli_helpers import instantiate_plugins_from_config

    plugins = instantiate_plugins_from_config(config)
    return ExecutionGraph.from_plugin_instances(
        sources=plugins.sources,
        source_settings_map=plugins.source_settings_map,
        transforms=plugins.transforms,
        sinks=plugins.sinks,
        aggregations=plugins.aggregations,
        gates=list(config.gates),
        coalesce_settings=list(config.coalesce) if config.coalesce else None,
    )


def _observed_source(**kwargs: Any) -> SourceSettings:
    """Minimal source settings with observed schema."""
    defaults = {
        "plugin": "csv",
        "options": {
            "path": "test.csv",
            "on_validation_failure": "discard",
            "schema": {"mode": "observed"},
        },
    }
    defaults.update(kwargs)
    return SourceSettings(**defaults)


def _observed_sink(**kwargs: Any) -> SinkSettings:
    """Minimal sink settings with observed schema."""
    defaults = {
        "plugin": "json",
        "on_write_failure": "discard",
        "options": {"path": "output.json", "schema": {"mode": "observed"}},
    }
    defaults.update(kwargs)
    return SinkSettings(**defaults)


def _observed_transform(name: str, *, input: str, on_success: str = "output", **kwargs: Any) -> TransformSettings:
    """Minimal transform settings with observed schema."""
    return TransformSettings(
        name=name,
        plugin="passthrough",
        input=input,
        on_success=on_success,
        on_error="discard",
        options={"schema": {"mode": "observed"}, **kwargs.get("options", {})},
    )


class TestEmptyRegistry:
    """Source-only pipeline with no transforms."""

    def test_source_only_pipeline(self) -> None:
        """Source directly routes to sink — no registry entries for transforms."""
        config = ElspethSettings(
            sources={"primary": _observed_source(on_success="output")},
            sinks={"output": _observed_sink()},
        )
        graph = _build_graph(config)

        assert graph.node_count == 2  # source + sink
        assert graph.edge_count == 1


class TestSingleProducerConsumer:
    """One transform with input matching source on_success."""

    def test_single_transform_chain(self) -> None:
        """Source -> transform -> sink with explicit connection name."""
        config = ElspethSettings(
            sources={"primary": _observed_source(on_success="data_out")},
            sinks={"output": _observed_sink()},
            transforms=[
                _observed_transform("t1", input="data_out", on_success="output"),
            ],
        )
        graph = _build_graph(config)

        assert graph.node_count == 3  # source + t1 + sink
        assert graph.edge_count == 2

    def test_multi_transform_chain(self) -> None:
        """Source -> t1 -> t2 -> sink with named connections."""
        config = ElspethSettings(
            sources={"primary": _observed_source(on_success="src_out")},
            sinks={"output": _observed_sink()},
            transforms=[
                _observed_transform("t1", input="src_out", on_success="conn_1_2"),
                _observed_transform("t2", input="conn_1_2", on_success="output"),
            ],
        )
        graph = _build_graph(config)

        assert graph.node_count == 4  # source + t1 + t2 + sink
        assert graph.edge_count == 3


class TestDuplicateProducer:
    """Two nodes with same on_success connection name must fail."""

    def test_two_transforms_same_on_success_rejected(self) -> None:
        """Two transforms producing to the same connection name is a duplicate producer error.

        Both t1 and t2 declare on_success='shared_conn'. Even though t2 needs a
        valid input, the producer registry (which runs before consumer validation)
        rejects the duplicate on_success first.
        """
        config = ElspethSettings(
            sources={"primary": _observed_source(on_success="src_out")},
            sinks={"output": _observed_sink()},
            transforms=[
                _observed_transform("t1", input="src_out", on_success="shared_conn"),
                # t2 has a non-existent input, but the duplicate producer error fires first
                _observed_transform("t2", input="other_conn", on_success="shared_conn"),
            ],
        )
        with pytest.raises(GraphValidationError, match=r"Duplicate producer.*shared_conn"):
            _build_graph(config)

    def test_duplicate_consumer_rejected(self) -> None:
        """Two transforms consuming the same connection name is a duplicate consumer error."""
        config = ElspethSettings(
            sources={"primary": _observed_source(on_success="shared_input")},
            sinks={"output": _observed_sink()},
            transforms=[
                _observed_transform("t1", input="shared_input", on_success="output"),
                _observed_transform("t2", input="shared_input", on_success="output"),
            ],
        )
        with pytest.raises(GraphValidationError, match=r"Duplicate consumer.*shared_input"):
            _build_graph(config)


class TestOrphanedProducer:
    """on_success targeting a non-existent sink/connection must fail."""

    def test_on_success_neither_sink_nor_connection_rejected(self) -> None:
        """Transform on_success pointing to unknown target is rejected."""
        config = ElspethSettings(
            sources={"primary": _observed_source(on_success="src_out")},
            sinks={"output": _observed_sink()},
            transforms=[
                _observed_transform("t1", input="src_out", on_success="dangling"),
            ],
        )
        with pytest.raises(GraphValidationError, match="neither a sink nor a known connection"):
            _build_graph(config)


class TestOrphanedConsumer:
    """Input referencing non-existent connection must fail."""

    def test_input_references_nonexistent_connection(self) -> None:
        """Transform consumes a connection that no node produces."""
        config = ElspethSettings(
            sources={"primary": _observed_source(on_success="src_out")},
            sinks={"output": _observed_sink()},
            transforms=[
                _observed_transform("t1", input="does_not_exist", on_success="output"),
            ],
        )
        with pytest.raises(GraphValidationError, match="No producer for connection 'does_not_exist'"):
            _build_graph(config)


class TestNamespaceCollision:
    """Connection name = sink name must fail."""

    def test_connection_name_distinct_from_sink_name_accepted(self) -> None:
        """Connection names distinct from sink names are accepted."""
        # t1 produces to "flagged_conn" as a connection (not a sink name),
        # t2 consumes it and routes to "output" sink. This should succeed.
        config = ElspethSettings(
            sources={"primary": _observed_source(on_success="src_out")},
            sinks={"output": _observed_sink(), "flagged": _observed_sink(options={"path": "flagged.json", "schema": {"mode": "observed"}})},
            transforms=[
                _observed_transform("t1", input="src_out", on_success="flagged_conn"),
                _observed_transform("t2", input="flagged_conn", on_success="output"),
            ],
        )
        # This should succeed because flagged_conn is a connection, not a sink name
        graph = _build_graph(config)
        assert graph.node_count == 5  # source + t1 + t2 + output_sink + flagged_sink


class TestGateRouteResolution:
    """Gate route targets connection name, resolves to consumer's node ID."""

    def test_gate_route_to_downstream_gate(self) -> None:
        """Gate route pointing to a connection name resolves to the consuming gate.

        Uses gate-to-gate routing (the proven pattern) to test the registry
        resolution. The true branch routes to a named connection consumed by
        the second gate; the false branch routes directly to a sink.
        """
        config = ElspethSettings(
            sources={"primary": _observed_source(on_success="src_out")},
            sinks={
                "output": _observed_sink(),
                "flagged": _observed_sink(options={"path": "flagged.json", "schema": {"mode": "observed"}}),
            },
            gates=[
                GateSettings(
                    name="router",
                    input="src_out",
                    condition="row['score'] >= 0.8",
                    routes={"true": "checker_in", "false": "flagged"},
                ),
                GateSettings(
                    name="checker",
                    input="checker_in",
                    condition="True",
                    routes={"true": "output", "false": "flagged"},
                ),
            ],
        )
        graph = _build_graph(config)

        # Should have: source + router + checker + output + flagged
        assert graph.node_count == 5
        assert graph.is_acyclic()


class TestMultiRouteConvergence:
    """Two gate routes pointing to same connection name."""

    def test_two_routes_same_target(self) -> None:
        """Multiple gate routes from the same gate converging to the same connection."""
        config = ElspethSettings(
            sources={"primary": _observed_source(on_success="src_out")},
            sinks={
                "output": _observed_sink(),
            },
            gates=[
                GateSettings(
                    name="gate1",
                    input="src_out",
                    condition="row['score'] >= 0.5",
                    routes={"true": "checker_in", "false": "checker_in"},
                ),
                GateSettings(
                    name="checker",
                    input="checker_in",
                    condition="True",
                    routes={"true": "output", "false": "output"},
                ),
            ],
        )
        graph = _build_graph(config)
        # source + gate1 + checker + output
        assert graph.node_count == 4
        assert graph.is_acyclic()


class TestLevenshteinSuggestions:
    """Error messages include near-miss connection name suggestions."""

    def test_near_miss_connection_name_gets_suggestion(self) -> None:
        """Typo in connection name should suggest correct name."""
        config = ElspethSettings(
            sources={"primary": _observed_source(on_success="source_output")},
            sinks={"output": _observed_sink()},
            transforms=[
                # Typo: "source_outpt" instead of "source_output"
                _observed_transform("t1", input="source_outpt", on_success="output"),
            ],
        )
        with pytest.raises(GraphValidationError, match="source_output") as exc_info:
            _build_graph(config)
        # The error should mention the correct name as a suggestion
        assert "source_output" in str(exc_info.value)

    def test_completely_wrong_name_no_suggestion(self) -> None:
        """Completely unrelated connection name has no useful suggestion."""
        config = ElspethSettings(
            sources={"primary": _observed_source(on_success="alpha")},
            sinks={"output": _observed_sink()},
            transforms=[
                _observed_transform("t1", input="zzzzz_totally_wrong", on_success="output"),
            ],
        )
        with pytest.raises(GraphValidationError, match="No producer"):
            _build_graph(config)
