# tests/integration/pipeline/orchestrator/test_gate_to_gate_routing.py
"""Regression tests for gate-to-gate route jump resolution.

When gate1 routes to gate2 via a PROCESSING_NODE destination, the jump
resolution walker must recognize that gates are self-routing — they
determine sink destinations at execution time via their own routes config.
Previously, the walker only recognized TransformProtocol.on_success and
coalesce nodes as sink sources, causing an OrchestrationInvariantError
when the target gate was terminal (no continue edge).

Target code:
- src/elspeth/engine/dag_navigator.py: DAGNavigator.resolve_jump_target_sink()

Tests:
1. Config gate routes to terminal config gate — must not crash
2. Gate chain where downstream gate CONTINUEs to transform → sink
3. Existing transform chain behavior preserved (regression guard)
"""

from __future__ import annotations

from typing import Any

from elspeth.contracts import Determinism, RunStatus
from elspeth.core.config import GateSettings
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from elspeth.testing import make_pipeline_row
from tests.fixtures.base_classes import _TestSchema, as_sink, as_source, as_transform
from tests.fixtures.landscape import make_landscape_db
from tests.fixtures.pipeline import build_production_graph
from tests.fixtures.plugins import CollectSink, ListSource


class TestConfigGateToConfigGate:
    """Config gate routes to a terminal config gate — the crash scenario."""

    def test_gate_routes_to_terminal_gate(self, payload_store) -> None:
        """gate1 routes to gate2, gate2 routes to sinks — must not crash."""
        db = make_landscape_db()

        # gate1: routes "forward" to gate2's input connection
        gate1 = GateSettings(
            name="gate1",
            input="source_out",
            condition="True",
            routes={"true": "gate2_in", "false": "default"},
        )

        # gate2: terminal gate — routes directly to sinks
        gate2 = GateSettings(
            name="gate2",
            input="gate2_in",
            condition="row['value'] > 50",
            routes={"true": "high", "false": "default"},
        )

        source = ListSource([{"value": 10}, {"value": 100}, {"value": 30}])
        default_sink = CollectSink(name="default")
        high_sink = CollectSink(name="high")

        config = PipelineConfig(
            sources={"primary": as_source(source)},
            transforms=[],
            sinks={"default": as_sink(default_sink), "high": as_sink(high_sink)},
            gates=[gate1, gate2],
        )

        orchestrator = Orchestrator(db)
        result = orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        # elspeth-5069612f3c: gate route_to_sink is intentional MOVE
        # (rows_routed_success). Gate-routed-only run -> COMPLETED.
        assert result.status == RunStatus.COMPLETED
        # gate2 routes: value > 50 → high, else → default
        assert len(high_sink.results) == 1  # value=100
        assert len(default_sink.results) == 2  # value=10, value=30

    def test_gate_chain_three_deep(self, payload_store) -> None:
        """gate1 → gate2 → gate3, all config gates, gate3 terminal."""
        db = make_landscape_db()

        gate1 = GateSettings(
            name="gate1",
            input="source_out",
            condition="True",
            routes={"true": "gate2_in", "false": "default"},
        )
        gate2 = GateSettings(
            name="gate2",
            input="gate2_in",
            condition="True",
            routes={"true": "gate3_in", "false": "default"},
        )
        gate3 = GateSettings(
            name="gate3",
            input="gate3_in",
            condition="row['value'] > 50",
            routes={"true": "high", "false": "default"},
        )

        source = ListSource([{"value": 10}, {"value": 100}])
        default_sink = CollectSink(name="default")
        high_sink = CollectSink(name="high")

        config = PipelineConfig(
            sources={"primary": as_source(source)},
            transforms=[],
            sinks={"default": as_sink(default_sink), "high": as_sink(high_sink)},
            gates=[gate1, gate2, gate3],
        )

        orchestrator = Orchestrator(db)
        result = orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        # elspeth-5069612f3c: gate route_to_sink is intentional MOVE
        # (rows_routed_success). Gate-routed-only run -> COMPLETED.
        assert result.status == RunStatus.COMPLETED
        assert len(high_sink.results) == 1  # value=100
        assert len(default_sink.results) == 1  # value=10


class TestGateToGateWithDownstreamTransform:
    """Source → transform → gate1, where gate1 routes to terminal gate2."""

    def test_transform_then_gate_routes_to_terminal_gate(self, payload_store) -> None:
        """source → transform → gate1 (routes to gate2) → gate2 (routes to sinks)."""
        from elspeth.contracts.schema_contract import PipelineRow
        from elspeth.plugins.infrastructure.base import BaseTransform
        from elspeth.plugins.infrastructure.results import TransformResult

        db = make_landscape_db()

        class TagTransform(BaseTransform):
            """Adds a 'tagged' field to prove the transform ran."""

            name = "tag_transform"
            determinism = Determinism.DETERMINISTIC
            input_schema = _TestSchema
            output_schema = _TestSchema

            def __init__(self) -> None:
                super().__init__({"schema": {"mode": "observed"}})

            def process(self, row: PipelineRow, ctx: Any) -> TransformResult:
                data = row.to_dict()
                data["tagged"] = True
                return TransformResult.success(make_pipeline_row(data), success_reason={"action": "tagged"})

        # gate1 receives from transform, routes to gate2's input
        gate1 = GateSettings(
            name="gate1",
            input="transform_out",
            condition="True",
            routes={"true": "gate2_in", "false": "default"},
        )
        # gate2 is terminal — routes directly to sinks
        gate2 = GateSettings(
            name="gate2",
            input="gate2_in",
            condition="row['value'] > 50",
            routes={"true": "high", "false": "default"},
        )

        source = ListSource([{"value": 42}, {"value": 100}])
        default_sink = CollectSink(name="default")
        high_sink = CollectSink(name="high")

        transform = TagTransform()

        config = PipelineConfig(
            sources={"primary": as_source(source)},
            transforms=[as_transform(transform)],
            sinks={"default": as_sink(default_sink), "high": as_sink(high_sink)},
            gates=[gate1, gate2],
        )

        orchestrator = Orchestrator(db)
        result = orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        # elspeth-5069612f3c: gate route_to_sink is intentional MOVE
        # (rows_routed_success). Gate-routed-only run -> COMPLETED.
        assert result.status == RunStatus.COMPLETED
        # gate2: value > 50 → high, else → default
        assert len(high_sink.results) == 1  # value=100
        assert len(default_sink.results) == 1  # value=42


class TestGateRouteToTransformChain:
    """Regression guard: gate routes to a transform chain (existing behavior)."""

    def test_gate_routes_to_sink_directly(self, payload_store) -> None:
        """Config gate routes to sinks without jump — baseline test."""
        db = make_landscape_db()

        gate = GateSettings(
            name="router",
            input="source_out",
            condition="row['value'] > 50",
            routes={"true": "high", "false": "default"},
        )

        source = ListSource([{"value": 10}, {"value": 100}])
        default_sink = CollectSink(name="default")
        high_sink = CollectSink(name="high")

        config = PipelineConfig(
            sources={"primary": as_source(source)},
            transforms=[],
            sinks={"default": as_sink(default_sink), "high": as_sink(high_sink)},
            gates=[gate],
        )

        orchestrator = Orchestrator(db)
        result = orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        # elspeth-5069612f3c: gate route_to_sink is intentional MOVE
        # (rows_routed_success). Gate-routed-only run -> COMPLETED.
        assert result.status == RunStatus.COMPLETED
        assert len(high_sink.results) == 1
        assert len(default_sink.results) == 1
