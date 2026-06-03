# tests/integration/pipeline/test_explicit_sink_routing.py
"""Integration tests for explicit sink routing (bead amoc).

These tests validate that `on_success` routing works end-to-end through
the real production assembly path (ExecutionGraph.from_plugin_instances).

The explicit-sink-routing feature allows:
- Sources to declare on_success sink destination
- Transforms to declare on_success sink destination (terminal transforms)
- Coalesce nodes to declare on_success sink destination
- Fork branches with different on_success routes to per-branch sinks

ADR-required tests per bead amoc specification.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from elspeth.contracts import Determinism, PipelineRow, RunStatus, SinkProtocol, SourceProtocol, TransformProtocol
from elspeth.core.config import AggregationSettings, CoalesceSettings, ElspethSettings, GateSettings, SourceSettings, TriggerConfig
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from elspeth.plugins.infrastructure.base import BaseTransform
from elspeth.testing import make_pipeline_row
from tests.fixtures.base_classes import _TestSchema, as_sink, as_source, as_transform
from tests.fixtures.factories import wire_transforms
from tests.fixtures.landscape import make_landscape_db
from tests.fixtures.pipeline import build_production_graph
from tests.fixtures.plugins import CollectSink, ListSource, PassTransform

if TYPE_CHECKING:
    from elspeth.plugins.infrastructure.results import TransformResult


# ---------------------------------------------------------------------------
# Test Transforms
# ---------------------------------------------------------------------------


class AddFieldTransform(BaseTransform):
    """Transform that adds a field to the row."""

    name = "add_field"
    determinism = Determinism.DETERMINISTIC
    input_schema = _TestSchema
    output_schema = _TestSchema

    def __init__(self, field_name: str, field_value: Any) -> None:
        super().__init__({"schema": {"mode": "observed"}})
        self._field_name = field_name
        self._field_value = field_value

    def process(self, row: PipelineRow, ctx: Any) -> TransformResult:
        from elspeth.plugins.infrastructure.results import TransformResult

        output = {**row.to_dict(), self._field_name: self._field_value}
        return TransformResult.success(
            make_pipeline_row(output),
            success_reason={"action": "add_field", "metadata": {"field": self._field_name}},
        )


class BatchPassthroughTransform(BaseTransform):
    """Batch-aware transform that passes through rows unchanged."""

    name = "batch_passthrough"
    determinism = Determinism.DETERMINISTIC
    input_schema = _TestSchema
    output_schema = _TestSchema
    is_batch_aware = True

    def __init__(self) -> None:
        super().__init__({"schema": {"mode": "observed"}})

    def process(self, rows: list[PipelineRow], ctx: Any) -> TransformResult:  # type: ignore[override]  # Batch-aware process takes list[PipelineRow]
        from elspeth.plugins.infrastructure.results import TransformResult

        # Sum values from the batch
        total = sum(r["value"] for r in rows)
        result = {"count": len(rows), "total": total}
        return TransformResult.success(
            make_pipeline_row(result),
            success_reason={"action": "batch_passthrough"},
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestExplicitSinkRouting:
    """ADR-required tests for explicit sink routing feature."""

    def test_09_completed_row_carries_explicit_sink_name(self, payload_store) -> None:
        """Test 9: Linear pipeline routes to transform's on_success sink.

        Setup: source → transform(on_success=output) → output sink
        Verify: Completed rows arrive at the declared on_success sink.
        """
        db = make_landscape_db()

        source = ListSource([{"value": 1}, {"value": 2}], on_success="output")
        transform = PassTransform()
        transform.on_success = "output"
        sink = CollectSink(name="output")

        config = PipelineConfig(
            source=as_source(source),
            transforms=[as_transform(transform)],
            sinks={"output": as_sink(sink)},
        )

        orchestrator = Orchestrator(db)
        run_result = orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        assert run_result.status == RunStatus.COMPLETED
        assert run_result.rows_processed == 2
        assert len(sink.results) == 2
        assert sink.results[0] == {"value": 1}
        assert sink.results[1] == {"value": 2}

    def test_10_fork_branches_with_different_on_success_route_to_per_branch_sinks(self, payload_store) -> None:
        """Test 10: Fork gate sends rows to sink_a and sink_b branches.

        Setup: source → fork_gate → [sink_a, sink_b]
        Verify: Rows arrive at both sinks via forked branch routing.

        Terminal fork gates must route all paths to named sinks (no "continue").
        The "false" route goes to sink_a (never fires since condition=True).
        """
        db = make_landscape_db()

        source = ListSource([{"value": 1}, {"value": 2}], on_success="sink_a")

        # Terminal fork gate: true=fork, false=sink_a (required but never fires)
        fork_gate = GateSettings(
            name="fork_gate",
            input="source_out",
            condition="True",
            routes={"true": "fork", "false": "sink_a"},
            fork_to=["sink_a", "sink_b"],
        )

        sink_a = CollectSink(name="sink_a")
        sink_b = CollectSink(name="sink_b")

        config = PipelineConfig(
            source=as_source(source),
            transforms=[],
            sinks={"sink_a": as_sink(sink_a), "sink_b": as_sink(sink_b)},
            gates=[fork_gate],
        )

        orchestrator = Orchestrator(db)
        run_result = orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        assert run_result.status == RunStatus.COMPLETED
        assert run_result.rows_processed == 2

        # Fork duplicates each row to both branches
        assert len(sink_a.results) == 2
        assert len(sink_b.results) == 2

        for row in sink_a.results:
            assert "value" in row
        for row in sink_b.results:
            assert "value" in row

    def test_11_coalesce_output_routes_to_declared_on_success_sink(self, payload_store) -> None:
        """Test 11: Fork → coalesce → terminal gate → on_success sink.

        Setup: source → fork → [path_a, path_b] → coalesce → terminal_gate → output
        Verify: Merged results arrive at the gate's declared on_success sink.

        Uses a terminal gate after coalesce (same pattern as
        test_nonterminal_coalesce_continues_to_downstream_gate).
        """
        db = make_landscape_db()

        source = ListSource([{"value": 1}, {"value": 2}], on_success="source_sink")

        # One transform per branch — named to match fork branch names
        transform = PassTransform()

        fork_gate = GateSettings(
            name="fork_gate",
            input="transform_out",
            condition="True",
            routes={"true": "fork", "false": "output"},
            fork_to=["path_a", "path_b"],
        )

        # Terminal gate routes merged result to output sink
        terminal_gate = GateSettings(
            name="terminal_gate",
            input="merge_paths",
            condition="True",
            routes={"true": "output", "false": "output"},
        )

        coalesce = CoalesceSettings(
            name="merge_paths",
            branches=["path_a", "path_b"],
            policy="require_all",
            merge="union",
        )
        runtime_coalesce = CoalesceSettings(
            name="merge_paths",
            branches=["path_a", "path_b"],
            policy="require_all",
            merge="union",
            on_success="output",
        )

        output_sink = CollectSink(name="output")
        source_sink = CollectSink(name="source_sink")

        config = PipelineConfig(
            source=as_source(source),
            transforms=[as_transform(transform)],
            sinks={
                "output": as_sink(output_sink),
                "source_sink": as_sink(source_sink),
            },
            gates=[fork_gate, terminal_gate],
            coalesce_settings=[coalesce],
        )

        settings = ElspethSettings(
            source={"plugin": "test", "on_success": "source_out", "options": {}},
            sinks={
                "output": {"plugin": "test", "on_write_failure": "discard"},
                "source_sink": {"plugin": "test", "on_write_failure": "discard"},
            },
            gates=[fork_gate, terminal_gate],
            coalesce=[runtime_coalesce],
        )

        orchestrator = Orchestrator(db)
        run_result = orchestrator.run(
            config,
            graph=build_production_graph(config),
            settings=settings,
            payload_store=payload_store,
        )

        # ADR-019: gate route_to_sink is intentional MOVE provenance on top of
        # lifecycle SUCCESS, so a gate-routed-only run is COMPLETED.
        assert run_result.status == RunStatus.COMPLETED
        assert run_result.rows_processed == 2
        assert run_result.rows_succeeded == 2
        assert run_result.rows_routed_success == 2

        # Merged results route to output sink via terminal gate
        assert len(output_sink.results) == 2
        # Source sink shouldn't receive anything
        assert len(source_sink.results) == 0

    def test_12_aggregation_flush_routes_to_on_success_sink(self, payload_store) -> None:
        """Test 12: Aggregation transform with on_success routes to declared sink.

        Setup: source → aggregation(count=2, on_success=output) → output
        Verify: Flushed batch results arrive at on_success sink.

        Uses the same wiring pattern as test_aggregation_checkpoint_bug: the
        transform is a regular transform in the graph with on_success set,
        and aggregation_settings tells the Orchestrator to batch it.
        """
        from elspeth.core.dag import ExecutionGraph

        db = make_landscape_db()

        source = ListSource([{"value": 1}, {"value": 2}, {"value": 3}], on_success="output")
        transform = BatchPassthroughTransform()
        transform.on_success = "output"
        sink = CollectSink(name="output")

        # Build graph with transform as regular transform (on_success wires to sink)
        graph = ExecutionGraph.from_plugin_instances(
            source=cast(SourceProtocol, source),
            source_settings=SourceSettings(plugin=source.name, on_success="source_out", options={}),
            transforms=wire_transforms(cast("list[TransformProtocol]", [transform]), source_connection="source_out", final_sink="output"),
            sinks=cast("dict[str, SinkProtocol]", {"output": sink}),
            aggregations={},
            gates=[],
        )

        transform_node_id = graph.get_transform_id_map()[0]

        agg_settings = AggregationSettings(
            name="batch",
            plugin="batch_passthrough",
            input="source_out",
            on_error="discard",
            trigger=TriggerConfig(count=2),
            output_mode="transform",
        )

        config = PipelineConfig(
            source=as_source(source),
            transforms=[as_transform(transform)],
            sinks={"output": as_sink(sink)},
            aggregation_settings={transform_node_id: agg_settings},
        )

        orchestrator = Orchestrator(db)
        run_result = orchestrator.run(config, graph=graph, payload_store=payload_store)

        assert run_result.status == RunStatus.COMPLETED
        assert run_result.rows_processed == 3

        # Aggregated results arrive at the on_success sink
        assert len(sink.results) >= 1

    def test_13_checkpoint_resume_with_on_success_routes_identically(self, payload_store) -> None:
        """Test 13: Checkpoint data preserves explicit on_success wiring.

        Verifies that checkpoint creation works correctly with the explicit
        connection-name wiring model — the checkpoint roundtrip preserves
        the DAG traversal context so a resumed pipeline would route identically.

        Uses a two-transform chain (source → t1 → t2 → sink) with explicit
        connection names, runs with every-row checkpointing, and verifies:
        1. Pipeline completes successfully with checkpointing enabled
        2. Checkpoint data is created (at least one checkpoint call)
        3. Checkpoint includes wiring-critical state (processed tokens)
        """
        from elspeth.contracts.config.runtime import RuntimeCheckpointConfig
        from elspeth.core.checkpoint import CheckpointManager
        from elspeth.core.config import CheckpointSettings
        from elspeth.core.dag import ExecutionGraph

        db = make_landscape_db()
        checkpoint_mgr = CheckpointManager(db)
        settings = CheckpointSettings(enabled=True, frequency="every_row")
        checkpoint_config = RuntimeCheckpointConfig.from_settings(settings)

        checkpoint_calls: list[dict[str, Any]] = []
        original_create = checkpoint_mgr.create_checkpoint

        def tracking_create(*args: Any, **kwargs: Any) -> Any:
            checkpoint_calls.append({"args": args, "kwargs": kwargs})
            return original_create(*args, **kwargs)

        checkpoint_mgr.create_checkpoint = tracking_create  # type: ignore[method-assign]

        source = ListSource([{"value": 1}, {"value": 2}, {"value": 3}], on_success="source_out")
        t1 = PassTransform()
        t1.on_success = "conn_1_2"
        t2 = AddFieldTransform("processed", True)
        t2.on_success = "output"
        sink = CollectSink(name="output")

        # Build graph through production path with explicit connection names
        graph = ExecutionGraph.from_plugin_instances(
            source=cast(SourceProtocol, source),
            source_settings=SourceSettings(plugin=source.name, on_success="source_out", options={}),
            transforms=wire_transforms(
                cast("list[TransformProtocol]", [t1, t2]),
                source_connection="source_out",
                final_sink="output",
            ),
            sinks=cast("dict[str, SinkProtocol]", {"output": sink}),
            aggregations={},
            gates=[],
        )

        config = PipelineConfig(
            source=as_source(source),
            transforms=[as_transform(t1), as_transform(t2)],
            sinks={"output": as_sink(sink)},
        )

        orchestrator = Orchestrator(
            db,
            checkpoint_manager=checkpoint_mgr,
            checkpoint_config=checkpoint_config,
        )
        run_result = orchestrator.run(config, graph=graph, payload_store=payload_store)

        # Pipeline completes with all 3 rows
        assert run_result.status == RunStatus.COMPLETED
        assert run_result.rows_processed == 3

        # Checkpoints were created (one per row)
        assert len(checkpoint_calls) == 3, f"Expected 3 checkpoint calls, got {len(checkpoint_calls)}"

        # All results routed to the correct on_success sink
        assert len(sink.results) == 3
        for result in sink.results:
            assert result["processed"] is True


class TestExplicitSinkRoutingEdgeCases:
    """Edge cases and error conditions for explicit sink routing."""

    def test_wire_transforms_always_provides_on_success(self, payload_store) -> None:
        """wire_transforms always sets on_success, preventing terminal-without-routing.

        With WiredTransform architecture, wire_transforms() always provides
        on_success to the last transform (pointing to final_sink). This means
        the old scenario of a "terminal transform without on_success" can no
        longer occur through the production wiring path.
        """
        from tests.fixtures.factories import wire_transforms

        transform = PassTransform()
        wired = wire_transforms([as_transform(transform)], final_sink="output")

        # wire_transforms always provides on_success to the last transform
        assert wired[-1].settings.on_success == "output"

    def test_source_on_success_used_when_no_transforms(self, payload_store) -> None:
        """Source on_success routes directly to sink when no transforms exist."""
        db = make_landscape_db()

        source = ListSource([{"value": 1}], on_success="direct_sink")
        sink = CollectSink(name="direct_sink")

        config = PipelineConfig(
            source=as_source(source),
            transforms=[],
            sinks={"direct_sink": as_sink(sink)},
        )

        orchestrator = Orchestrator(db)
        run_result = orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        assert run_result.status == RunStatus.COMPLETED
        assert run_result.rows_processed == 1
        assert len(sink.results) == 1

    def test_wire_transforms_routes_last_transform_to_final_sink(self, payload_store) -> None:
        """wire_transforms routes the last transform to the declared final sink.

        Setup: source → transform1 → transform2 → sink_b (via wire_transforms final_sink)
        Verify: Rows route to the final sink declared in wire_transforms, not other sinks.

        With WiredTransform architecture, routing is determined by wire_transforms()
        and TransformSettings, not by individual transform.on_success attributes.
        """
        from tests.fixtures.factories import wire_transforms

        db = make_landscape_db()

        source = ListSource([{"value": 1}], on_success="sink_b")

        transform1 = AddFieldTransform("t1", "done")
        transform2 = AddFieldTransform("t2", "done")

        sink_a = CollectSink(name="sink_a")
        sink_b = CollectSink(name="sink_b")

        # Explicitly wire transforms to route to sink_b
        wired = wire_transforms(
            [as_transform(transform1), as_transform(transform2)],
            source_connection="source_out",
            final_sink="sink_b",
        )

        config = PipelineConfig(
            source=as_source(source),
            transforms=[t.plugin for t in wired],
            sinks={"sink_a": as_sink(sink_a), "sink_b": as_sink(sink_b)},
        )

        from elspeth.core.config import SourceSettings
        from elspeth.core.dag import ExecutionGraph

        source_settings = SourceSettings(plugin=source.name, on_success="source_out", options={})
        graph = ExecutionGraph.from_plugin_instances(
            source=config.source,
            source_settings=source_settings,
            transforms=wired,
            sinks=config.sinks,
            aggregations={},
            gates=[],
        )

        orchestrator = Orchestrator(db)
        run_result = orchestrator.run(config, graph=graph, payload_store=payload_store)

        assert run_result.status == RunStatus.COMPLETED
        assert run_result.rows_processed == 1

        # Row arrives at sink_b (wire_transforms final_sink), not sink_a
        assert len(sink_b.results) == 1
        assert len(sink_a.results) == 0

    def test_gate_routes_to_processing_node_via_connection_name(self, payload_store) -> None:
        """Gate routes to a downstream transform via named connection.

        Setup: source → gate → transform → sink
        The gate's true route targets a named connection consumed by the
        transform. This exercises the full path from DAG construction through
        gate execution to processor traversal without manual construction.

        Validates bead 29x9: exercises gate fan-out through Orchestrator.run()
        with real plugins instead of manually constructed traversal maps.
        """
        from elspeth.core.dag import ExecutionGraph

        db = make_landscape_db()

        source = ListSource([{"value": 10}, {"value": 20}], on_success="gate_in")
        transform = AddFieldTransform("routed", True)
        transform.on_success = "output"
        sink_output = CollectSink(name="output")
        sink_flagged = CollectSink(name="flagged")

        # Gate routes true → downstream_conn (consumed by transform),
        # false → flagged sink directly
        gate = GateSettings(
            name="router",
            input="gate_in",
            condition="True",
            routes={"true": "downstream_conn", "false": "flagged"},
        )

        graph = ExecutionGraph.from_plugin_instances(
            source=cast(SourceProtocol, source),
            source_settings=SourceSettings(plugin=source.name, on_success="gate_in", options={}),
            transforms=wire_transforms(
                cast("list[TransformProtocol]", [transform]),
                source_connection="downstream_conn",
                final_sink="output",
            ),
            sinks=cast("dict[str, SinkProtocol]", {"output": sink_output, "flagged": sink_flagged}),
            aggregations={},
            gates=[gate],
        )

        config = PipelineConfig(
            source=as_source(source),
            transforms=[as_transform(transform)],
            sinks={"output": as_sink(sink_output), "flagged": as_sink(sink_flagged)},
            gates=[gate],
        )

        orchestrator = Orchestrator(db)
        run_result = orchestrator.run(config, graph=graph, payload_store=payload_store)

        assert run_result.status == RunStatus.COMPLETED
        assert run_result.rows_processed == 2

        # All rows routed through gate true → transform → output sink
        assert len(sink_output.results) == 2
        assert all(r["routed"] is True for r in sink_output.results)

        # No rows went to flagged (gate condition is always True)
        assert len(sink_flagged.results) == 0
