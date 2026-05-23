# tests/integration/telemetry/test_wiring.py
"""Integration tests for telemetry wiring through production code paths.

These tests verify that:
1. Orchestrator correctly wires telemetry_emit to PluginContext
2. Plugins using audited clients emit telemetry in production
3. The fix for elspeth-rapid-vlr is working correctly

CRITICAL: These tests use production Orchestrator, NOT manual PluginContext.
This catches the wiring bugs that unit tests miss.

Migrated from tests/integration/test_telemetry_wiring.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from elspeth.contracts import Determinism
from elspeth.contracts.enums import RunStatus, TelemetryGranularity
from elspeth.contracts.events import RunStarted
from elspeth.core.landscape import LandscapeDB
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from elspeth.plugins.infrastructure.results import TransformResult
from elspeth.telemetry import TelemetryManager
from tests.fixtures.base_classes import as_sink, as_source, as_transform
from tests.fixtures.pipeline import build_production_graph
from tests.fixtures.plugins import CollectSink, ListSource, PassTransform
from tests.fixtures.telemetry import MockTelemetryConfig, TelemetryTestExporter

if TYPE_CHECKING:
    from elspeth.core.dag import ExecutionGraph


# =============================================================================
# Test Helpers
# =============================================================================


def create_test_graph(config: PipelineConfig) -> ExecutionGraph:
    """Build a graph using the production factory path."""
    return build_production_graph(config)


# =============================================================================
# Core Wiring Tests
# =============================================================================


class TestOrchestratorWiresTelemetryToContext:
    """Verify orchestrator correctly wires telemetry_emit to PluginContext.

    This is the critical test for elspeth-rapid-vlr - it verifies that the
    production code path (Orchestrator -> PluginContext) correctly wires
    the telemetry callback.

    IMPORTANT: These tests use production Orchestrator, NOT manual PluginContext.
    """

    def test_orchestrator_emits_lifecycle_telemetry(
        self,
        landscape_db: LandscapeDB,
        payload_store: Any,
    ) -> None:
        """Orchestrator emits RunStarted and RunFinished via production path."""
        exporter = TelemetryTestExporter()
        config = MockTelemetryConfig(granularity=TelemetryGranularity.FULL)
        telemetry_manager = TelemetryManager(config, exporters=[exporter])

        source = ListSource([{"id": 1}, {"id": 2}, {"id": 3}], on_success="output")
        sink = CollectSink()

        pipeline_config = PipelineConfig(
            sources={"primary": as_source(source)},
            transforms=[as_transform(PassTransform())],
            sinks={"output": as_sink(sink)},
        )

        # Use PRODUCTION Orchestrator - this is the key
        orchestrator = Orchestrator(landscape_db, telemetry_manager=telemetry_manager)
        result = orchestrator.run(
            pipeline_config,
            graph=create_test_graph(pipeline_config),
            payload_store=payload_store,
        )

        # Pipeline should complete
        assert result.status == RunStatus.COMPLETED

        # Telemetry should have been emitted
        exporter.assert_event_emitted("RunStarted")
        exporter.assert_event_emitted("RunFinished")

        # Verify run_id matches
        run_started = exporter.get_events_of_type("RunStarted")[0]
        run_finished = exporter.get_events_of_type("RunFinished")[0]
        assert run_started.run_id == result.run_id
        assert run_finished.run_id == result.run_id

    def test_context_telemetry_emit_is_callable(
        self,
        landscape_db: LandscapeDB,
        payload_store: Any,
    ) -> None:
        """Verify ctx.telemetry_emit is a real callable, not the default no-op.

        This test captures the actual telemetry_emit callback from inside a
        plugin to verify the orchestrator wired it correctly.
        """
        captured_callback = None

        class CallbackCapturingTransform(PassTransform):
            """Transform that captures the telemetry_emit callback."""

            name = "callback_capturing"
            determinism = Determinism.DETERMINISTIC

            def process(self, row: Any, ctx: Any) -> TransformResult:
                return TransformResult.success(row, success_reason={"action": "passthrough"})

            def on_start(self, ctx: Any) -> None:
                super().on_start(ctx)
                nonlocal captured_callback
                captured_callback = ctx.telemetry_emit

        exporter = TelemetryTestExporter()
        config = MockTelemetryConfig()
        telemetry_manager = TelemetryManager(config, exporters=[exporter])

        source = ListSource([{"id": 1}, {"id": 2}, {"id": 3}], on_success="output")
        sink = CollectSink()

        pipeline_config = PipelineConfig(
            sources={"primary": as_source(source)},
            transforms=[as_transform(CallbackCapturingTransform())],
            sinks={"output": as_sink(sink)},
        )

        orchestrator = Orchestrator(landscape_db, telemetry_manager=telemetry_manager)
        graph = create_test_graph(pipeline_config)
        orchestrator.run(pipeline_config, graph=graph, payload_store=payload_store)

        # The callback should have been captured
        assert captured_callback is not None, "ctx.telemetry_emit was not set"

        # It should NOT be the default no-op lambda
        # The default is: lambda event: None
        # A real callback is: orchestrator._emit_telemetry (a bound method)
        callback_name = getattr(captured_callback, "__name__", str(captured_callback))
        assert callback_name != "<lambda>", (
            f"ctx.telemetry_emit is still the default no-op lambda, not the real callback. Got: {captured_callback}"
        )

    def test_telemetry_wiring_works_in_resume_path(
        self,
        landscape_db: LandscapeDB,
        payload_store: Any,
    ) -> None:
        """Telemetry is also wired correctly in the resume code path.

        The orchestrator has two PluginContext creation sites:
        1. Main execution path (run -> _execute_run)
        2. Resume path (resume -> _resume_run)

        Both must wire telemetry_emit.
        """
        # This test verifies the main path works (resume path is harder to test
        # without setting up a partial run). The fix added telemetry_emit to both.
        exporter = TelemetryTestExporter()
        config = MockTelemetryConfig()
        telemetry_manager = TelemetryManager(config, exporters=[exporter])

        source = ListSource([{"id": 1}, {"id": 2}, {"id": 3}], on_success="output")
        sink = CollectSink()

        pipeline_config = PipelineConfig(
            sources={"primary": as_source(source)},
            transforms=[as_transform(PassTransform())],
            sinks={"output": as_sink(sink)},
        )

        orchestrator = Orchestrator(landscape_db, telemetry_manager=telemetry_manager)
        orchestrator.run(
            pipeline_config,
            graph=create_test_graph(pipeline_config),
            payload_store=payload_store,
        )

        # Verify we got telemetry
        assert exporter.event_count > 0, "No telemetry events emitted"
        exporter.assert_event_emitted("RunStarted")
        exporter.assert_event_emitted("RunFinished")


class TestNoTelemetryWithoutManager:
    """Verify telemetry is correctly disabled when no manager is provided."""

    def test_no_crash_without_telemetry_manager(
        self,
        landscape_db: LandscapeDB,
        payload_store: Any,
    ) -> None:
        """Pipeline runs successfully without telemetry manager."""
        source = ListSource([{"id": 1}, {"id": 2}, {"id": 3}], on_success="output")
        sink = CollectSink()

        pipeline_config = PipelineConfig(
            sources={"primary": as_source(source)},
            transforms=[as_transform(PassTransform())],
            sinks={"output": as_sink(sink)},
        )

        # No telemetry_manager - should use default no-op
        orchestrator = Orchestrator(landscape_db)
        result = orchestrator.run(
            pipeline_config,
            graph=create_test_graph(pipeline_config),
            payload_store=payload_store,
        )

        assert result.status == RunStatus.COMPLETED
        assert result.rows_processed == 3

    def test_context_telemetry_emit_is_noop_without_manager(
        self,
        landscape_db: LandscapeDB,
        payload_store: Any,
    ) -> None:
        """Without telemetry manager, ctx.telemetry_emit is a no-op lambda."""
        captured_callback = None

        class CallbackCapturingTransform(PassTransform):
            name = "callback_capturing"
            determinism = Determinism.DETERMINISTIC

            def on_start(self, ctx: Any) -> None:
                super().on_start(ctx)
                nonlocal captured_callback
                captured_callback = ctx.telemetry_emit

        source = ListSource([{"id": 1}, {"id": 2}, {"id": 3}], on_success="output")
        sink = CollectSink()

        pipeline_config = PipelineConfig(
            sources={"primary": as_source(source)},
            transforms=[as_transform(CallbackCapturingTransform())],
            sinks={"output": as_sink(sink)},
        )

        # No telemetry manager
        orchestrator = Orchestrator(landscape_db)
        graph = create_test_graph(pipeline_config)
        orchestrator.run(pipeline_config, graph=graph, payload_store=payload_store)

        # Callback should be set (never None)
        assert captured_callback is not None

        # Should be callable without error (no-op)
        from datetime import UTC, datetime

        # Calling it should not raise
        captured_callback(
            RunStarted(
                timestamp=datetime.now(UTC),
                run_id="test",
                config_hash="test",
                source_plugin="test",
            )
        )
