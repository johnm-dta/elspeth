# tests/unit/engine/orchestrator/test_cleanup_failure_ceremony.py
"""Plugin-cleanup failures must not mask an in-flight run failure.

``cleanup_plugins`` runs in the orchestrator's ``finally`` blocks. Raising a
fresh ``RuntimeError`` there while an exception is already propagating
REPLACES that exception: a ``_RunFailedWithPartialResultError`` carrying real
partial counters degrades to the generic failure ceremony (zeroed counters)
and the caller sees a cleanup error instead of the actual run failure.

Invariants proven here:
- during exception propagation, cleanup failures are recorded (structured
  log) and the original exception continues unchanged;
- with no in-flight exception, cleanup failures still raise ``RuntimeError``
  (the success path must not swallow them);
- end-to-end through ``Orchestrator.run``: a mid-iteration source failure
  followed by a failing ``sink.close()`` surfaces the source's error and the
  failed-ceremony ``RunSummary`` keeps the partial counters.
"""

from __future__ import annotations

import threading
from typing import Any, cast

import pytest
import structlog.testing

from elspeth.contracts import Determinism, SinkProtocol, SourceProtocol
from elspeth.contracts.events import RunSummary
from elspeth.contracts.plugin_context import PluginContext
from elspeth.core.config import SourceSettings
from elspeth.core.dag import ExecutionGraph
from elspeth.core.landscape.database import LandscapeDB
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from elspeth.engine.orchestrator.cleanup import cleanup_plugins
from tests.fixtures.base_classes import as_sink, as_source
from tests.fixtures.plugins import CollectSink, ListSource
from tests.fixtures.stores import MockPayloadStore


class FailingCloseSink(CollectSink):
    """Sink that writes normally but raises on close()."""

    def close(self) -> None:
        raise RuntimeError("sink close failure")


SENSITIVE_CLEANUP_MESSAGE = "cleanup failed for sk-1234567890abcdef1234567890abcdef password=hunter2 https://user:pass@example.test/path"


class SensitiveFailingCloseSink(CollectSink):
    """Sink that raises a secret-bearing cleanup error."""

    def close(self) -> None:
        raise RuntimeError(SENSITIVE_CLEANUP_MESSAGE)


class MidIterationFailingSource(ListSource):
    """Source that yields its rows, then raises mid-iteration."""

    determinism = Determinism.IO_READ

    def load(self, ctx: Any) -> Any:
        yield from self.wrap_rows(self._data)
        raise ValueError("source exploded mid-iteration")


class RecordingEventBus:
    """Event bus that records every emitted event."""

    def __init__(self) -> None:
        self.events: list[Any] = []

    def subscribe(self, event_type: type, handler: Any) -> None:
        pass

    def emit(self, event: Any) -> None:
        self.events.append(event)


def _config_with_failing_close_sink() -> PipelineConfig:
    sink = FailingCloseSink("default")
    return PipelineConfig(
        sources={"primary": as_source(ListSource([], name="src"))},
        transforms=[],
        sinks={"default": as_sink(sink)},
    )


def _config_with_sensitive_failing_close_sink() -> PipelineConfig:
    sink = SensitiveFailingCloseSink("default")
    return PipelineConfig(
        sources={"primary": as_source(ListSource([], name="src"))},
        transforms=[],
        sinks={"default": as_sink(sink)},
    )


class TestCleanupDoesNotMaskPendingException:
    """Direct cleanup_plugins contract: pending exception wins."""

    def test_cleanup_failure_during_exception_propagation_preserves_original(self) -> None:
        config = _config_with_failing_close_sink()
        ctx = PluginContext(run_id="test", config={}, landscape=None)

        with pytest.raises(ValueError, match="primary run failure"):
            try:
                raise ValueError("primary run failure")
            finally:
                cleanup_plugins(config, ctx, include_source=True)

    def test_cleanup_failure_during_exception_propagation_is_logged(self) -> None:
        config = _config_with_failing_close_sink()
        ctx = PluginContext(run_id="test", config={}, landscape=None)

        with structlog.testing.capture_logs() as captured, pytest.raises(ValueError, match="primary run failure"):
            try:
                raise ValueError("primary run failure")
            finally:
                cleanup_plugins(config, ctx, include_source=True)

        events = [entry["event"] for entry in captured]
        assert "Plugin cleanup failed during exception propagation; original error preserved" in events

    def test_cleanup_failure_without_pending_exception_still_raises(self) -> None:
        """The success path must surface cleanup failures, not swallow them."""
        config = _config_with_failing_close_sink()
        ctx = PluginContext(run_id="test", config={}, landscape=None)

        with pytest.raises(RuntimeError, match="Plugin cleanup failed"):
            cleanup_plugins(config, ctx, include_source=True)

    def test_cleanup_failure_public_surfaces_preserve_benign_error_text(self) -> None:
        """Benign cleanup diagnostics keep a bounded message preview for operators."""
        config = _config_with_failing_close_sink()
        ctx = PluginContext(run_id="test", config={}, landscape=None)

        with structlog.testing.capture_logs() as captured, pytest.raises(RuntimeError) as exc_info:
            cleanup_plugins(config, ctx, include_source=True)

        public_error = str(exc_info.value)
        assert "sink.close(default)" in public_error
        assert "RuntimeError" in public_error
        assert "sink close failure" in public_error
        assert "<redacted-plugin-error>" not in public_error

        hook_log = next(entry for entry in captured if entry["event"] == "Plugin cleanup hook failed")
        assert hook_log["error"] == "sink close failure"

    def test_cleanup_failure_public_surfaces_scrub_plugin_error_text(self) -> None:
        """Cleanup diagnostics keep location/type metadata without leaking plugin exception text."""
        config = _config_with_sensitive_failing_close_sink()
        ctx = PluginContext(run_id="test", config={}, landscape=None)

        with structlog.testing.capture_logs() as captured, pytest.raises(RuntimeError) as exc_info:
            cleanup_plugins(config, ctx, include_source=True)

        public_error = str(exc_info.value)
        captured_text = repr(captured)

        for leaked in (
            "sk-1234567890abcdef1234567890abcdef",
            "password=hunter2",
            "https://user:pass@example.test/path",
        ):
            assert leaked not in public_error
            assert leaked not in captured_text

        assert "sink.close(default)" in public_error
        assert "RuntimeError" in public_error
        assert "<redacted-secret>" in public_error

        hook_log = next(entry for entry in captured if entry["event"] == "Plugin cleanup hook failed")
        assert hook_log["hook"] == "sink.close"
        assert hook_log["plugin"] == "default"
        assert hook_log["error_type"] == "RuntimeError"
        assert hook_log["error"] == "<redacted-secret>"
        assert "exc_info" not in hook_log


class TestPartialResultCeremonySurvivesCleanupFailure:
    """End-to-end: run failure + cleanup failure keeps the partial-result ceremony."""

    def _build_pipeline(self) -> tuple[MidIterationFailingSource, FailingCloseSink, PipelineConfig, ExecutionGraph]:
        source = MidIterationFailingSource([{"value": 1}], name="mid_fail_source", on_success="default")
        sink = FailingCloseSink("default")

        source_settings = SourceSettings(plugin=source.name, on_success="default", options={})
        graph = ExecutionGraph.from_plugin_instances(
            sources={"primary": cast(SourceProtocol, source)},
            source_settings_map={"primary": source_settings},
            transforms=[],
            sinks=cast("dict[str, SinkProtocol]", {"default": sink}),
            aggregations={},
            gates=[],
        )
        config = PipelineConfig(
            sources={"primary": as_source(source)},
            transforms=[],
            sinks={"default": as_sink(sink)},
        )
        return source, sink, config, graph

    def test_original_error_and_partial_counters_survive_failing_cleanup(self) -> None:
        db = LandscapeDB.in_memory()
        event_bus = RecordingEventBus()
        _source, _sink, config, graph = self._build_pipeline()

        orchestrator = Orchestrator(db, event_bus=event_bus)

        # The source's ValueError must propagate — NOT the RuntimeError from
        # the sink's failing close() in the cleanup finally block.
        with pytest.raises(ValueError, match="source exploded mid-iteration"):
            orchestrator.run(
                config,
                graph=graph,
                payload_store=MockPayloadStore(),
                shutdown_event=threading.Event(),
            )

        summaries = [event for event in event_bus.events if isinstance(event, RunSummary)]
        assert len(summaries) == 1
        summary = summaries[0]
        assert summary.status.value == "failed"
        # Partial-result ceremony: the row ingested before the failure is
        # reported, not the generic ceremony's zeroed counters.
        assert summary.total_rows == 1
