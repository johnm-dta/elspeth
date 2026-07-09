# tests/unit/engine/orchestrator/test_phase_error_masking.py
"""Tests that PhaseError emission failures don't mask original exceptions.

Bug: ~5 sites in orchestrator/core.py emit PhaseError events between
`except` and `raise`. If the PhaseError handler throws, the original
exception is replaced by the handler's exception.

Issue: elspeth-614ba26b06
"""

from __future__ import annotations

import json
import threading
from typing import Any, cast

import pytest

from elspeth.cli_formatters import create_json_formatters
from elspeth.contracts import Determinism, SinkProtocol, SourceProtocol
from elspeth.contracts.events import PhaseError, PipelinePhase
from elspeth.core.config import SourceSettings
from elspeth.core.dag import ExecutionGraph
from elspeth.core.landscape.database import LandscapeDB
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from elspeth.engine.orchestrator.ceremony import RunCeremony
from tests.fixtures.base_classes import as_sink, as_source
from tests.fixtures.plugins import CollectSink, ListSource
from tests.fixtures.stores import MockPayloadStore

# ---------------------------------------------------------------------------
# Event bus that throws on PhaseError events
# ---------------------------------------------------------------------------


class MaskingEventBus:
    """Event bus that raises when a PhaseError is emitted.

    Simulates a handler bug in a formatter or observer subscribed to
    PhaseError events. All other events pass through silently.
    """

    def __init__(self) -> None:
        self.events: list[Any] = []

    def subscribe(self, event_type: type, handler: Any) -> None:
        pass

    def emit(self, event: Any) -> None:
        self.events.append(event)
        if isinstance(event, PhaseError):
            raise RuntimeError("handler bug during PhaseError emission")


class CapturingEventBus:
    """Event bus that records emitted events."""

    def __init__(self) -> None:
        self.events: list[Any] = []

    def subscribe(self, event_type: type, handler: Any) -> None:
        pass

    def emit(self, event: Any) -> None:
        self.events.append(event)


# ---------------------------------------------------------------------------
# Failing source
# ---------------------------------------------------------------------------


class FailingSource(ListSource):
    """Source that throws on load()."""

    determinism = Determinism.IO_READ

    def __init__(self) -> None:
        super().__init__([], name="failing_source", on_success="default")

    def load(self, ctx: Any) -> Any:
        raise ValueError("original source error — must survive masking")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPhaseErrorDoesNotMaskOriginalException:
    """Verify that a PhaseError handler failure preserves the original exception."""

    def _build_pipeline_with_failing_source(self) -> tuple[FailingSource, CollectSink, ExecutionGraph]:
        """Build a minimal source → sink pipeline with a source that fails on load()."""
        source = FailingSource()
        sink = CollectSink("default")

        source_settings = SourceSettings(
            plugin=source.name,
            on_success="default",
            options={},
        )

        graph = ExecutionGraph.from_plugin_instances(
            sources={"primary": cast(SourceProtocol, source)},
            source_settings_map={"primary": source_settings},
            transforms=[],
            sinks=cast("dict[str, SinkProtocol]", {"default": sink}),
            aggregations={},
            gates=[],
        )
        return source, sink, graph

    def test_source_phase_error_preserves_original_exception(self) -> None:
        """SOURCE phase: if emit(PhaseError) fails, source's ValueError must propagate."""
        db = LandscapeDB.in_memory()
        event_bus = MaskingEventBus()
        payload_store = MockPayloadStore()

        source, sink, graph = self._build_pipeline_with_failing_source()

        config = PipelineConfig(
            sources={"primary": as_source(source)},
            transforms=[],
            sinks={"default": as_sink(sink)},
        )

        orchestrator = Orchestrator(db, event_bus=event_bus)

        # The original ValueError must propagate, NOT the RuntimeError
        # from the PhaseError handler
        with pytest.raises(ValueError, match="original source error"):
            orchestrator.run(
                config,
                graph=graph,
                payload_store=payload_store,
                shutdown_event=threading.Event(),
            )


class TestPhaseErrorPublicPayload:
    """Verify that public PhaseError events do not leak raw exception text."""

    def test_phase_error_redacts_secret_exception_text_before_event_and_json_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        event_bus = CapturingEventBus()
        ceremony = RunCeremony(events=event_bus, telemetry=None)
        api_key = "sk-" + ("a" * 24)
        secret_url = "https://user:" + "passw0rd" + "@example.test/resource"
        sql_fragment = "password=" + "hunter2"
        error = RuntimeError(f"provider failed for {secret_url}; SQL {sql_fragment}; tool output {api_key}")

        ceremony.emit_phase_error(PipelinePhase.PROCESS, error, target="processor")

        event = event_bus.events[-1]
        assert isinstance(event, PhaseError)
        assert not hasattr(event, "error")
        assert event.error_type == "RuntimeError"
        assert event.error_message == "<redacted-secret>"
        assert secret_url not in event.error_message
        assert sql_fragment not in event.error_message
        assert api_key not in event.error_message

        create_json_formatters()[PhaseError](event)
        payload = json.loads(capsys.readouterr().err)
        assert payload["error_type"] == "RuntimeError"
        assert payload["error"] == "<redacted-secret>"
        assert secret_url not in json.dumps(payload)
        assert sql_fragment not in json.dumps(payload)
        assert api_key not in json.dumps(payload)

    def test_phase_error_preserves_non_sensitive_error_type_and_message(self) -> None:
        event_bus = CapturingEventBus()
        ceremony = RunCeremony(events=event_bus, telemetry=None)

        ceremony.emit_phase_error(PipelinePhase.SOURCE, ValueError("source row count mismatch"), target="source_a")

        event = event_bus.events[-1]
        assert isinstance(event, PhaseError)
        assert event.error_type == "ValueError"
        assert event.error_message == "source row count mismatch"
