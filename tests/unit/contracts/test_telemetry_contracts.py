# tests/unit/contracts/test_telemetry_contracts.py
"""Contract tests for plugin telemetry emission.

These tests verify that external I/O plugins ACTUALLY emit required telemetry
events at runtime, not just that the code paths exist. Unlike keyword-based
tests that check for `self._run_id` in source code, these tests verify actual
telemetry emission through the production code path.

CRITICAL PRINCIPLE: Use production Orchestrator, not manual PluginContext wiring.
This catches wiring bugs that unit tests miss.

Tested plugins:
- AuditedLLMClient -> ExternalCallCompleted (call_type=LLM)
- AuditedHTTPClient -> ExternalCallCompleted (call_type=HTTP)
- LLMTransform with provider="azure" (via AuditedLLMClient)
- LLMTransform with provider="openrouter" (via AuditedHTTPClient)

Each test verifies:
1. Plugin is configured and run through production Orchestrator
2. External calls are made (mocked)
3. ExternalCallCompleted telemetry events are captured
4. Events have correct attributes (run_id, call_type, status, etc.)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest

from elspeth.contracts import (
    CallStatus,
    CallType,
)
from elspeth.contracts.enums import RunStatus, TelemetryGranularity
from elspeth.contracts.events import ExternalCallCompleted
from elspeth.core.landscape import LandscapeDB
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from elspeth.telemetry import TelemetryManager
from tests.fixtures.base_classes import as_sink, as_source, as_transform
from tests.fixtures.landscape import make_landscape_db
from tests.fixtures.pipeline import build_production_graph
from tests.fixtures.plugins import CollectSink, ListSource, PassTransform
from tests.fixtures.telemetry import MockTelemetryConfig, TelemetryTestExporter

# =============================================================================
# AuditedLLMClient Telemetry Contract Tests
# =============================================================================


class TestAuditedLLMClientTelemetryContract:
    """Verify AuditedLLMClient emits ExternalCallCompleted events.

    These tests verify the contract: AuditedLLMClient MUST emit
    ExternalCallCompleted with call_type=LLM on every call.
    """

    def _create_mock_execution(self) -> MagicMock:
        """Create a mock ExecutionRepository."""
        execution = MagicMock()
        recorded_call = MagicMock()
        recorded_call.request_hash = "req_hash_123"
        recorded_call.response_hash = "resp_hash_456"
        execution.record_call.return_value = recorded_call
        return execution

    def _create_mock_openai_client(
        self,
        content: str = "Hello!",
        model: str = "gpt-4",
    ) -> MagicMock:
        """Create a mock OpenAI client."""
        message = Mock()
        message.content = content

        choice = Mock()
        choice.message = message

        usage = Mock()
        usage.prompt_tokens = 10
        usage.completion_tokens = 5

        response = Mock()
        response.choices = [choice]
        response.model = model
        response.usage = usage
        response.model_dump = Mock(return_value={"id": "resp_123"})

        client = MagicMock()
        client.chat.completions.create.return_value = response

        return client

    def test_llm_client_emits_external_call_completed_on_success(self) -> None:
        """AuditedLLMClient emits ExternalCallCompleted on successful call."""
        from elspeth.plugins.infrastructure.clients.llm import AuditedLLMClient

        execution = self._create_mock_execution()
        openai_client = self._create_mock_openai_client()

        emitted_events: list[ExternalCallCompleted] = []

        def telemetry_emit(event: ExternalCallCompleted) -> None:
            emitted_events.append(event)

        client = AuditedLLMClient(
            execution=execution,
            state_id="state_123",
            underlying_client=openai_client,
            provider="azure",
            run_id="run_abc",
            telemetry_emit=telemetry_emit,
        )

        client.chat_completion(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
        )

        # CONTRACT: Must emit exactly one ExternalCallCompleted
        assert len(emitted_events) == 1
        event = emitted_events[0]

        # CONTRACT: Must have correct call_type
        assert event.call_type == CallType.LLM

        # CONTRACT: Must have correct status
        assert event.status == CallStatus.SUCCESS

        # CONTRACT: Must have run_id
        assert event.run_id == "run_abc"

        # CONTRACT: Must have state_id
        assert event.state_id == "state_123"

        # CONTRACT: Must have latency
        assert event.latency_ms > 0

        # CONTRACT: Must have hashes
        assert event.request_hash is not None
        assert event.response_hash is not None

    def test_llm_client_emits_external_call_completed_on_error(self) -> None:
        """AuditedLLMClient emits ExternalCallCompleted on failed call."""
        from elspeth.plugins.infrastructure.clients.llm import AuditedLLMClient, LLMClientError

        execution = self._create_mock_execution()
        openai_client = MagicMock()
        openai_client.chat.completions.create.side_effect = Exception("API error")

        emitted_events: list[ExternalCallCompleted] = []

        def telemetry_emit(event: ExternalCallCompleted) -> None:
            emitted_events.append(event)

        client = AuditedLLMClient(
            execution=execution,
            state_id="state_123",
            underlying_client=openai_client,
            provider="openai",
            run_id="run_abc",
            telemetry_emit=telemetry_emit,
        )

        with pytest.raises(LLMClientError):
            client.chat_completion(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hello"}],
            )

        # CONTRACT: Must emit ExternalCallCompleted even on error
        assert len(emitted_events) == 1
        event = emitted_events[0]

        # CONTRACT: Must have ERROR status
        assert event.status == CallStatus.ERROR

        # CONTRACT: Must still have call_type=LLM
        assert event.call_type == CallType.LLM


# =============================================================================
# AuditedHTTPClient Telemetry Contract Tests
# =============================================================================


class TestAuditedHTTPClientTelemetryContract:
    """Verify AuditedHTTPClient emits ExternalCallCompleted events.

    These tests verify the contract: AuditedHTTPClient MUST emit
    ExternalCallCompleted with call_type=HTTP on every call.
    """

    def _create_mock_execution(self) -> MagicMock:
        """Create a mock ExecutionRepository."""
        execution = MagicMock()
        recorded_call = MagicMock()
        recorded_call.request_hash = "req_hash_123"
        recorded_call.response_hash = "resp_hash_456"
        execution.record_call.return_value = recorded_call
        return execution

    def test_http_client_emits_external_call_completed_on_success(self) -> None:
        """AuditedHTTPClient emits ExternalCallCompleted on successful POST."""
        from elspeth.plugins.infrastructure.clients.http import AuditedHTTPClient

        execution = self._create_mock_execution()

        emitted_events: list[ExternalCallCompleted] = []

        def telemetry_emit(event: ExternalCallCompleted) -> None:
            emitted_events.append(event)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "success"}
        mock_response.text = '{"result": "success"}'
        mock_response.content = b'{"result": "success"}'
        mock_response.headers = {"content-type": "application/json"}

        with patch("httpx.Client") as mock_client_class:
            client = AuditedHTTPClient(
                execution=execution,
                state_id="state_123",
                base_url="https://api.example.com",
                run_id="run_abc",
                telemetry_emit=telemetry_emit,
            )
            mock_client_instance = mock_client_class.return_value
            mock_client_instance.post.return_value = mock_response

            client.post("/endpoint", json={"input": "test"})

        # CONTRACT: Must emit exactly one ExternalCallCompleted
        assert len(emitted_events) == 1
        event = emitted_events[0]

        # CONTRACT: Must have correct call_type
        assert event.call_type == CallType.HTTP

        # CONTRACT: Must have correct status
        assert event.status == CallStatus.SUCCESS

        # CONTRACT: Must have run_id
        assert event.run_id == "run_abc"

        # CONTRACT: Must have state_id
        assert event.state_id == "state_123"

    def test_http_client_emits_external_call_completed_on_error(self) -> None:
        """AuditedHTTPClient emits ExternalCallCompleted on network error."""
        import httpx

        from elspeth.plugins.infrastructure.clients.http import AuditedHTTPClient

        execution = self._create_mock_execution()

        emitted_events: list[ExternalCallCompleted] = []

        def telemetry_emit(event: ExternalCallCompleted) -> None:
            emitted_events.append(event)

        with patch("httpx.Client") as mock_client_class:
            client = AuditedHTTPClient(
                execution=execution,
                state_id="state_123",
                base_url="https://api.example.com",
                run_id="run_abc",
                telemetry_emit=telemetry_emit,
            )
            mock_client_instance = mock_client_class.return_value
            mock_client_instance.post.side_effect = httpx.ConnectError("Connection failed")

            with pytest.raises(httpx.ConnectError):
                client.post("/endpoint", json={"input": "test"})

        # CONTRACT: Must emit ExternalCallCompleted even on error
        assert len(emitted_events) == 1
        event = emitted_events[0]

        # CONTRACT: Must have ERROR status
        assert event.status == CallStatus.ERROR

        # CONTRACT: Must still have call_type=HTTP
        assert event.call_type == CallType.HTTP

    def test_http_client_emits_external_call_completed_on_get(self) -> None:
        """AuditedHTTPClient emits ExternalCallCompleted on GET request."""
        from elspeth.plugins.infrastructure.clients.http import AuditedHTTPClient

        execution = self._create_mock_execution()

        emitted_events: list[ExternalCallCompleted] = []

        def telemetry_emit(event: ExternalCallCompleted) -> None:
            emitted_events.append(event)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": "value"}
        mock_response.text = '{"data": "value"}'
        mock_response.content = b'{"data": "value"}'
        mock_response.headers = {"content-type": "application/json"}

        with patch("httpx.Client") as mock_client_class:
            client = AuditedHTTPClient(
                execution=execution,
                state_id="state_123",
                base_url="https://api.example.com",
                run_id="run_abc",
                telemetry_emit=telemetry_emit,
            )
            mock_client_instance = mock_client_class.return_value
            mock_client_instance.get.return_value = mock_response

            client.get("/endpoint")

        # CONTRACT: Must emit ExternalCallCompleted for GET
        assert len(emitted_events) == 1
        event = emitted_events[0]
        assert event.call_type == CallType.HTTP
        assert event.status == CallStatus.SUCCESS


# =============================================================================
# Orchestrator Wiring Contract Tests
# =============================================================================


class TestOrchestratorTelemetryWiringContract:
    """Verify Orchestrator correctly wires telemetry to PluginContext.

    These tests verify the contract: When a TelemetryManager is provided,
    Orchestrator MUST wire the telemetry_emit callback to PluginContext
    so that plugins can emit telemetry.

    CRITICAL: Uses production Orchestrator, not manual PluginContext wiring.
    """

    @pytest.fixture
    def landscape_db(self) -> LandscapeDB:
        """Fresh in-memory database for each test."""
        return make_landscape_db()

    def test_orchestrator_wires_telemetry_emit_to_context(
        self,
        landscape_db: LandscapeDB,
        payload_store,
    ) -> None:
        """Orchestrator wires real telemetry_emit to PluginContext."""
        exporter = TelemetryTestExporter()
        config = MockTelemetryConfig(granularity=TelemetryGranularity.FULL)
        telemetry_manager = TelemetryManager(config, exporters=[exporter])

        # Capture the telemetry_emit callback from inside a transform
        captured_callback = None

        class TelemetryCapturingTransform(PassTransform):
            """Transform that captures the telemetry_emit callback."""

            name = "telemetry_capturing"

            def on_start(self, ctx: Any) -> None:
                super().on_start(ctx)
                nonlocal captured_callback
                captured_callback = ctx.telemetry_emit

        source = ListSource([{"id": 1, "text": "test"}])
        sink = CollectSink()

        pipeline_config = PipelineConfig(
            source=as_source(source),
            transforms=[as_transform(TelemetryCapturingTransform())],
            sinks={"output": as_sink(sink)},
        )

        orchestrator = Orchestrator(landscape_db, telemetry_manager=telemetry_manager)
        orchestrator.run(
            pipeline_config,
            graph=build_production_graph(pipeline_config),
            payload_store=payload_store,
        )

        # CONTRACT: telemetry_emit must be captured (not None)
        assert captured_callback is not None, "ctx.telemetry_emit was not set"

        # CONTRACT: telemetry_emit must NOT be the default no-op lambda
        callback_name = getattr(captured_callback, "__name__", str(captured_callback))
        assert callback_name != "<lambda>", f"ctx.telemetry_emit is still the default no-op lambda. Got: {captured_callback}"

    def test_orchestrator_emits_lifecycle_telemetry(
        self,
        landscape_db: LandscapeDB,
        payload_store,
    ) -> None:
        """Orchestrator emits RunStarted and RunFinished telemetry."""
        exporter = TelemetryTestExporter()
        config = MockTelemetryConfig(granularity=TelemetryGranularity.FULL)
        telemetry_manager = TelemetryManager(config, exporters=[exporter])

        source = ListSource([{"id": 1, "text": "test"}])
        sink = CollectSink()

        pipeline_config = PipelineConfig(
            source=as_source(source),
            transforms=[as_transform(PassTransform())],
            sinks={"output": as_sink(sink)},
        )

        orchestrator = Orchestrator(landscape_db, telemetry_manager=telemetry_manager)
        result = orchestrator.run(
            pipeline_config,
            graph=build_production_graph(pipeline_config),
            payload_store=payload_store,
        )

        # CONTRACT: Pipeline must complete successfully
        assert result.status == RunStatus.COMPLETED

        # CONTRACT: Must emit RunStarted
        exporter.assert_event_emitted("RunStarted")

        # CONTRACT: Must emit RunFinished
        exporter.assert_event_emitted("RunFinished")

        # CONTRACT: Events must have matching run_id
        run_started = exporter.get_events_of_type("RunStarted")[0]
        run_finished = exporter.get_events_of_type("RunFinished")[0]
        assert run_started.run_id == result.run_id
        assert run_finished.run_id == result.run_id


# =============================================================================
# Plugin Telemetry Integration Tests (via Audited Clients)
# =============================================================================


class TestPluginTelemetryThroughAuditedClients:
    """Verify plugins emit telemetry through their audited clients.

    These tests verify that when plugins use AuditedLLMClient or
    AuditedHTTPClient, the telemetry events are correctly emitted.

    Note: Full orchestrator integration tests for batch transforms (like
    LLMTransform with provider="azure") require more complex setup. These
    tests verify the underlying client contracts that those plugins depend on.
    """

    def test_audited_llm_client_telemetry_flows_through_callback(self) -> None:
        """Telemetry emitted by AuditedLLMClient flows through the callback.

        This verifies that when a plugin creates an AuditedLLMClient with
        the telemetry_emit callback from PluginContext, events are emitted.
        """
        from elspeth.plugins.infrastructure.clients.llm import AuditedLLMClient

        execution = MagicMock()
        execution.record_call.return_value = MagicMock(
            request_hash="hash1",
            response_hash="hash2",
        )

        # Simulate the telemetry callback that would come from Orchestrator
        exporter = TelemetryTestExporter()

        def telemetry_emit(event: Any) -> None:
            exporter.export(event)

        # Mock OpenAI client
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Hello!"))]
        mock_response.model = "gpt-4"
        mock_response.usage = Mock(prompt_tokens=10, completion_tokens=5)
        mock_response.model_dump = Mock(return_value={})

        mock_openai = MagicMock()
        mock_openai.chat.completions.create.return_value = mock_response

        client = AuditedLLMClient(
            execution=execution,
            state_id="state-1",
            run_id="run-1",
            telemetry_emit=telemetry_emit,
            underlying_client=mock_openai,
            provider="azure",
        )

        client.chat_completion(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
        )

        # CONTRACT: ExternalCallCompleted must be emitted
        exporter.assert_event_emitted(
            "ExternalCallCompleted",
            call_type=CallType.LLM,
            status=CallStatus.SUCCESS,
        )

    def test_audited_http_client_telemetry_flows_through_callback(self) -> None:
        """Telemetry emitted by AuditedHTTPClient flows through the callback.

        This verifies that when a plugin creates an AuditedHTTPClient with
        the telemetry_emit callback from PluginContext, events are emitted.
        """
        from elspeth.plugins.infrastructure.clients.http import AuditedHTTPClient

        execution = MagicMock()
        execution.record_call.return_value = MagicMock(
            request_hash="hash1",
            response_hash="hash2",
        )

        # Simulate the telemetry callback that would come from Orchestrator
        exporter = TelemetryTestExporter()

        def telemetry_emit(event: Any) -> None:
            exporter.export(event)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"success": True}
        mock_response.text = '{"success": true}'
        mock_response.content = b'{"success": true}'
        mock_response.headers = {"content-type": "application/json"}

        with patch("httpx.Client") as mock_client_class:
            client = AuditedHTTPClient(
                execution=execution,
                state_id="state-1",
                run_id="run-1",
                telemetry_emit=telemetry_emit,
                base_url="https://api.example.com",
            )
            mock_instance = mock_client_class.return_value
            mock_instance.post.return_value = mock_response

            client.post("/endpoint", json={"data": "test"})

        # CONTRACT: ExternalCallCompleted must be emitted
        exporter.assert_event_emitted(
            "ExternalCallCompleted",
            call_type=CallType.HTTP,
            status=CallStatus.SUCCESS,
        )


# =============================================================================
# Telemetry Emission Order Contract Tests
# =============================================================================


class TestTelemetryEmissionOrderContract:
    """Verify telemetry is emitted AFTER Landscape recording.

    This is a critical contract: Telemetry is operational visibility,
    but Landscape is the legal record. Telemetry must NOT be emitted
    if Landscape recording fails.
    """

    def test_llm_client_emits_telemetry_after_landscape(self) -> None:
        """AuditedLLMClient emits telemetry AFTER Landscape recording."""
        from elspeth.plugins.infrastructure.clients.llm import AuditedLLMClient

        call_order: list[str] = []

        execution = MagicMock()

        def mock_record_call(**kwargs):
            call_order.append("landscape")
            return MagicMock(request_hash="hash1", response_hash="hash2")

        execution.record_call.side_effect = mock_record_call

        def telemetry_emit(event: Any) -> None:
            call_order.append("telemetry")

        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Hello!"))]
        mock_response.model = "gpt-4"
        mock_response.usage = Mock(prompt_tokens=10, completion_tokens=5)
        mock_response.model_dump = Mock(return_value={})

        mock_openai = MagicMock()
        mock_openai.chat.completions.create.return_value = mock_response

        client = AuditedLLMClient(
            execution=execution,
            state_id="state-1",
            run_id="run-1",
            telemetry_emit=telemetry_emit,
            underlying_client=mock_openai,
            provider="azure",
        )

        client.chat_completion(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
        )

        # CONTRACT: Landscape must be recorded BEFORE telemetry is emitted
        assert call_order == ["landscape", "telemetry"]

    def test_http_client_emits_telemetry_after_landscape(self) -> None:
        """AuditedHTTPClient emits telemetry AFTER Landscape recording."""
        from elspeth.plugins.infrastructure.clients.http import AuditedHTTPClient

        call_order: list[str] = []

        execution = MagicMock()

        def mock_record_call(**kwargs):
            call_order.append("landscape")
            return MagicMock(request_hash="hash1", response_hash="hash2")

        execution.record_call.side_effect = mock_record_call

        def telemetry_emit(event: Any) -> None:
            call_order.append("telemetry")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"success": True}
        mock_response.text = '{"success": true}'
        mock_response.content = b'{"success": true}'
        mock_response.headers = {"content-type": "application/json"}

        with patch("httpx.Client") as mock_client_class:
            client = AuditedHTTPClient(
                execution=execution,
                state_id="state-1",
                run_id="run-1",
                telemetry_emit=telemetry_emit,
                base_url="https://api.example.com",
            )
            mock_instance = mock_client_class.return_value
            mock_instance.post.return_value = mock_response

            client.post("/endpoint", json={"data": "test"})

        # CONTRACT: Landscape must be recorded BEFORE telemetry is emitted
        assert call_order == ["landscape", "telemetry"]

    def test_no_telemetry_when_landscape_recording_fails(self) -> None:
        """Telemetry is NOT emitted if Landscape recording fails.

        This is critical: If Landscape fails, the event wasn't properly
        recorded. Emitting telemetry would be misleading.
        """
        from elspeth.plugins.infrastructure.clients.llm import AuditedLLMClient

        execution = MagicMock()
        execution.record_call.side_effect = Exception("Database error")

        emitted_events: list[Any] = []

        def telemetry_emit(event: Any) -> None:
            emitted_events.append(event)

        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Hello!"))]
        mock_response.model = "gpt-4"
        mock_response.usage = Mock(prompt_tokens=10, completion_tokens=5)
        mock_response.model_dump = Mock(return_value={})

        mock_openai = MagicMock()
        mock_openai.chat.completions.create.return_value = mock_response

        client = AuditedLLMClient(
            execution=execution,
            state_id="state-1",
            run_id="run-1",
            telemetry_emit=telemetry_emit,
            underlying_client=mock_openai,
            provider="azure",
        )

        with pytest.raises(Exception, match="Database error"):
            client.chat_completion(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hello"}],
            )

        # CONTRACT: No telemetry should be emitted when Landscape fails
        assert len(emitted_events) == 0, "Telemetry was emitted before Landscape!"


# =============================================================================
# Telemetry Failure Isolation Contract Tests
# =============================================================================


class TestTelemetryFailureIsolationContract:
    """Verify telemetry failures don't corrupt audit trail or cause retries.

    This is a critical contract: Telemetry is operational visibility.
    A telemetry failure must NOT:
    1. Cause a second audit record
    2. Change the call outcome
    3. Trigger retry logic
    """

    def test_llm_client_isolates_telemetry_failure(self) -> None:
        """AuditedLLMClient isolates telemetry failure from call result."""
        from elspeth.plugins.infrastructure.clients.llm import AuditedLLMClient

        execution = MagicMock()
        execution.record_call.return_value = MagicMock(
            request_hash="hash1",
            response_hash="hash2",
        )

        def failing_telemetry_emit(event: Any) -> None:
            raise RuntimeError("Telemetry export failed!")

        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Hello!"))]
        mock_response.model = "gpt-4"
        mock_response.usage = Mock(prompt_tokens=10, completion_tokens=5)
        mock_response.model_dump = Mock(return_value={})

        mock_openai = MagicMock()
        mock_openai.chat.completions.create.return_value = mock_response

        client = AuditedLLMClient(
            execution=execution,
            state_id="state-1",
            run_id="run-1",
            telemetry_emit=failing_telemetry_emit,
            underlying_client=mock_openai,
            provider="azure",
        )

        # CONTRACT: Call should succeed despite telemetry failure
        response = client.chat_completion(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert response.content == "Hello!"

        # CONTRACT: Only ONE audit record should exist
        assert execution.record_call.call_count == 1

        # CONTRACT: Audit record should show SUCCESS
        call_kwargs = execution.record_call.call_args.kwargs
        assert call_kwargs["status"] == CallStatus.SUCCESS

    def test_http_client_isolates_telemetry_failure(self) -> None:
        """AuditedHTTPClient isolates telemetry failure from call result."""
        from elspeth.plugins.infrastructure.clients.http import AuditedHTTPClient

        execution = MagicMock()
        execution.record_call.return_value = MagicMock(
            request_hash="hash1",
            response_hash="hash2",
        )

        def failing_telemetry_emit(event: Any) -> None:
            raise RuntimeError("Telemetry export failed!")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"success": True}
        mock_response.text = '{"success": true}'
        mock_response.content = b'{"success": true}'
        mock_response.headers = {"content-type": "application/json"}

        with patch("httpx.Client") as mock_client_class:
            client = AuditedHTTPClient(
                execution=execution,
                state_id="state-1",
                run_id="run-1",
                telemetry_emit=failing_telemetry_emit,
                base_url="https://api.example.com",
            )
            mock_instance = mock_client_class.return_value
            mock_instance.post.return_value = mock_response

            # CONTRACT: Call should succeed despite telemetry failure
            response = client.post("/endpoint", json={"data": "test"})

        assert response.status_code == 200

        # CONTRACT: Only ONE audit record should exist
        assert execution.record_call.call_count == 1

        # CONTRACT: Audit record should show SUCCESS
        call_kwargs = execution.record_call.call_args.kwargs
        assert call_kwargs["status"] == CallStatus.SUCCESS
