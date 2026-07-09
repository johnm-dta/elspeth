# tests/telemetry/exporters/test_datadog.py
"""Tests for Datadog telemetry exporter.

Tests cover:
- Configuration validation (port range, unsupported api_key)
- Span creation for telemetry events
- Tag serialization (datetime, enum, dict, tuple handling)
- Flush and close lifecycle
- Error handling (export failures don't crash pipeline)

Note: These tests mock ddtrace since it may not be installed in the test environment.
"""

import sys
from datetime import UTC, datetime
from types import ModuleType
from typing import Any
from unittest.mock import patch

import pytest

from elspeth.contracts import GateEvaluated, TokenCompleted, TokenUsage
from elspeth.contracts.enums import RoutingMode, RunStatus, TerminalOutcome, TerminalPath
from elspeth.contracts.events import (
    RunFinished,
    RunStarted,
)
from elspeth.telemetry.errors import TelemetryExporterError
from elspeth.telemetry.exporters.datadog import DatadogExporter

# All required fields for a valid Datadog exporter config
_VALID_CONFIG: dict[str, object] = {
    "service_name": "test-service",
    "env": "test",
    "agent_host": "localhost",
    "agent_port": 8126,
}


class _CallArgs:
    def __init__(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        self.args = args
        self.kwargs = kwargs

    def __getitem__(self, index: int) -> tuple[Any, ...] | dict[str, Any]:
        if index == 0:
            return self.args
        if index == 1:
            return self.kwargs
        raise IndexError(index)


class _CallRecorder:
    def __init__(self, *, return_value: Any = None, side_effect: Any = None) -> None:
        self.return_value = return_value
        self.side_effect = side_effect
        self.call_args: _CallArgs | None = None
        self.call_args_list: list[_CallArgs] = []

    @property
    def call_count(self) -> int:
        return len(self.call_args_list)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.call_args = _CallArgs(args, kwargs)
        self.call_args_list.append(self.call_args)
        if self.side_effect is not None:
            if isinstance(self.side_effect, BaseException):
                raise self.side_effect
            if isinstance(self.side_effect, type) and issubclass(self.side_effect, BaseException):
                raise self.side_effect()
            if callable(self.side_effect):
                return self.side_effect(*args, **kwargs)
        return self.return_value

    def assert_called_once(self) -> None:
        assert self.call_count == 1

    def assert_any_call(self, *args: Any, **kwargs: Any) -> None:
        assert any(call.args == args and call.kwargs == kwargs for call in self.call_args_list)


class _FakeDatadogSpan:
    def __init__(self) -> None:
        self.start_ns = 0
        self.set_tag = _CallRecorder()
        self.finish = _CallRecorder()


class _FakeDatadogTracer:
    def __init__(self, span: _FakeDatadogSpan) -> None:
        self.start_span = _CallRecorder(return_value=span)
        self.flush = _CallRecorder()
        self.shutdown = _CallRecorder()


class _FakeDatadogModule(ModuleType):
    def __init__(self, tracer: _FakeDatadogTracer) -> None:
        super().__init__("ddtrace")
        self.tracer = tracer


def create_mock_ddtrace_module() -> tuple[_FakeDatadogModule, _FakeDatadogTracer, _FakeDatadogSpan]:
    """Create a fake ddtrace module with tracer.

    Returns:
        Tuple of (module, tracer, span)
    """
    span = _FakeDatadogSpan()
    tracer = _FakeDatadogTracer(span)
    module = _FakeDatadogModule(tracer)

    return module, tracer, span


class TestDatadogExporterConfiguration:
    """Tests for DatadogExporter configuration."""

    def test_name_property(self) -> None:
        """Exporter name is 'datadog'."""
        exporter = DatadogExporter()
        assert exporter.name == "datadog"

    def test_empty_config_rejects_missing_required_fields(self) -> None:
        """Empty config crashes — all fields are required."""
        exporter = DatadogExporter()
        with pytest.raises(TelemetryExporterError, match="service_name"):
            exporter.configure({})

    def test_valid_configuration_no_env_mutation(self) -> None:
        """Valid configuration does not mutate global env vars."""
        mock_module, _mock_tracer, _ = create_mock_ddtrace_module()

        with patch.dict(sys.modules, {"ddtrace": mock_module}), patch.dict("os.environ", {}, clear=False):
            exporter = DatadogExporter()
            exporter.configure(dict(_VALID_CONFIG))

            # Env vars are set transiently for ddtrace init, then restored
            import os

            assert os.environ.get("DD_AGENT_HOST") is None
            assert os.environ.get("DD_TRACE_AGENT_PORT") is None

    def test_custom_agent_host_and_port(self) -> None:
        """Custom agent host and port are set via environment variables."""
        mock_module, _mock_tracer, _ = create_mock_ddtrace_module()

        with patch.dict(sys.modules, {"ddtrace": mock_module}), patch.dict("os.environ", {}, clear=False):
            exporter = DatadogExporter()
            exporter.configure(
                {
                    **_VALID_CONFIG,
                    "agent_host": "datadog-agent.internal",
                    "agent_port": 9126,
                }
            )

            # Env vars are restored after ddtrace init — no global mutation
            import os

            assert os.environ.get("DD_AGENT_HOST") is None
            assert os.environ.get("DD_TRACE_AGENT_PORT") is None

    def test_pre_existing_env_vars_are_restored(self) -> None:
        """Pre-existing DD_AGENT_HOST/DD_TRACE_AGENT_PORT are restored after configure."""
        mock_module, _mock_tracer, _ = create_mock_ddtrace_module()

        pre_existing = {"DD_AGENT_HOST": "original-host", "DD_TRACE_AGENT_PORT": "9999"}
        with patch.dict(sys.modules, {"ddtrace": mock_module}), patch.dict("os.environ", pre_existing, clear=False):
            exporter = DatadogExporter()
            exporter.configure({**_VALID_CONFIG, "agent_host": "new-host", "agent_port": 1234})

            import os

            assert os.environ.get("DD_AGENT_HOST") == "original-host"
            assert os.environ.get("DD_TRACE_AGENT_PORT") == "9999"

    def test_missing_service_name_raises(self) -> None:
        """Missing service_name raises TelemetryExporterError."""
        exporter = DatadogExporter()
        config = {k: v for k, v in _VALID_CONFIG.items() if k != "service_name"}
        with pytest.raises(TelemetryExporterError, match="service_name"):
            exporter.configure(config)

    def test_missing_env_raises(self) -> None:
        """Missing env raises TelemetryExporterError."""
        exporter = DatadogExporter()
        config = {k: v for k, v in _VALID_CONFIG.items() if k != "env"}
        with pytest.raises(TelemetryExporterError, match="env"):
            exporter.configure(config)

    def test_missing_agent_host_raises(self) -> None:
        """Missing agent_host raises TelemetryExporterError."""
        exporter = DatadogExporter()
        config = {k: v for k, v in _VALID_CONFIG.items() if k != "agent_host"}
        with pytest.raises(TelemetryExporterError, match="agent_host"):
            exporter.configure(config)

    def test_missing_agent_port_raises(self) -> None:
        """Missing agent_port raises TelemetryExporterError."""
        exporter = DatadogExporter()
        config = {k: v for k, v in _VALID_CONFIG.items() if k != "agent_port"}
        with pytest.raises(TelemetryExporterError, match="agent_port"):
            exporter.configure(config)

    def test_invalid_port_zero_raises(self) -> None:
        """Port 0 raises TelemetryExporterError."""
        exporter = DatadogExporter()
        with pytest.raises(TelemetryExporterError) as exc_info:
            exporter.configure({**_VALID_CONFIG, "agent_port": 0})
        assert "agent_port" in str(exc_info.value)

    def test_invalid_port_negative_raises(self) -> None:
        """Negative port raises TelemetryExporterError."""
        exporter = DatadogExporter()
        with pytest.raises(TelemetryExporterError) as exc_info:
            exporter.configure({**_VALID_CONFIG, "agent_port": -1})
        assert "agent_port" in str(exc_info.value)

    def test_invalid_port_too_high_raises(self) -> None:
        """Port > 65535 raises TelemetryExporterError."""
        exporter = DatadogExporter()
        with pytest.raises(TelemetryExporterError) as exc_info:
            exporter.configure({**_VALID_CONFIG, "agent_port": 65536})
        assert "agent_port" in str(exc_info.value)

    def test_invalid_port_string_raises(self) -> None:
        """String port raises TelemetryExporterError."""
        exporter = DatadogExporter()
        with pytest.raises(TelemetryExporterError) as exc_info:
            exporter.configure({**_VALID_CONFIG, "agent_port": "8126"})
        assert "'agent_port' must be an integer" in str(exc_info.value)
        assert "str" in str(exc_info.value)

    def test_service_name_wrong_type_raises(self) -> None:
        """Non-string service_name raises TelemetryExporterError."""
        exporter = DatadogExporter()
        with pytest.raises(TelemetryExporterError) as exc_info:
            exporter.configure({**_VALID_CONFIG, "service_name": 123})
        assert "'service_name' must be a string" in str(exc_info.value)
        assert "int" in str(exc_info.value)

    def test_env_wrong_type_raises(self) -> None:
        """Non-string env raises TelemetryExporterError."""
        exporter = DatadogExporter()
        with pytest.raises(TelemetryExporterError) as exc_info:
            exporter.configure({**_VALID_CONFIG, "env": ["production"]})
        assert "'env' must be a string" in str(exc_info.value)
        assert "list" in str(exc_info.value)

    def test_agent_host_wrong_type_raises(self) -> None:
        """Non-string agent_host raises TelemetryExporterError."""
        exporter = DatadogExporter()
        with pytest.raises(TelemetryExporterError) as exc_info:
            exporter.configure({**_VALID_CONFIG, "agent_host": {"host": "localhost"}})
        assert "'agent_host' must be a string" in str(exc_info.value)
        assert "dict" in str(exc_info.value)

    def test_version_wrong_type_raises(self) -> None:
        """Non-string version raises TelemetryExporterError."""
        exporter = DatadogExporter()
        with pytest.raises(TelemetryExporterError) as exc_info:
            exporter.configure({**_VALID_CONFIG, "version": 1.2})
        assert "'version' must be a string" in str(exc_info.value)
        assert "float" in str(exc_info.value)

    def test_version_null_is_valid(self) -> None:
        """null/None version is valid (optional field)."""
        mock_module, _, _ = create_mock_ddtrace_module()

        with patch.dict(sys.modules, {"ddtrace": mock_module}):
            exporter = DatadogExporter()
            exporter.configure({**_VALID_CONFIG, "version": None})
            assert exporter._version is None

    def test_service_name_configuration(self) -> None:
        """Service name is stored from config."""
        mock_module, _, _ = create_mock_ddtrace_module()

        with patch.dict(sys.modules, {"ddtrace": mock_module}):
            exporter = DatadogExporter()
            exporter.configure({**_VALID_CONFIG, "service_name": "my-pipeline"})
            assert exporter._service_name == "my-pipeline"

    def test_env_configuration(self) -> None:
        """Environment is stored from config."""
        mock_module, _, _ = create_mock_ddtrace_module()

        with patch.dict(sys.modules, {"ddtrace": mock_module}):
            exporter = DatadogExporter()
            exporter.configure({**_VALID_CONFIG, "env": "staging"})
            assert exporter._env == "staging"

    def test_version_configuration(self) -> None:
        """Version is stored from config."""
        mock_module, _, _ = create_mock_ddtrace_module()

        with patch.dict(sys.modules, {"ddtrace": mock_module}):
            exporter = DatadogExporter()
            exporter.configure({**_VALID_CONFIG, "version": "1.2.3"})
            assert exporter._version == "1.2.3"

    def test_api_key_is_rejected(self) -> None:
        """api_key option is rejected with a clear configuration error."""
        mock_module, _, _ = create_mock_ddtrace_module()

        with patch.dict(sys.modules, {"ddtrace": mock_module}):
            exporter = DatadogExporter()
            with pytest.raises(TelemetryExporterError) as exc_info:
                exporter.configure({**_VALID_CONFIG, "api_key": "dd-api-key"})
            assert "does not support 'api_key'" in str(exc_info.value)

    def test_ddtrace_not_installed_raises(self) -> None:
        """Missing ddtrace raises TelemetryExporterError."""
        exporter = DatadogExporter()

        # Remove ddtrace from sys.modules if present and make import fail
        with patch.dict(sys.modules, {"ddtrace": None}):
            with pytest.raises(TelemetryExporterError) as exc_info:
                exporter.configure(dict(_VALID_CONFIG))
            assert "ddtrace" in str(exc_info.value).lower()


class TestDatadogExporterSpanCreation:
    """Tests for span creation from telemetry events."""

    def _create_configured_exporter(self) -> tuple[DatadogExporter, _FakeDatadogTracer, _FakeDatadogSpan]:
        """Create a configured exporter with mocked tracer.

        Returns:
            Tuple of (exporter, mock_tracer, mock_span)
        """
        mock_module, mock_tracer, mock_span = create_mock_ddtrace_module()

        with patch.dict(sys.modules, {"ddtrace": mock_module}):
            exporter = DatadogExporter()
            exporter.configure(dict(_VALID_CONFIG))

        return exporter, mock_tracer, mock_span

    def test_span_uses_event_timestamp_not_export_time(self) -> None:
        """Span start/finish times come from event.timestamp, not export time.

        This is critical for buffered/async export scenarios where the event
        may be created at time T but exported at time T+10s. The span should
        reflect when the event actually occurred, not when it was exported.

        ddtrace 4.x API: start_ns is set directly on the span after creation.
        """
        mock_module, mock_tracer, mock_span = create_mock_ddtrace_module()
        mock_tracer.start_span.return_value = mock_span
        mock_span.start_ns = 0  # Initialize so we can check it was set

        with patch.dict(sys.modules, {"ddtrace": mock_module}):
            exporter = DatadogExporter()
            exporter.configure(dict(_VALID_CONFIG))

        # Create event with a known timestamp in the past
        event_timestamp = datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC)
        expected_unix_seconds = event_timestamp.timestamp()
        expected_ns = int(expected_unix_seconds * 1_000_000_000)

        event = RunStarted(
            timestamp=event_timestamp,
            run_id="run-123",
            config_hash="abc",
            source_plugin="csv",
        )
        exporter.export(event)

        # Verify start_span was called (ddtrace 4.x: no 'start' parameter)
        mock_tracer.start_span.assert_called_once()
        call_kwargs = mock_tracer.start_span.call_args[1]
        assert "start" not in call_kwargs, "ddtrace 4.x: start_span no longer accepts 'start' parameter"

        # Verify start_ns was set directly on the span (ddtrace 4.x API)
        assert mock_span.start_ns == expected_ns, (
            f"Span start_ns should be event timestamp ({expected_ns}), not auto-generated. Got: {mock_span.start_ns}"
        )

        # Verify span was finished with the same timestamp (instant span)
        mock_span.finish.assert_called_once()
        finish_call_kwargs = mock_span.finish.call_args[1] if mock_span.finish.call_args[1] else {}
        finish_time = finish_call_kwargs.get("finish_time")
        assert finish_time == expected_unix_seconds, (
            f"Span finish time should be event timestamp ({expected_unix_seconds}) for instant span. Got: {finish_time}"
        )

    def test_span_name_is_event_class_name(self) -> None:
        """Span name is the event class name."""
        exporter, mock_tracer, _ = self._create_configured_exporter()

        event = RunStarted(
            timestamp=datetime.now(UTC),
            run_id="run-123",
            config_hash="abc",
            source_plugin="csv",
        )
        exporter.export(event)

        mock_tracer.start_span.assert_called_once()
        call_kwargs = mock_tracer.start_span.call_args[1]
        assert call_kwargs["name"] == "RunStarted"

    def test_span_service_name_from_config(self) -> None:
        """Span service comes from configuration."""
        exporter, mock_tracer, _ = self._create_configured_exporter()

        event = RunStarted(
            timestamp=datetime.now(UTC),
            run_id="run-123",
            config_hash="abc",
            source_plugin="csv",
        )
        exporter.export(event)

        call_kwargs = mock_tracer.start_span.call_args[1]
        assert call_kwargs["service"] == "test-service"

    def test_span_resource_is_event_type(self) -> None:
        """Span resource is the event class name."""
        exporter, mock_tracer, _ = self._create_configured_exporter()

        event = RunStarted(
            timestamp=datetime.now(UTC),
            run_id="run-123",
            config_hash="abc",
            source_plugin="csv",
        )
        exporter.export(event)

        call_kwargs = mock_tracer.start_span.call_args[1]
        assert call_kwargs["resource"] == "RunStarted"

    def test_env_tag_set(self) -> None:
        """Environment tag is set on span."""
        exporter, _mock_tracer, mock_span = self._create_configured_exporter()

        event = RunStarted(
            timestamp=datetime.now(UTC),
            run_id="run-123",
            config_hash="abc",
            source_plugin="csv",
        )
        exporter.export(event)

        mock_span.set_tag.assert_any_call("env", "test")

    def test_version_tag_set_when_configured(self) -> None:
        """Version tag is set when configured."""
        mock_module, _mock_tracer, mock_span = create_mock_ddtrace_module()

        with patch.dict(sys.modules, {"ddtrace": mock_module}):
            exporter = DatadogExporter()
            exporter.configure({**_VALID_CONFIG, "version": "1.0.0"})

        event = RunStarted(
            timestamp=datetime.now(UTC),
            run_id="run-123",
            config_hash="abc",
            source_plugin="csv",
        )
        exporter.export(event)

        mock_span.set_tag.assert_any_call("version", "1.0.0")

    def test_run_id_tag_set(self) -> None:
        """Run ID is set as elspeth.run_id tag."""
        exporter, _mock_tracer, mock_span = self._create_configured_exporter()

        event = RunStarted(
            timestamp=datetime.now(UTC),
            run_id="run-123",
            config_hash="abc",
            source_plugin="csv",
        )
        exporter.export(event)

        mock_span.set_tag.assert_any_call("elspeth.run_id", "run-123")

    def test_event_type_tag_set(self) -> None:
        """Event type is set as elspeth.event_type tag."""
        exporter, _mock_tracer, mock_span = self._create_configured_exporter()

        event = RunStarted(
            timestamp=datetime.now(UTC),
            run_id="run-123",
            config_hash="abc",
            source_plugin="csv",
        )
        exporter.export(event)

        mock_span.set_tag.assert_any_call("elspeth.event_type", "RunStarted")


class TestDatadogExporterTagSerialization:
    """Tests for tag serialization from event fields."""

    def _create_configured_exporter(self) -> tuple[DatadogExporter, _FakeDatadogTracer, _FakeDatadogSpan]:
        """Create a configured exporter with mocked tracer.

        Returns:
            Tuple of (exporter, mock_tracer, mock_span)
        """
        mock_module, mock_tracer, mock_span = create_mock_ddtrace_module()

        with patch.dict(sys.modules, {"ddtrace": mock_module}):
            exporter = DatadogExporter()
            exporter.configure(dict(_VALID_CONFIG))

        return exporter, mock_tracer, mock_span

    def test_string_field_set_directly(self) -> None:
        """String fields are set directly as tags."""
        exporter, _mock_tracer, mock_span = self._create_configured_exporter()

        event = RunStarted(
            timestamp=datetime.now(UTC),
            run_id="run-123",
            config_hash="abc123",
            source_plugin="csv_source",
        )
        exporter.export(event)

        mock_span.set_tag.assert_any_call("elspeth.config_hash", "abc123")
        mock_span.set_tag.assert_any_call("elspeth.source_plugin", "csv_source")

    def test_datetime_serialized_as_iso8601(self) -> None:
        """datetime fields are serialized as ISO 8601 strings."""
        exporter, _mock_tracer, mock_span = self._create_configured_exporter()

        ts = datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC)
        event = RunStarted(
            timestamp=ts,
            run_id="run-123",
            config_hash="abc",
            source_plugin="csv",
        )
        exporter.export(event)

        mock_span.set_tag.assert_any_call("elspeth.timestamp", "2024-01-15T10:30:00+00:00")

    def test_enum_serialized_as_value(self) -> None:
        """Enum fields are serialized as their string values."""
        exporter, _mock_tracer, mock_span = self._create_configured_exporter()

        event = RunFinished(
            timestamp=datetime.now(UTC),
            run_id="run-123",
            status=RunStatus.COMPLETED,
            row_count=100,
            duration_ms=5000.0,
        )
        exporter.export(event)

        mock_span.set_tag.assert_any_call("elspeth.status", "completed")

    def test_tuple_serialized_as_list(self) -> None:
        """Tuple fields are serialized as lists."""
        exporter, _mock_tracer, mock_span = self._create_configured_exporter()

        event = GateEvaluated(
            timestamp=datetime.now(UTC),
            run_id="run-123",
            row_id="row-1",
            token_id="token-1",
            node_id="gate-1",
            plugin_name="threshold_gate",
            routing_mode=RoutingMode.MOVE,
            destinations=("sink_a", "sink_b"),
        )
        exporter.export(event)

        mock_span.set_tag.assert_any_call("elspeth.destinations", ["sink_a", "sink_b"])

    def test_none_fields_skipped(self) -> None:
        """None fields are not set as tags."""
        exporter, _mock_tracer, mock_span = self._create_configured_exporter()

        event = TokenCompleted(
            timestamp=datetime.now(UTC),
            run_id="run-123",
            row_id="row-1",
            token_id="token-1",
            outcome=TerminalOutcome.FAILURE,
            path=TerminalPath.UNROUTED,
            sink_name=None,
        )
        exporter.export(event)

        # Collect all tag keys that were set
        tag_keys = [call[0][0] for call in mock_span.set_tag.call_args_list]
        assert "elspeth.sink_name" not in tag_keys

    def test_dict_flattened_to_dotted_keys(self) -> None:
        """Dict fields are flattened to dotted tag keys."""
        exporter, _mock_tracer, mock_span = self._create_configured_exporter()

        from elspeth.contracts.enums import CallStatus, CallType
        from elspeth.contracts.events import ExternalCallCompleted

        event = ExternalCallCompleted(
            timestamp=datetime.now(UTC),
            run_id="run-123",
            state_id="state-1",
            call_type=CallType.LLM,
            provider="azure-openai",
            status=CallStatus.SUCCESS,
            latency_ms=150.0,
            token_usage=TokenUsage(prompt_tokens=100, completion_tokens=50),
        )
        exporter.export(event)

        mock_span.set_tag.assert_any_call("elspeth.token_usage.prompt_tokens", 100)
        mock_span.set_tag.assert_any_call("elspeth.token_usage.completion_tokens", 50)

    def test_deeply_nested_dict_bounded_at_max_depth(self) -> None:
        """Deeply nested dicts are serialized as string beyond max depth.

        Regression test for bead ipfu: _set_tag_value recursed into dicts without
        any depth limit, meaning adversarial or malformed event data with deeply
        nested structures could cause a RecursionError (stack overflow).
        """
        exporter, _mock_tracer, mock_span = self._create_configured_exporter()

        # Build a dict nested deeper than _MAX_TAG_DEPTH (5)
        deep: dict[str, object] = {"leaf": "value"}
        for i in range(8):
            deep = {f"level_{i}": deep}

        # Call _set_tag_value directly to test the depth bounding
        exporter._set_tag_value(mock_span, "elspeth.deep", deep)

        # Collect all tag keys that were set
        tag_calls = {call[0][0]: call[0][1] for call in mock_span.set_tag.call_args_list}

        # At depth 5, the remaining dict should be serialized as string
        # rather than continuing to recurse
        deep_keys = [k for k in tag_calls if k.startswith("elspeth.deep")]
        assert len(deep_keys) > 0

        # Verify that no key goes deeper than _MAX_TAG_DEPTH levels of nesting
        max_dots = max(k.count(".") for k in deep_keys)
        # "elspeth.deep" = 1 dot, plus up to _MAX_TAG_DEPTH levels = 6 dots max
        assert max_dots <= 1 + exporter._MAX_TAG_DEPTH

        # The deepest tag should be a string (serialized remaining dict)
        deepest_key = max(deep_keys, key=lambda k: k.count("."))
        deepest_value = tag_calls[deepest_key]
        assert isinstance(deepest_value, str), f"At max depth, remaining dict should be serialized as str, got {type(deepest_value)}"

    def test_shallow_dict_still_flattened(self) -> None:
        """Shallow dicts (under max depth) are still flattened normally."""
        exporter, _mock_tracer, mock_span = self._create_configured_exporter()

        shallow = {"a": {"b": "value"}}
        exporter._set_tag_value(mock_span, "elspeth.test", shallow)

        mock_span.set_tag.assert_any_call("elspeth.test.a.b", "value")

    def test_int_field_set_directly(self) -> None:
        """Integer fields are set directly as tags."""
        exporter, _mock_tracer, mock_span = self._create_configured_exporter()

        event = RunFinished(
            timestamp=datetime.now(UTC),
            run_id="run-123",
            status=RunStatus.COMPLETED,
            row_count=42,
            duration_ms=5000.0,
        )
        exporter.export(event)

        mock_span.set_tag.assert_any_call("elspeth.row_count", 42)

    def test_float_field_set_directly(self) -> None:
        """Float fields are set directly as tags."""
        exporter, _mock_tracer, mock_span = self._create_configured_exporter()

        event = RunFinished(
            timestamp=datetime.now(UTC),
            run_id="run-123",
            status=RunStatus.COMPLETED,
            row_count=100,
            duration_ms=1234.56,
        )
        exporter.export(event)

        mock_span.set_tag.assert_any_call("elspeth.duration_ms", 1234.56)


class TestDatadogExporterLifecycle:
    """Tests for exporter lifecycle (flush, close, error handling)."""

    def _create_configured_exporter(self) -> tuple[DatadogExporter, _FakeDatadogTracer, _FakeDatadogSpan]:
        """Create a configured exporter with mocked tracer.

        Returns:
            Tuple of (exporter, mock_tracer, mock_span)
        """
        mock_module, mock_tracer, mock_span = create_mock_ddtrace_module()

        with patch.dict(sys.modules, {"ddtrace": mock_module}):
            exporter = DatadogExporter()
            exporter.configure(dict(_VALID_CONFIG))

        return exporter, mock_tracer, mock_span

    def test_flush_calls_tracer_flush(self) -> None:
        """flush() calls tracer.flush()."""
        exporter, mock_tracer, _ = self._create_configured_exporter()
        exporter.flush()
        mock_tracer.flush.assert_called_once()

    def test_close_flushes_first(self) -> None:
        """close() flushes before shutdown."""
        exporter, mock_tracer, _ = self._create_configured_exporter()
        exporter.close()
        mock_tracer.flush.assert_called_once()

    def test_close_shuts_down_tracer(self) -> None:
        """close() calls tracer.shutdown()."""
        exporter, mock_tracer, _ = self._create_configured_exporter()
        exporter.close()
        mock_tracer.shutdown.assert_called_once()

    def test_close_is_idempotent(self) -> None:
        """close() can be called multiple times safely."""
        exporter, mock_tracer, _ = self._create_configured_exporter()
        exporter.close()
        exporter.close()  # Should not raise
        # shutdown called only once (tracer set to None after first close)
        mock_tracer.shutdown.assert_called_once()

    def test_export_without_configure_logs_warning(self) -> None:
        """export() without configure() logs warning and continues."""
        exporter = DatadogExporter()
        event = RunStarted(
            timestamp=datetime.now(UTC),
            run_id="run-123",
            config_hash="abc",
            source_plugin="csv",
        )

        # Should not raise
        exporter.export(event)

    def test_export_failure_does_not_raise(self) -> None:
        """Export failures are logged but don't raise exceptions."""
        exporter, mock_tracer, _ = self._create_configured_exporter()
        mock_tracer.start_span.side_effect = ConnectionError("Tracer transport error")

        event = RunStarted(
            timestamp=datetime.now(UTC),
            run_id="run-123",
            config_hash="abc",
            source_plugin="csv",
        )

        # Should not raise
        exporter.export(event)

    def test_flush_failure_does_not_raise(self) -> None:
        """flush() failures are logged but don't raise exceptions."""
        exporter, mock_tracer, _ = self._create_configured_exporter()
        mock_tracer.flush.side_effect = ConnectionError("Flush error")

        # Should not raise
        exporter.flush()

    def test_close_failure_does_not_raise(self) -> None:
        """close() shutdown failures are logged but don't raise."""
        exporter, mock_tracer, _ = self._create_configured_exporter()
        mock_tracer.shutdown.side_effect = ConnectionError("Shutdown error")

        # Should not raise
        exporter.close()

    def test_programming_error_in_export_crashes(self) -> None:
        """Programming errors (non-transport) must crash — not be swallowed."""
        exporter, mock_tracer, _ = self._create_configured_exporter()
        mock_tracer.start_span.side_effect = ValueError("Bad payload construction")

        event = RunStarted(
            timestamp=datetime.now(UTC),
            run_id="run-123",
            config_hash="abc",
            source_plugin="csv",
        )

        with pytest.raises(ValueError, match="Bad payload construction"):
            exporter.export(event)


class TestDatadogExporterRegistration:
    """Tests for plugin registration."""

    def test_exporter_in_builtin_plugin(self) -> None:
        """DatadogExporter is registered in BuiltinExportersPlugin."""
        from elspeth.telemetry.exporters import BuiltinExportersPlugin, DatadogExporter

        plugin = BuiltinExportersPlugin()
        exporters = plugin.elspeth_get_exporters()
        assert DatadogExporter in exporters

    def test_exporter_in_package_all(self) -> None:
        """DatadogExporter is exported from package __all__."""
        from elspeth.telemetry import exporters

        assert "DatadogExporter" in exporters.__all__
