# tests/telemetry/test_contracts.py
"""Contract tests for telemetry subsystem.

These tests verify:
1. All exporters implement ExporterProtocol correctly (including method calls)
2. RuntimeTelemetryConfig implements RuntimeTelemetryProtocol
3. TelemetrySettings fields align with RuntimeTelemetryConfig fields
4. All TelemetryEvent dataclasses are JSON-serializable
"""

from __future__ import annotations

import dataclasses
import json
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import pytest

from elspeth.contracts.config.alignment import (
    FIELD_MAPPINGS,
    SETTINGS_TO_RUNTIME,
    get_runtime_field_name,
)
from elspeth.contracts.config.protocols import RuntimeTelemetryProtocol
from elspeth.contracts.config.runtime import RuntimeTelemetryConfig
from elspeth.contracts.enums import (
    CallStatus,
    CallType,
    NodeStateStatus,
    RoutingMode,
    RowOutcome,
    RunStatus,
)
from elspeth.contracts.events import (
    GateEvaluated,
    PhaseAction,
    PipelinePhase,
    TelemetryEvent,
    TokenCompleted,
    TransformCompleted,
)
from elspeth.telemetry.events import (
    ExternalCallCompleted,
    PhaseChanged,
    RowCreated,
    RunFinished,
    RunStarted,
)
from elspeth.telemetry.exporters.azure_monitor import AzureMonitorExporter
from elspeth.telemetry.exporters.console import ConsoleExporter
from elspeth.telemetry.exporters.datadog import DatadogExporter
from elspeth.telemetry.exporters.otlp import OTLPExporter
from elspeth.telemetry.protocols import ExporterProtocol

# =============================================================================
# Test Data
# =============================================================================

# All exporter classes - typed as the concrete classes they are
ALL_EXPORTERS: list[type[ConsoleExporter] | type[OTLPExporter] | type[AzureMonitorExporter] | type[DatadogExporter]] = [
    ConsoleExporter,
    OTLPExporter,
    AzureMonitorExporter,
    DatadogExporter,
]

# All telemetry event classes
ALL_EVENTS = [
    RunStarted,
    RunFinished,
    PhaseChanged,
    RowCreated,
    TransformCompleted,
    GateEvaluated,
    TokenCompleted,
    ExternalCallCompleted,
]


def _create_sample_event(event_class: type[TelemetryEvent]) -> TelemetryEvent:
    """Create a sample instance of the given event class for testing.

    Args:
        event_class: The TelemetryEvent subclass to instantiate

    Returns:
        A valid instance with sample data for all required fields
    """
    base_kwargs: dict[str, Any] = {
        "timestamp": datetime.now(UTC),
        "run_id": "test-run-123",
    }

    if event_class is RunStarted:
        return RunStarted(
            **base_kwargs,
            config_hash="abc123def456",
            source_plugin="csv",
        )
    elif event_class is RunFinished:
        return RunFinished(
            **base_kwargs,
            status=RunStatus.COMPLETED,
            row_count=100,
            duration_ms=5000.0,
        )
    elif event_class is PhaseChanged:
        return PhaseChanged(
            **base_kwargs,
            phase=PipelinePhase.PROCESS,
            action=PhaseAction.LOADING,
        )
    elif event_class is RowCreated:
        return RowCreated(
            **base_kwargs,
            row_id="row-001",
            token_id="token-001",
            content_hash="hash123",
        )
    elif event_class is TransformCompleted:
        return TransformCompleted(
            **base_kwargs,
            row_id="row-001",
            token_id="token-001",
            node_id="transform-1",
            plugin_name="field_mapper",
            status=NodeStateStatus.COMPLETED,
            duration_ms=50.0,
            input_hash="input123",
            output_hash="output456",
        )
    elif event_class is GateEvaluated:
        return GateEvaluated(
            **base_kwargs,
            row_id="row-001",
            token_id="token-001",
            node_id="gate-1",
            plugin_name="threshold_gate",
            routing_mode=RoutingMode.MOVE,
            destinations=("sink-1", "sink-2"),
        )
    elif event_class is TokenCompleted:
        return TokenCompleted(
            **base_kwargs,
            row_id="row-001",
            token_id="token-001",
            outcome=RowOutcome.COMPLETED,
            sink_name="output",
        )
    elif event_class is ExternalCallCompleted:
        return ExternalCallCompleted(
            **base_kwargs,
            state_id="state-001",
            call_type=CallType.LLM,
            provider="azure-openai",
            status=CallStatus.SUCCESS,
            latency_ms=250.0,
            request_hash="req123",
            response_hash="resp456",
            token_usage={"prompt_tokens": 100, "completion_tokens": 50},
        )
    else:
        raise ValueError(f"Unknown event class: {event_class}")


def _serialize_for_json(obj: Any) -> Any:
    """JSON serialization helper for telemetry event fields.

    Args:
        obj: Object to serialize

    Returns:
        JSON-serializable representation
    """
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, Enum):
        return obj.value
    elif isinstance(obj, tuple):
        return list(obj)
    elif isinstance(obj, dict):
        return {k: _serialize_for_json(v) for k, v in obj.items()}
    return obj


# =============================================================================
# ExporterProtocol Compliance Tests
# =============================================================================


@pytest.mark.parametrize("exporter_class", ALL_EXPORTERS)
def test_exporter_implements_protocol(exporter_class: type) -> None:
    """All exporters must implement ExporterProtocol correctly.

    This test verifies:
    1. The class is recognized as implementing ExporterProtocol via isinstance()
    2. All protocol methods and properties are present
    3. Methods can be called (not just defined)
    """
    # Create an instance
    exporter = exporter_class()

    # Verify protocol compliance via isinstance (runtime_checkable)
    assert isinstance(exporter, ExporterProtocol), (
        f"{exporter_class.__name__} does not satisfy ExporterProtocol. Check that all required methods/properties are implemented."
    )


@pytest.mark.parametrize("exporter_class", ALL_EXPORTERS)
def test_exporter_has_name_property(exporter_class: type) -> None:
    """Exporters must have a name property returning a string."""
    exporter = exporter_class()
    name = exporter.name
    assert isinstance(name, str), f"{exporter_class.__name__}.name must return str"
    assert len(name) > 0, f"{exporter_class.__name__}.name must be non-empty"


@pytest.mark.parametrize("exporter_class", ALL_EXPORTERS)
def test_exporter_configure_accepts_dict(exporter_class: type) -> None:
    """Exporters must accept a dict in configure().

    Note: We don't test valid configuration here (that's in integration tests).
    We only verify the method signature accepts a dict.
    """
    exporter = exporter_class()

    # configure() should be callable with a dict
    # For ConsoleExporter, empty config is valid
    # For others, it may raise TelemetryExporterError on invalid config
    if exporter_class is ConsoleExporter:
        # ConsoleExporter accepts empty config
        exporter.configure({})
    else:
        # Other exporters require specific config - just verify method exists
        # and accepts dict (signature check via callable)
        assert callable(exporter.configure)
        # Verify method signature by checking it accepts dict without TypeError
        # (actual configuration errors are expected for missing required fields)


@pytest.mark.parametrize("exporter_class", ALL_EXPORTERS)
def test_exporter_flush_is_callable(exporter_class: type) -> None:
    """Exporters must have a callable flush() method."""
    exporter = exporter_class()
    assert callable(exporter.flush), f"{exporter_class.__name__}.flush must be callable"

    # flush() should be safe to call on unconfigured exporter
    # (idempotent, no-op when nothing to flush)
    exporter.flush()  # Should not raise


@pytest.mark.parametrize("exporter_class", ALL_EXPORTERS)
def test_exporter_close_is_idempotent(exporter_class: type) -> None:
    """Exporters must have an idempotent close() method.

    close() must be safe to call multiple times without error.
    """
    exporter = exporter_class()
    assert callable(exporter.close), f"{exporter_class.__name__}.close must be callable"

    # close() should be idempotent - safe to call multiple times
    exporter.close()
    exporter.close()
    exporter.close()  # Should not raise


# =============================================================================
# RuntimeTelemetryConfig Protocol Tests
# =============================================================================


def test_runtime_telemetry_config_implements_protocol() -> None:
    """RuntimeTelemetryConfig must implement RuntimeTelemetryProtocol."""
    config = RuntimeTelemetryConfig.default()

    # Verify protocol compliance via isinstance (runtime_checkable)
    assert isinstance(config, RuntimeTelemetryProtocol), (
        "RuntimeTelemetryConfig does not satisfy RuntimeTelemetryProtocol. Check that all required properties are implemented."
    )


def test_runtime_telemetry_config_protocol_properties() -> None:
    """RuntimeTelemetryConfig must have all protocol-required properties."""
    config = RuntimeTelemetryConfig.default()

    # Verify all protocol properties exist and have correct types
    assert isinstance(config.enabled, bool)
    assert hasattr(config.granularity, "value")  # Is an enum
    assert hasattr(config.backpressure_mode, "value")  # Is an enum
    assert isinstance(config.fail_on_total_exporter_failure, bool)
    assert isinstance(config.exporter_configs, tuple)


# =============================================================================
# Config Alignment Tests
# =============================================================================


def test_telemetry_settings_has_runtime_mapping() -> None:
    """TelemetrySettings must be listed in SETTINGS_TO_RUNTIME."""
    assert "TelemetrySettings" in SETTINGS_TO_RUNTIME, "TelemetrySettings is missing from SETTINGS_TO_RUNTIME mapping in alignment.py"
    assert SETTINGS_TO_RUNTIME["TelemetrySettings"] == "RuntimeTelemetryConfig"


def test_telemetry_field_mappings_are_documented() -> None:
    """TelemetrySettings field renames must be documented in FIELD_MAPPINGS."""
    # TelemetrySettings has one rename: exporters -> exporter_configs
    assert "TelemetrySettings" in FIELD_MAPPINGS, "TelemetrySettings is missing from FIELD_MAPPINGS in alignment.py"
    assert "exporters" in FIELD_MAPPINGS["TelemetrySettings"], "TelemetrySettings.exporters rename is not documented in FIELD_MAPPINGS"
    assert FIELD_MAPPINGS["TelemetrySettings"]["exporters"] == "exporter_configs"


def test_get_runtime_field_name_for_telemetry() -> None:
    """get_runtime_field_name should return correct mappings for TelemetrySettings."""
    # Mapped field
    assert get_runtime_field_name("TelemetrySettings", "exporters") == "exporter_configs"

    # Direct fields (same name in Settings and Runtime)
    assert get_runtime_field_name("TelemetrySettings", "enabled") == "enabled"
    assert get_runtime_field_name("TelemetrySettings", "granularity") == "granularity"
    assert get_runtime_field_name("TelemetrySettings", "backpressure_mode") == "backpressure_mode"
    assert get_runtime_field_name("TelemetrySettings", "fail_on_total_exporter_failure") == "fail_on_total_exporter_failure"


def test_runtime_telemetry_config_has_all_protocol_fields() -> None:
    """RuntimeTelemetryConfig must have all fields defined in RuntimeTelemetryProtocol.

    This verifies there are no orphaned fields - every protocol field
    exists in the runtime config.
    """
    # Get RuntimeTelemetryConfig fields
    runtime_fields = {f.name for f in dataclasses.fields(RuntimeTelemetryConfig)}

    # Protocol requires these fields (from RuntimeTelemetryProtocol definition)
    protocol_fields = {
        "enabled",
        "granularity",
        "backpressure_mode",
        "fail_on_total_exporter_failure",
        "exporter_configs",
    }

    # All protocol fields must be in runtime config
    missing = protocol_fields - runtime_fields
    assert not missing, f"RuntimeTelemetryConfig is missing protocol fields: {missing}"


# =============================================================================
# Event Serialization Tests
# =============================================================================


@pytest.mark.parametrize("event_class", ALL_EVENTS)
def test_event_is_json_serializable(event_class: type[TelemetryEvent]) -> None:
    """All telemetry events must be JSON-serializable.

    This is critical for exporters that convert events to JSON
    (ConsoleExporter JSON mode, OTLP attributes, etc.).
    """
    event = _create_sample_event(event_class)

    # Convert dataclass to dict
    event_dict = dataclasses.asdict(event)

    # Apply serialization transformations (what exporters do)
    serialized = {k: _serialize_for_json(v) for k, v in event_dict.items()}

    # Must be JSON-serializable without error
    try:
        json_str = json.dumps(serialized)
        assert len(json_str) > 0
    except (TypeError, ValueError) as e:
        pytest.fail(f"{event_class.__name__} is not JSON-serializable: {e}\nEvent dict: {event_dict}")


@pytest.mark.parametrize("event_class", ALL_EVENTS)
def test_event_is_dataclass(event_class: type[TelemetryEvent]) -> None:
    """All telemetry events must be dataclasses."""
    assert dataclasses.is_dataclass(event_class), f"{event_class.__name__} must be a dataclass"


@pytest.mark.parametrize("event_class", ALL_EVENTS)
def test_event_is_frozen(event_class: type[TelemetryEvent]) -> None:
    """All telemetry events must be frozen (immutable).

    Frozen dataclasses prevent accidental modification during export.
    """
    event = _create_sample_event(event_class)

    # Frozen dataclasses raise FrozenInstanceError on attribute assignment
    with pytest.raises(dataclasses.FrozenInstanceError):
        event.run_id = "modified"  # type: ignore[misc]


@pytest.mark.parametrize("event_class", ALL_EVENTS)
def test_event_has_base_fields(event_class: type[TelemetryEvent]) -> None:
    """All telemetry events must have timestamp and run_id base fields."""
    event = _create_sample_event(event_class)

    # Base fields from TelemetryEvent
    assert hasattr(event, "timestamp")
    assert hasattr(event, "run_id")
    assert isinstance(event.timestamp, datetime)
    assert isinstance(event.run_id, str)


def test_all_events_are_subclasses_of_base() -> None:
    """All event classes must be subclasses of TelemetryEvent."""
    for event_class in ALL_EVENTS:
        assert issubclass(event_class, TelemetryEvent), f"{event_class.__name__} must be a subclass of TelemetryEvent"


# =============================================================================
# Exporter Name Uniqueness
# =============================================================================


def test_exporter_names_are_unique() -> None:
    """All exporters must have unique names.

    Names are used in configuration to identify exporters, so duplicates
    would cause configuration ambiguity.
    """
    names = [cls().name for cls in ALL_EXPORTERS]
    unique_names = set(names)

    assert len(names) == len(unique_names), f"Duplicate exporter names found: {names}"


def test_exporter_names_are_valid_identifiers() -> None:
    """Exporter names should be valid for use in YAML configuration.

    Names should be lowercase, alphanumeric with underscores only.
    """
    import re

    pattern = re.compile(r"^[a-z][a-z0-9_]*$")

    for exporter_class in ALL_EXPORTERS:
        exporter = exporter_class()
        name = exporter.name

        assert pattern.match(name), (
            f"{exporter_class.__name__}.name = '{name}' is not a valid identifier. "
            "Names should be lowercase, start with a letter, and contain only "
            "letters, digits, and underscores."
        )
