# tests/unit/telemetry/test_events.py
"""Unit tests for telemetry event dataclasses.

Tests cover:
- Event instantiation and immutability
- JSON serialization (dataclasses.asdict + json.dumps)
- JSON roundtrip (serialize then deserialize)
- All event types with their specific fields
"""

import json
from dataclasses import asdict
from datetime import UTC, datetime

import pytest

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
from elspeth.telemetry import (
    ExternalCallCompleted,
    PhaseChanged,
    RowCreated,
    RunFinished,
    RunStarted,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def base_timestamp() -> datetime:
    """Fixed timestamp for deterministic tests."""
    return datetime(2026, 1, 30, 12, 0, 0, tzinfo=UTC)


@pytest.fixture
def base_run_id() -> str:
    """Fixed run ID for tests."""
    return "run-123-abc"


# =============================================================================
# Base TelemetryEvent Tests
# =============================================================================


class TestTelemetryEvent:
    """Tests for the base TelemetryEvent class."""

    def test_instantiation(self, base_timestamp: datetime, base_run_id: str) -> None:
        """TelemetryEvent can be instantiated with required fields."""
        event = TelemetryEvent(timestamp=base_timestamp, run_id=base_run_id)
        assert event.timestamp == base_timestamp
        assert event.run_id == base_run_id

    def test_frozen(self, base_timestamp: datetime, base_run_id: str) -> None:
        """TelemetryEvent is immutable (frozen)."""
        event = TelemetryEvent(timestamp=base_timestamp, run_id=base_run_id)
        with pytest.raises(AttributeError):
            event.run_id = "new-run-id"  # type: ignore[misc]

    def test_slots(self, base_timestamp: datetime, base_run_id: str) -> None:
        """TelemetryEvent uses slots for memory efficiency."""
        event = TelemetryEvent(timestamp=base_timestamp, run_id=base_run_id)
        assert hasattr(event, "__slots__")
        # With frozen=True and slots=True, attempting to add a new attribute
        # raises TypeError (not AttributeError) due to dataclass internals
        with pytest.raises((AttributeError, TypeError)):
            event.new_attribute = "value"  # type: ignore[attr-defined]


# =============================================================================
# Lifecycle Event Tests
# =============================================================================


class TestRunStarted:
    """Tests for RunStarted event."""

    def test_instantiation(self, base_timestamp: datetime, base_run_id: str) -> None:
        """RunStarted can be instantiated with all required fields."""
        event = RunStarted(
            timestamp=base_timestamp,
            run_id=base_run_id,
            config_hash="abc123",
            source_plugin="csv_source",
        )
        assert event.config_hash == "abc123"
        assert event.source_plugin == "csv_source"

    def test_json_roundtrip(self, base_timestamp: datetime, base_run_id: str) -> None:
        """RunStarted survives JSON roundtrip."""
        event = RunStarted(
            timestamp=base_timestamp,
            run_id=base_run_id,
            config_hash="abc123",
            source_plugin="csv_source",
        )
        serialized = json.dumps(asdict(event), default=str)
        deserialized = json.loads(serialized)

        assert deserialized["run_id"] == base_run_id
        assert deserialized["config_hash"] == "abc123"
        assert deserialized["source_plugin"] == "csv_source"


class TestRunFinished:
    """Tests for RunFinished event."""

    def test_instantiation(self, base_timestamp: datetime, base_run_id: str) -> None:
        """RunFinished can be instantiated with all required fields."""
        event = RunFinished(
            timestamp=base_timestamp,
            run_id=base_run_id,
            status=RunStatus.COMPLETED,
            row_count=100,
            duration_ms=5000.5,
        )
        assert event.status == RunStatus.COMPLETED
        assert event.row_count == 100
        assert event.duration_ms == 5000.5

    def test_json_roundtrip(self, base_timestamp: datetime, base_run_id: str) -> None:
        """RunFinished survives JSON roundtrip with enum serialization."""
        event = RunFinished(
            timestamp=base_timestamp,
            run_id=base_run_id,
            status=RunStatus.FAILED,
            row_count=50,
            duration_ms=2500.0,
        )
        serialized = json.dumps(asdict(event), default=str)
        deserialized = json.loads(serialized)

        assert deserialized["status"] == "failed"
        assert deserialized["row_count"] == 50
        assert deserialized["duration_ms"] == 2500.0


class TestPhaseChanged:
    """Tests for PhaseChanged event."""

    def test_instantiation(self, base_timestamp: datetime, base_run_id: str) -> None:
        """PhaseChanged can be instantiated with all required fields."""
        event = PhaseChanged(
            timestamp=base_timestamp,
            run_id=base_run_id,
            phase=PipelinePhase.PROCESS,
            action=PhaseAction.PROCESSING,
        )
        assert event.phase == PipelinePhase.PROCESS
        assert event.action == PhaseAction.PROCESSING

    def test_json_roundtrip(self, base_timestamp: datetime, base_run_id: str) -> None:
        """PhaseChanged survives JSON roundtrip with enum serialization."""
        event = PhaseChanged(
            timestamp=base_timestamp,
            run_id=base_run_id,
            phase=PipelinePhase.SOURCE,
            action=PhaseAction.LOADING,
        )
        serialized = json.dumps(asdict(event), default=str)
        deserialized = json.loads(serialized)

        assert deserialized["phase"] == "source"
        assert deserialized["action"] == "loading"


# =============================================================================
# Row-Level Event Tests
# =============================================================================


class TestRowCreated:
    """Tests for RowCreated event."""

    def test_instantiation(self, base_timestamp: datetime, base_run_id: str) -> None:
        """RowCreated can be instantiated with all required fields."""
        event = RowCreated(
            timestamp=base_timestamp,
            run_id=base_run_id,
            row_id="row-1",
            token_id="token-1",
            content_hash="hash123",
        )
        assert event.row_id == "row-1"
        assert event.token_id == "token-1"
        assert event.content_hash == "hash123"

    def test_json_roundtrip(self, base_timestamp: datetime, base_run_id: str) -> None:
        """RowCreated survives JSON roundtrip."""
        event = RowCreated(
            timestamp=base_timestamp,
            run_id=base_run_id,
            row_id="row-1",
            token_id="token-1",
            content_hash="hash123",
        )
        serialized = json.dumps(asdict(event), default=str)
        deserialized = json.loads(serialized)

        assert deserialized["row_id"] == "row-1"
        assert deserialized["token_id"] == "token-1"
        assert deserialized["content_hash"] == "hash123"


class TestTransformCompleted:
    """Tests for TransformCompleted event."""

    def test_instantiation(self, base_timestamp: datetime, base_run_id: str) -> None:
        """TransformCompleted can be instantiated with all required fields."""
        event = TransformCompleted(
            timestamp=base_timestamp,
            run_id=base_run_id,
            row_id="row-1",
            token_id="token-1",
            node_id="node-1",
            plugin_name="field_mapper",
            status=NodeStateStatus.COMPLETED,
            duration_ms=10.5,
            input_hash="in-hash",
            output_hash="out-hash",
        )
        assert event.node_id == "node-1"
        assert event.plugin_name == "field_mapper"
        assert event.status == NodeStateStatus.COMPLETED
        assert event.duration_ms == 10.5

    def test_json_roundtrip(self, base_timestamp: datetime, base_run_id: str) -> None:
        """TransformCompleted survives JSON roundtrip."""
        event = TransformCompleted(
            timestamp=base_timestamp,
            run_id=base_run_id,
            row_id="row-1",
            token_id="token-1",
            node_id="node-1",
            plugin_name="llm_classifier",
            status=NodeStateStatus.FAILED,
            duration_ms=500.0,
            input_hash="in-hash",
            output_hash="out-hash",
        )
        serialized = json.dumps(asdict(event), default=str)
        deserialized = json.loads(serialized)

        assert deserialized["node_id"] == "node-1"
        assert deserialized["plugin_name"] == "llm_classifier"
        assert deserialized["status"] == "failed"
        assert deserialized["duration_ms"] == 500.0


class TestGateEvaluated:
    """Tests for GateEvaluated event."""

    def test_instantiation(self, base_timestamp: datetime, base_run_id: str) -> None:
        """GateEvaluated can be instantiated with all required fields."""
        event = GateEvaluated(
            timestamp=base_timestamp,
            run_id=base_run_id,
            row_id="row-1",
            token_id="token-1",
            node_id="gate-1",
            plugin_name="threshold_gate",
            routing_mode=RoutingMode.MOVE,
            destinations=("high_priority", "urgent"),
        )
        assert event.routing_mode == RoutingMode.MOVE
        assert event.destinations == ("high_priority", "urgent")

    def test_json_roundtrip_tuple_to_list(self, base_timestamp: datetime, base_run_id: str) -> None:
        """GateEvaluated tuple becomes list in JSON, which is expected."""
        event = GateEvaluated(
            timestamp=base_timestamp,
            run_id=base_run_id,
            row_id="row-1",
            token_id="token-1",
            node_id="gate-1",
            plugin_name="threshold_gate",
            routing_mode=RoutingMode.COPY,
            destinations=("sink_a", "sink_b"),
        )
        serialized = json.dumps(asdict(event), default=str)
        deserialized = json.loads(serialized)

        assert deserialized["routing_mode"] == "copy"
        # JSON doesn't have tuples, so it becomes a list
        assert deserialized["destinations"] == ["sink_a", "sink_b"]

    def test_empty_destinations(self, base_timestamp: datetime, base_run_id: str) -> None:
        """GateEvaluated handles empty destinations."""
        event = GateEvaluated(
            timestamp=base_timestamp,
            run_id=base_run_id,
            row_id="row-1",
            token_id="token-1",
            node_id="gate-1",
            plugin_name="drop_gate",
            routing_mode=RoutingMode.MOVE,
            destinations=(),
        )
        serialized = json.dumps(asdict(event), default=str)
        deserialized = json.loads(serialized)

        assert deserialized["destinations"] == []


class TestTokenCompleted:
    """Tests for TokenCompleted event."""

    def test_instantiation_with_sink(self, base_timestamp: datetime, base_run_id: str) -> None:
        """TokenCompleted can be instantiated with a sink name."""
        event = TokenCompleted(
            timestamp=base_timestamp,
            run_id=base_run_id,
            row_id="row-1",
            token_id="token-1",
            outcome=RowOutcome.COMPLETED,
            sink_name="output",
        )
        assert event.outcome == RowOutcome.COMPLETED
        assert event.sink_name == "output"

    def test_instantiation_without_sink(self, base_timestamp: datetime, base_run_id: str) -> None:
        """TokenCompleted can be instantiated without a sink name."""
        event = TokenCompleted(
            timestamp=base_timestamp,
            run_id=base_run_id,
            row_id="row-1",
            token_id="token-1",
            outcome=RowOutcome.QUARANTINED,
            sink_name=None,
        )
        assert event.outcome == RowOutcome.QUARANTINED
        assert event.sink_name is None

    def test_json_roundtrip(self, base_timestamp: datetime, base_run_id: str) -> None:
        """TokenCompleted survives JSON roundtrip."""
        event = TokenCompleted(
            timestamp=base_timestamp,
            run_id=base_run_id,
            row_id="row-1",
            token_id="token-1",
            outcome=RowOutcome.ROUTED,
            sink_name="error_sink",
        )
        serialized = json.dumps(asdict(event), default=str)
        deserialized = json.loads(serialized)

        assert deserialized["outcome"] == "routed"
        assert deserialized["sink_name"] == "error_sink"


# =============================================================================
# External Call Event Tests
# =============================================================================


class TestExternalCallCompleted:
    """Tests for ExternalCallCompleted event."""

    def test_instantiation_minimal(self, base_timestamp: datetime, base_run_id: str) -> None:
        """ExternalCallCompleted can be instantiated with minimal fields."""
        event = ExternalCallCompleted(
            timestamp=base_timestamp,
            run_id=base_run_id,
            state_id="state-1",
            call_type=CallType.HTTP,
            provider="internal-api",
            status=CallStatus.SUCCESS,
            latency_ms=50.0,
        )
        assert event.call_type == CallType.HTTP
        assert event.provider == "internal-api"
        assert event.status == CallStatus.SUCCESS
        assert event.request_hash is None
        assert event.response_hash is None
        assert event.token_usage is None

    def test_instantiation_full(self, base_timestamp: datetime, base_run_id: str) -> None:
        """ExternalCallCompleted can be instantiated with all fields."""
        event = ExternalCallCompleted(
            timestamp=base_timestamp,
            run_id=base_run_id,
            state_id="state-1",
            call_type=CallType.LLM,
            provider="azure-openai",
            status=CallStatus.SUCCESS,
            latency_ms=1500.0,
            request_hash="req-hash",
            response_hash="resp-hash",
            token_usage={"prompt_tokens": 100, "completion_tokens": 50},
        )
        assert event.call_type == CallType.LLM
        assert event.request_hash == "req-hash"
        assert event.response_hash == "resp-hash"
        assert event.token_usage == {"prompt_tokens": 100, "completion_tokens": 50}

    def test_json_roundtrip_with_token_usage(self, base_timestamp: datetime, base_run_id: str) -> None:
        """ExternalCallCompleted survives JSON roundtrip including dict fields."""
        event = ExternalCallCompleted(
            timestamp=base_timestamp,
            run_id=base_run_id,
            state_id="state-1",
            call_type=CallType.LLM,
            provider="anthropic",
            status=CallStatus.ERROR,
            latency_ms=5000.0,
            request_hash="req-hash",
            response_hash="resp-hash",
            token_usage={"prompt_tokens": 200, "completion_tokens": 0},
        )
        serialized = json.dumps(asdict(event), default=str)
        deserialized = json.loads(serialized)

        assert deserialized["call_type"] == "llm"
        assert deserialized["status"] == "error"
        assert deserialized["token_usage"] == {
            "prompt_tokens": 200,
            "completion_tokens": 0,
        }

    def test_json_roundtrip_without_optional_fields(self, base_timestamp: datetime, base_run_id: str) -> None:
        """ExternalCallCompleted handles None optional fields in JSON."""
        event = ExternalCallCompleted(
            timestamp=base_timestamp,
            run_id=base_run_id,
            state_id="state-1",
            call_type=CallType.SQL,
            provider="postgres",
            status=CallStatus.SUCCESS,
            latency_ms=25.0,
        )
        serialized = json.dumps(asdict(event), default=str)
        deserialized = json.loads(serialized)

        assert deserialized["request_hash"] is None
        assert deserialized["response_hash"] is None
        assert deserialized["token_usage"] is None


# =============================================================================
# Inheritance Tests
# =============================================================================


class TestInheritance:
    """Tests verifying all event types inherit from TelemetryEvent."""

    @pytest.mark.parametrize(
        "event_class",
        [
            RunStarted,
            RunFinished,
            PhaseChanged,
            RowCreated,
            TransformCompleted,
            GateEvaluated,
            TokenCompleted,
            ExternalCallCompleted,
        ],
    )
    def test_inherits_from_telemetry_event(self, event_class: type) -> None:
        """All event classes inherit from TelemetryEvent."""
        assert issubclass(event_class, TelemetryEvent)

    @pytest.mark.parametrize(
        "event_class",
        [
            TelemetryEvent,
            RunStarted,
            RunFinished,
            PhaseChanged,
            RowCreated,
            TransformCompleted,
            GateEvaluated,
            TokenCompleted,
            ExternalCallCompleted,
        ],
    )
    def test_all_frozen(self, event_class: type) -> None:
        """All event classes are frozen dataclasses."""
        # __dataclass_fields__ exists for dataclasses
        assert hasattr(event_class, "__dataclass_fields__")
        # frozen dataclasses have this attribute
        assert event_class.__dataclass_params__.frozen  # type: ignore[attr-defined]

    @pytest.mark.parametrize(
        "event_class",
        [
            TelemetryEvent,
            RunStarted,
            RunFinished,
            PhaseChanged,
            RowCreated,
            TransformCompleted,
            GateEvaluated,
            TokenCompleted,
            ExternalCallCompleted,
        ],
    )
    def test_all_have_slots(self, event_class: type) -> None:
        """All event classes use slots for memory efficiency."""
        assert hasattr(event_class, "__slots__")
