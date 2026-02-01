# tests/engine/test_executors.py
"""Dedicated tests for engine executors module.

This file provides unit tests for the executor classes that were previously
only tested indirectly via integration tests. Each executor wraps plugin
calls with audit recording and error handling.

Test Coverage:
- MissingEdgeError: Exception for audit integrity violations
- GateOutcome: Dataclass for gate execution results
- TransformExecutor: Transform execution with audit recording
- GateExecutor: Gate evaluation with routing
- AggregationExecutor: Batch buffering and flush execution
- SinkExecutor: Sink writes with artifact recording

Note: TransformExecutor has significant existing coverage in
test_transform_error_routing.py. These tests fill gaps for edge cases.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import pytest

from elspeth.contracts import (
    ArtifactDescriptor,
    PendingOutcome,
    RoutingAction,
    RowOutcome,
    TokenInfo,
)
from elspeth.contracts.enums import Determinism, NodeType, RoutingKind, RoutingMode, TriggerType
from elspeth.contracts.errors import OrchestrationInvariantError
from elspeth.contracts.schema import SchemaConfig
from elspeth.contracts.types import NodeID
from elspeth.core.config import AggregationSettings, TriggerConfig
from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
from elspeth.engine.executors import (
    AggregationExecutor,
    GateExecutor,
    GateOutcome,
    MissingEdgeError,
    SinkExecutor,
    TransformExecutor,
)
from elspeth.engine.spans import SpanFactory
from elspeth.plugins.context import PluginContext
from elspeth.plugins.results import GateResult, TransformResult
from tests.conftest import (
    _TestSchema,
    _TestSinkBase,
    _TestTransformBase,
    as_sink,
    as_transform,
)

if TYPE_CHECKING:
    pass

# Dynamic schema for tests that don't care about specific fields
DYNAMIC_SCHEMA = SchemaConfig.from_dict({"fields": "dynamic"})


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def landscape_setup() -> tuple[LandscapeDB, LandscapeRecorder, Any]:
    """Set up LandscapeDB, recorder, and run for tests."""
    db = LandscapeDB.in_memory()
    recorder = LandscapeRecorder(db)
    run = recorder.begin_run(config={}, canonical_version="v1")
    return db, recorder, run


@pytest.fixture
def span_factory() -> SpanFactory:
    """Create a span factory for tests."""
    return SpanFactory()


# =============================================================================
# MissingEdgeError Tests
# =============================================================================


class TestMissingEdgeError:
    """Tests for MissingEdgeError exception class."""

    def test_stores_node_id_and_label(self) -> None:
        """MissingEdgeError stores the node_id and label that caused it."""
        node_id = NodeID("gate_123")
        label = "path_a"

        error = MissingEdgeError(node_id=node_id, label=label)

        assert error.node_id == node_id
        assert error.label == label

    def test_message_includes_node_and_label(self) -> None:
        """Error message includes both node_id and label for debugging."""
        node_id = NodeID("my_gate")
        label = "unknown_route"

        error = MissingEdgeError(node_id=node_id, label=label)

        assert "my_gate" in str(error)
        assert "unknown_route" in str(error)
        assert "No edge registered" in str(error)

    def test_is_exception_subclass(self) -> None:
        """MissingEdgeError is a proper Exception subclass."""
        error = MissingEdgeError(node_id=NodeID("x"), label="y")

        assert isinstance(error, Exception)

    def test_can_be_raised_and_caught(self) -> None:
        """MissingEdgeError can be raised and caught like any exception."""
        node_id = NodeID("gate_1")
        label = "missing"

        with pytest.raises(MissingEdgeError) as exc_info:
            raise MissingEdgeError(node_id=node_id, label=label)

        assert exc_info.value.node_id == node_id
        assert exc_info.value.label == label


# =============================================================================
# GateOutcome Tests
# =============================================================================


class TestGateOutcome:
    """Tests for GateOutcome dataclass."""

    def test_basic_construction(self) -> None:
        """GateOutcome can be constructed with required fields."""
        token = TokenInfo(
            row_id="row-1",
            token_id="tok-1",
            row_data={"x": 1},
        )
        result = GateResult(
            row={"x": 1},
            action=RoutingAction.continue_(),
        )

        outcome = GateOutcome(
            result=result,
            updated_token=token,
        )

        assert outcome.result is result
        assert outcome.updated_token is token
        assert outcome.child_tokens == []
        assert outcome.sink_name is None

    def test_with_child_tokens_for_fork(self) -> None:
        """GateOutcome stores child tokens for fork operations."""
        parent_token = TokenInfo(
            row_id="row-1",
            token_id="parent",
            row_data={"x": 1},
        )
        child_a = TokenInfo(
            row_id="row-1",
            token_id="child-a",
            row_data={"x": 1},
            branch_name="path_a",
        )
        child_b = TokenInfo(
            row_id="row-1",
            token_id="child-b",
            row_data={"x": 1},
            branch_name="path_b",
        )

        result = GateResult(
            row={"x": 1},
            action=RoutingAction.fork_to_paths(["path_a", "path_b"]),
        )

        outcome = GateOutcome(
            result=result,
            updated_token=parent_token,
            child_tokens=[child_a, child_b],
        )

        assert len(outcome.child_tokens) == 2
        assert outcome.child_tokens[0].branch_name == "path_a"
        assert outcome.child_tokens[1].branch_name == "path_b"

    def test_with_sink_name_for_route(self) -> None:
        """GateOutcome stores sink_name when routing to a sink."""
        token = TokenInfo(
            row_id="row-1",
            token_id="tok-1",
            row_data={"x": 1},
        )
        result = GateResult(
            row={"x": 1},
            action=RoutingAction.route("error_sink"),
        )

        outcome = GateOutcome(
            result=result,
            updated_token=token,
            sink_name="error_sink",
        )

        assert outcome.sink_name == "error_sink"


# =============================================================================
# TransformExecutor Tests
# =============================================================================


class MockTransform(_TestTransformBase):
    """Mock transform for executor tests."""

    name = "mock_transform"

    def __init__(
        self,
        result: TransformResult | None = None,
        on_error: str | None = None,
        raises: Exception | None = None,
    ) -> None:
        self._result = result or TransformResult.success({"transformed": True}, success_reason={"action": "test"})
        self._on_error = on_error
        self._raises = raises

    def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
        if self._raises is not None:
            raise self._raises
        return self._result


class TestTransformExecutor:
    """Tests for TransformExecutor class."""

    def test_execute_success_updates_token(
        self,
        landscape_setup: tuple[LandscapeDB, LandscapeRecorder, Any],
        span_factory: SpanFactory,
    ) -> None:
        """Successful transform execution updates the token with new row data."""
        _db, recorder, run = landscape_setup

        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="mock_transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        transform = MockTransform(TransformResult.success({"output": 42}, success_reason={"action": "test"}))
        transform.node_id = node.node_id

        ctx = PluginContext(
            run_id=run.run_id,
            config={},
            node_id=node.node_id,
            landscape=recorder,
        )

        token = TokenInfo(
            row_id="row-1",
            token_id="tok-1",
            row_data={"input": 1},
        )

        # Create row/token in landscape
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=node.node_id,
            row_index=0,
            data=token.row_data,
            row_id=token.row_id,
        )
        recorder.create_token(row_id=row.row_id, token_id=token.token_id)

        executor = TransformExecutor(recorder, span_factory)
        result, updated_token, error_sink = executor.execute_transform(
            transform=as_transform(transform),
            token=token,
            ctx=ctx,
            step_in_pipeline=1,
        )

        assert result.status == "success"
        assert updated_token.row_data == {"output": 42}
        assert error_sink is None

    def test_execute_populates_audit_fields(
        self,
        landscape_setup: tuple[LandscapeDB, LandscapeRecorder, Any],
        span_factory: SpanFactory,
    ) -> None:
        """Transform execution populates input_hash, output_hash, and duration_ms."""
        _db, recorder, run = landscape_setup

        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="mock_transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        transform = MockTransform(TransformResult.success({"value": 1}, success_reason={"action": "test"}))
        transform.node_id = node.node_id

        ctx = PluginContext(
            run_id=run.run_id,
            config={},
            node_id=node.node_id,
            landscape=recorder,
        )

        token = TokenInfo(
            row_id="row-1",
            token_id="tok-1",
            row_data={"input": 1},
        )

        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=node.node_id,
            row_index=0,
            data=token.row_data,
            row_id=token.row_id,
        )
        recorder.create_token(row_id=row.row_id, token_id=token.token_id)

        executor = TransformExecutor(recorder, span_factory)
        result, _, _ = executor.execute_transform(
            transform=as_transform(transform),
            token=token,
            ctx=ctx,
            step_in_pipeline=1,
        )

        assert result.input_hash is not None
        assert result.output_hash is not None
        assert result.duration_ms is not None
        assert result.duration_ms >= 0

    def test_error_without_on_error_raises_runtime_error(
        self,
        landscape_setup: tuple[LandscapeDB, LandscapeRecorder, Any],
        span_factory: SpanFactory,
    ) -> None:
        """Transform returning error without on_error config raises RuntimeError."""
        _db, recorder, run = landscape_setup

        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="mock_transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        transform = MockTransform(
            result=TransformResult.error({"reason": "validation_failed", "error": "bad data"}),
            on_error=None,  # No error handler configured
        )
        transform.node_id = node.node_id

        ctx = PluginContext(
            run_id=run.run_id,
            config={},
            node_id=node.node_id,
            landscape=recorder,
        )

        token = TokenInfo(
            row_id="row-1",
            token_id="tok-1",
            row_data={"input": 1},
        )

        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=node.node_id,
            row_index=0,
            data=token.row_data,
            row_id=token.row_id,
        )
        recorder.create_token(row_id=row.row_id, token_id=token.token_id)

        executor = TransformExecutor(recorder, span_factory)

        with pytest.raises(RuntimeError, match="no on_error configured"):
            executor.execute_transform(
                transform=as_transform(transform),
                token=token,
                ctx=ctx,
                step_in_pipeline=1,
            )

    def test_requires_node_id_set(
        self,
        landscape_setup: tuple[LandscapeDB, LandscapeRecorder, Any],
        span_factory: SpanFactory,
    ) -> None:
        """Transform without node_id raises assertion error."""
        _db, recorder, run = landscape_setup

        transform = MockTransform()
        transform.node_id = None  # Not set

        ctx = PluginContext(
            run_id=run.run_id,
            config={},
            node_id="irrelevant",
            landscape=recorder,
        )

        token = TokenInfo(
            row_id="row-1",
            token_id="tok-1",
            row_data={"input": 1},
        )

        executor = TransformExecutor(recorder, span_factory)

        with pytest.raises(OrchestrationInvariantError, match="executed without node_id"):
            executor.execute_transform(
                transform=as_transform(transform),
                token=token,
                ctx=ctx,
                step_in_pipeline=1,
            )


# =============================================================================
# GateExecutor Tests
# =============================================================================


class MockGate:
    """Mock gate for executor tests."""

    name = "mock_gate"
    node_id: str | None = None
    config: ClassVar[dict[str, Any]] = {"schema": {"fields": "dynamic"}}
    input_schema = _TestSchema
    output_schema = _TestSchema
    determinism = Determinism.DETERMINISTIC
    plugin_version = "1.0.0"

    def __init__(
        self,
        action: RoutingAction | None = None,
        raises: Exception | None = None,
    ) -> None:
        self._action = action or RoutingAction.continue_()
        self._raises = raises

    def evaluate(self, row: dict[str, Any], ctx: PluginContext) -> GateResult:
        if self._raises is not None:
            raise self._raises
        return GateResult(row=row, action=self._action)

    def on_start(self, ctx: Any) -> None:
        pass

    def on_complete(self, ctx: Any) -> None:
        pass

    def close(self) -> None:
        pass


class TestGateExecutor:
    """Tests for GateExecutor class."""

    def test_execute_continue_action(
        self,
        landscape_setup: tuple[LandscapeDB, LandscapeRecorder, Any],
        span_factory: SpanFactory,
    ) -> None:
        """Gate returning CONTINUE action updates token and records routing."""
        _db, recorder, run = landscape_setup

        # Register gate node
        gate_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="mock_gate",
            node_type=NodeType.GATE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # Register destination node for "continue" edge
        next_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="next_transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # Register edge in database (required for FK constraint)
        edge = recorder.register_edge(
            run_id=run.run_id,
            from_node_id=gate_node.node_id,
            to_node_id=next_node.node_id,
            label="continue",
            mode=RoutingMode.MOVE,
        )

        gate = MockGate(action=RoutingAction.continue_())
        gate.node_id = gate_node.node_id

        ctx = PluginContext(
            run_id=run.run_id,
            config={},
            node_id=gate_node.node_id,
            landscape=recorder,
        )

        token = TokenInfo(
            row_id="row-1",
            token_id="tok-1",
            row_data={"x": 1},
        )

        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=gate_node.node_id,
            row_index=0,
            data=token.row_data,
            row_id=token.row_id,
        )
        recorder.create_token(row_id=row.row_id, token_id=token.token_id)

        # Create edge map using actual edge_id from database
        edge_map = {(NodeID(gate_node.node_id), "continue"): edge.edge_id}

        executor = GateExecutor(recorder, span_factory, edge_map=edge_map)
        outcome = executor.execute_gate(
            gate=gate,  # type: ignore[arg-type]
            token=token,
            ctx=ctx,
            step_in_pipeline=1,
        )

        assert outcome.result.action.kind == RoutingKind.CONTINUE
        assert outcome.sink_name is None
        assert outcome.child_tokens == []

    def test_execute_route_to_sink(
        self,
        landscape_setup: tuple[LandscapeDB, LandscapeRecorder, Any],
        span_factory: SpanFactory,
    ) -> None:
        """Gate routing to a sink sets sink_name in outcome."""
        _db, recorder, run = landscape_setup

        # Register gate node
        gate_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="mock_gate",
            node_type=NodeType.GATE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # Register error sink node
        error_sink_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="error_sink",
            node_type=NodeType.SINK,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # Register edge in database (required for FK constraint)
        edge = recorder.register_edge(
            run_id=run.run_id,
            from_node_id=gate_node.node_id,
            to_node_id=error_sink_node.node_id,
            label="error_sink",
            mode=RoutingMode.MOVE,
        )

        gate = MockGate(action=RoutingAction.route("error_sink"))
        gate.node_id = gate_node.node_id

        ctx = PluginContext(
            run_id=run.run_id,
            config={},
            node_id=gate_node.node_id,
            landscape=recorder,
        )

        token = TokenInfo(
            row_id="row-1",
            token_id="tok-1",
            row_data={"x": 1},
        )

        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=gate_node.node_id,
            row_index=0,
            data=token.row_data,
            row_id=token.row_id,
        )
        recorder.create_token(row_id=row.row_id, token_id=token.token_id)

        # Edge and route resolution maps using actual edge_id
        edge_map = {(NodeID(gate_node.node_id), "error_sink"): edge.edge_id}
        route_resolution = {(NodeID(gate_node.node_id), "error_sink"): "error_sink"}

        executor = GateExecutor(
            recorder,
            span_factory,
            edge_map=edge_map,
            route_resolution_map=route_resolution,
        )
        outcome = executor.execute_gate(
            gate=gate,  # type: ignore[arg-type]
            token=token,
            ctx=ctx,
            step_in_pipeline=1,
        )

        assert outcome.sink_name == "error_sink"

    def test_execute_missing_edge_raises_error(
        self,
        landscape_setup: tuple[LandscapeDB, LandscapeRecorder, Any],
        span_factory: SpanFactory,
    ) -> None:
        """Gate routing to unregistered edge raises MissingEdgeError."""
        _db, recorder, run = landscape_setup

        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="mock_gate",
            node_type=NodeType.GATE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        gate = MockGate(action=RoutingAction.continue_())
        gate.node_id = node.node_id

        ctx = PluginContext(
            run_id=run.run_id,
            config={},
            node_id=node.node_id,
            landscape=recorder,
        )

        token = TokenInfo(
            row_id="row-1",
            token_id="tok-1",
            row_data={"x": 1},
        )

        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=node.node_id,
            row_index=0,
            data=token.row_data,
            row_id=token.row_id,
        )
        recorder.create_token(row_id=row.row_id, token_id=token.token_id)

        # Empty edge map - "continue" edge not registered
        executor = GateExecutor(recorder, span_factory, edge_map={})

        with pytest.raises(MissingEdgeError) as exc_info:
            executor.execute_gate(
                gate=gate,  # type: ignore[arg-type]
                token=token,
                ctx=ctx,
                step_in_pipeline=1,
            )

        assert exc_info.value.label == "continue"

    def test_requires_node_id_set(
        self,
        landscape_setup: tuple[LandscapeDB, LandscapeRecorder, Any],
        span_factory: SpanFactory,
    ) -> None:
        """Gate without node_id raises assertion error."""
        _db, recorder, run = landscape_setup

        gate = MockGate()
        gate.node_id = None

        ctx = PluginContext(
            run_id=run.run_id,
            config={},
            node_id="irrelevant",
            landscape=recorder,
        )

        token = TokenInfo(
            row_id="row-1",
            token_id="tok-1",
            row_data={"x": 1},
        )

        executor = GateExecutor(recorder, span_factory)

        with pytest.raises(OrchestrationInvariantError, match="executed without node_id"):
            executor.execute_gate(
                gate=gate,  # type: ignore[arg-type]
                token=token,
                ctx=ctx,
                step_in_pipeline=1,
            )


# =============================================================================
# AggregationExecutor Tests
# =============================================================================


class TestAggregationExecutor:
    """Tests for AggregationExecutor class."""

    def _register_agg_node(self, recorder: LandscapeRecorder, run_id: str) -> tuple[NodeID, AggregationSettings]:
        """Helper to register an aggregation node and return settings."""
        node = recorder.register_node(
            run_id=run_id,
            plugin_name="batch_stats",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        node_id = NodeID(node.node_id)
        trigger = TriggerConfig(count=5)
        settings = AggregationSettings(name="test_agg", plugin="batch_stats", trigger=trigger)
        return node_id, settings

    def test_buffer_row_increments_count(
        self,
        landscape_setup: tuple[LandscapeDB, LandscapeRecorder, Any],
        span_factory: SpanFactory,
    ) -> None:
        """Buffering rows increments the buffer count."""
        _db, recorder, run = landscape_setup

        # Register aggregation node in database
        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="batch_stats",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        node_id = NodeID(node.node_id)
        trigger = TriggerConfig(count=10)
        settings = {node_id: AggregationSettings(name="test_agg", plugin="batch_stats", trigger=trigger)}

        executor = AggregationExecutor(
            recorder,
            span_factory,
            run_id=run.run_id,
            aggregation_settings=settings,
        )

        # Create rows and tokens in landscape (required for batch_members FK)
        for i in range(3):
            row = recorder.create_row(
                run_id=run.run_id,
                source_node_id=node.node_id,
                row_index=i,
                data={"value": i},
                row_id=f"row-{i}",
            )
            recorder.create_token(row_id=row.row_id, token_id=f"tok-{i}")

        for i in range(3):
            token = TokenInfo(
                row_id=f"row-{i}",
                token_id=f"tok-{i}",
                row_data={"value": i},
            )
            executor.buffer_row(node_id, token)

        assert executor.get_buffer_count(node_id) == 3

    def test_should_flush_when_count_reached(
        self,
        landscape_setup: tuple[LandscapeDB, LandscapeRecorder, Any],
        span_factory: SpanFactory,
    ) -> None:
        """should_flush returns True when count trigger is reached."""
        _db, recorder, run = landscape_setup

        # Register aggregation node in database
        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="batch_stats",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        node_id = NodeID(node.node_id)
        trigger = TriggerConfig(count=3)
        settings = {node_id: AggregationSettings(name="test_agg", plugin="batch_stats", trigger=trigger)}

        executor = AggregationExecutor(
            recorder,
            span_factory,
            run_id=run.run_id,
            aggregation_settings=settings,
        )

        # Create rows and tokens in landscape (required for batch_members FK)
        for i in range(3):
            row = recorder.create_row(
                run_id=run.run_id,
                source_node_id=node.node_id,
                row_index=i,
                data={"value": i},
                row_id=f"row-{i}",
            )
            recorder.create_token(row_id=row.row_id, token_id=f"tok-{i}")

        # Buffer 2 rows - not enough
        for i in range(2):
            token = TokenInfo(
                row_id=f"row-{i}",
                token_id=f"tok-{i}",
                row_data={"value": i},
            )
            executor.buffer_row(node_id, token)

        assert not executor.should_flush(node_id)

        # Buffer 1 more - now should flush
        token = TokenInfo(
            row_id="row-2",
            token_id="tok-2",
            row_data={"value": 2},
        )
        executor.buffer_row(node_id, token)

        assert executor.should_flush(node_id)

    def test_get_trigger_type_returns_count(
        self,
        landscape_setup: tuple[LandscapeDB, LandscapeRecorder, Any],
        span_factory: SpanFactory,
    ) -> None:
        """get_trigger_type returns COUNT when count trigger fires."""
        _db, recorder, run = landscape_setup

        # Register aggregation node in database
        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="batch_stats",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        node_id = NodeID(node.node_id)
        trigger = TriggerConfig(count=2)
        settings = {node_id: AggregationSettings(name="test_agg", plugin="batch_stats", trigger=trigger)}

        executor = AggregationExecutor(
            recorder,
            span_factory,
            run_id=run.run_id,
            aggregation_settings=settings,
        )

        # Create rows and tokens in landscape (required for batch_members FK)
        for i in range(2):
            row = recorder.create_row(
                run_id=run.run_id,
                source_node_id=node.node_id,
                row_index=i,
                data={"value": i},
                row_id=f"row-{i}",
            )
            recorder.create_token(row_id=row.row_id, token_id=f"tok-{i}")

        for i in range(2):
            token = TokenInfo(
                row_id=f"row-{i}",
                token_id=f"tok-{i}",
                row_data={"value": i},
            )
            executor.buffer_row(node_id, token)

        # should_flush() must be called first - it sets _last_triggered as a side effect
        assert executor.should_flush(node_id) is True
        assert executor.get_trigger_type(node_id) == TriggerType.COUNT

    def test_get_buffered_rows_returns_copy(
        self,
        landscape_setup: tuple[LandscapeDB, LandscapeRecorder, Any],
        span_factory: SpanFactory,
    ) -> None:
        """get_buffered_rows returns a copy, not the internal buffer."""
        _db, recorder, run = landscape_setup

        # Register aggregation node in database
        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="batch_stats",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        node_id = NodeID(node.node_id)
        trigger = TriggerConfig(count=10)
        settings = {node_id: AggregationSettings(name="test_agg", plugin="batch_stats", trigger=trigger)}

        executor = AggregationExecutor(
            recorder,
            span_factory,
            run_id=run.run_id,
            aggregation_settings=settings,
        )

        # Create row and token in landscape (required for batch_members FK)
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=node.node_id,
            row_index=0,
            data={"value": 1},
            row_id="row-1",
        )
        recorder.create_token(row_id=row.row_id, token_id="tok-1")

        token = TokenInfo(
            row_id="row-1",
            token_id="tok-1",
            row_data={"value": 1},
        )
        executor.buffer_row(node_id, token)

        rows = executor.get_buffered_rows(node_id)
        rows.append({"extra": "row"})  # Modify the returned list

        # Original buffer unchanged
        assert executor.get_buffer_count(node_id) == 1

    def test_checkpoint_state_roundtrip(
        self,
        landscape_setup: tuple[LandscapeDB, LandscapeRecorder, Any],
        span_factory: SpanFactory,
    ) -> None:
        """Checkpoint state can be saved and restored."""
        _db, recorder, run = landscape_setup

        # Register aggregation node in database
        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="batch_stats",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        node_id = NodeID(node.node_id)
        trigger = TriggerConfig(count=10)
        settings = {node_id: AggregationSettings(name="test_agg", plugin="batch_stats", trigger=trigger)}

        executor = AggregationExecutor(
            recorder,
            span_factory,
            run_id=run.run_id,
            aggregation_settings=settings,
        )

        # Create rows and tokens in landscape (required for batch_members FK)
        for i in range(3):
            row = recorder.create_row(
                run_id=run.run_id,
                source_node_id=node.node_id,
                row_index=i,
                data={"value": i},
                row_id=f"row-{i}",
            )
            recorder.create_token(row_id=row.row_id, token_id=f"tok-{i}")

        # Buffer some rows
        for i in range(3):
            token = TokenInfo(
                row_id=f"row-{i}",
                token_id=f"tok-{i}",
                row_data={"value": i},
                branch_name="main" if i == 0 else None,
            )
            executor.buffer_row(node_id, token)

        # Get checkpoint state
        state = executor.get_checkpoint_state()

        # Create new executor and restore
        new_executor = AggregationExecutor(
            recorder,
            span_factory,
            run_id=run.run_id,
            aggregation_settings=settings,
        )
        new_executor.restore_from_checkpoint(state)

        # Verify state restored
        assert new_executor.get_buffer_count(node_id) == 3
        restored_rows = new_executor.get_buffered_rows(node_id)
        assert restored_rows[0] == {"value": 0}
        assert restored_rows[1] == {"value": 1}
        assert restored_rows[2] == {"value": 2}

    def test_checkpoint_version_mismatch_raises(
        self,
        landscape_setup: tuple[LandscapeDB, LandscapeRecorder, Any],
        span_factory: SpanFactory,
    ) -> None:
        """Restoring checkpoint with wrong version raises ValueError."""
        _db, recorder, run = landscape_setup

        node_id = NodeID("agg_node")
        trigger = TriggerConfig(count=10)
        settings = {node_id: AggregationSettings(name="test_agg", plugin="batch_stats", trigger=trigger)}

        executor = AggregationExecutor(
            recorder,
            span_factory,
            run_id=run.run_id,
            aggregation_settings=settings,
        )

        # Checkpoint with wrong version
        bad_state = {"_version": "999.0"}

        with pytest.raises(ValueError, match="Incompatible checkpoint version"):
            executor.restore_from_checkpoint(bad_state)


# =============================================================================
# SinkExecutor Tests
# =============================================================================


class MockSink(_TestSinkBase):
    """Mock sink for executor tests."""

    name = "mock_sink"

    def __init__(
        self,
        raises: Exception | None = None,
    ) -> None:
        super().__init__()
        self.written_rows: list[dict[str, Any]] = []
        self._raises = raises

    def write(self, rows: list[dict[str, Any]], ctx: PluginContext) -> ArtifactDescriptor:
        if self._raises is not None:
            raise self._raises
        self.written_rows.extend(rows)
        return ArtifactDescriptor.for_file(
            path="/tmp/output.csv",
            size_bytes=100,
            content_hash="abc123",
        )


class TestSinkExecutor:
    """Tests for SinkExecutor class."""

    def test_write_creates_artifact(
        self,
        landscape_setup: tuple[LandscapeDB, LandscapeRecorder, Any],
        span_factory: SpanFactory,
    ) -> None:
        """Successful write returns an artifact with content hash."""
        _db, recorder, run = landscape_setup

        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="mock_sink",
            node_type=NodeType.SINK,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        sink = MockSink()
        sink.node_id = node.node_id

        ctx = PluginContext(
            run_id=run.run_id,
            config={},
            node_id=node.node_id,
            landscape=recorder,
        )

        tokens = [
            TokenInfo(row_id="row-1", token_id="tok-1", row_data={"x": 1}),
            TokenInfo(row_id="row-2", token_id="tok-2", row_data={"x": 2}),
        ]

        # Create rows and tokens in landscape
        for i, token in enumerate(tokens):
            row = recorder.create_row(
                run_id=run.run_id,
                source_node_id=node.node_id,
                row_index=i,
                data=token.row_data,
                row_id=token.row_id,
            )
            recorder.create_token(row_id=row.row_id, token_id=token.token_id)

        executor = SinkExecutor(recorder, span_factory, run.run_id)
        artifact = executor.write(
            sink=as_sink(sink),
            tokens=tokens,
            ctx=ctx,
            step_in_pipeline=5,
            sink_name="mock_sink",
            pending_outcome=PendingOutcome(RowOutcome.COMPLETED),
        )

        assert artifact is not None
        assert artifact.content_hash == "abc123"
        assert artifact.path_or_uri == "file:///tmp/output.csv"

    def test_write_exception_records_failure_for_all_tokens(
        self,
        landscape_setup: tuple[LandscapeDB, LandscapeRecorder, Any],
        span_factory: SpanFactory,
    ) -> None:
        """Sink exception marks all token states as failed."""
        _db, recorder, run = landscape_setup

        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="mock_sink",
            node_type=NodeType.SINK,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        sink = MockSink(raises=OSError("disk full"))
        sink.node_id = node.node_id

        ctx = PluginContext(
            run_id=run.run_id,
            config={},
            node_id=node.node_id,
            landscape=recorder,
        )

        tokens = [
            TokenInfo(row_id="row-1", token_id="tok-1", row_data={"x": 1}),
            TokenInfo(row_id="row-2", token_id="tok-2", row_data={"x": 2}),
        ]

        for i, token in enumerate(tokens):
            row = recorder.create_row(
                run_id=run.run_id,
                source_node_id=node.node_id,
                row_index=i,
                data=token.row_data,
                row_id=token.row_id,
            )
            recorder.create_token(row_id=row.row_id, token_id=token.token_id)

        executor = SinkExecutor(recorder, span_factory, run.run_id)

        with pytest.raises(OSError, match="disk full"):
            executor.write(
                sink=as_sink(sink),
                tokens=tokens,
                ctx=ctx,
                step_in_pipeline=5,
                sink_name="mock_sink",
                pending_outcome=PendingOutcome(RowOutcome.COMPLETED),
            )

    def test_requires_sink_node_id_set(
        self,
        landscape_setup: tuple[LandscapeDB, LandscapeRecorder, Any],
        span_factory: SpanFactory,
    ) -> None:
        """Sink without node_id raises assertion error."""
        _db, recorder, run = landscape_setup

        sink = MockSink()
        sink.node_id = None

        ctx = PluginContext(
            run_id=run.run_id,
            config={},
            node_id="irrelevant",
            landscape=recorder,
        )

        tokens = [
            TokenInfo(row_id="row-1", token_id="tok-1", row_data={"x": 1}),
        ]

        executor = SinkExecutor(recorder, span_factory, run.run_id)

        with pytest.raises(OrchestrationInvariantError, match="executed without node_id"):
            executor.write(
                sink=as_sink(sink),
                tokens=tokens,
                ctx=ctx,
                step_in_pipeline=5,
                sink_name="mock_sink",
                pending_outcome=PendingOutcome(RowOutcome.COMPLETED),
            )

    def test_on_token_written_callback_called(
        self,
        landscape_setup: tuple[LandscapeDB, LandscapeRecorder, Any],
        span_factory: SpanFactory,
    ) -> None:
        """on_token_written callback is called for each token after write."""
        _db, recorder, run = landscape_setup

        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="mock_sink",
            node_type=NodeType.SINK,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        sink = MockSink()
        sink.node_id = node.node_id

        ctx = PluginContext(
            run_id=run.run_id,
            config={},
            node_id=node.node_id,
            landscape=recorder,
        )

        tokens = [
            TokenInfo(row_id="row-1", token_id="tok-1", row_data={"x": 1}),
            TokenInfo(row_id="row-2", token_id="tok-2", row_data={"x": 2}),
        ]

        for i, token in enumerate(tokens):
            row = recorder.create_row(
                run_id=run.run_id,
                source_node_id=node.node_id,
                row_index=i,
                data=token.row_data,
                row_id=token.row_id,
            )
            recorder.create_token(row_id=row.row_id, token_id=token.token_id)

        callback_tokens: list[TokenInfo] = []

        def on_written(token: TokenInfo) -> None:
            callback_tokens.append(token)

        executor = SinkExecutor(recorder, span_factory, run.run_id)
        executor.write(
            sink=as_sink(sink),
            tokens=tokens,
            ctx=ctx,
            step_in_pipeline=5,
            sink_name="mock_sink",
            pending_outcome=PendingOutcome(RowOutcome.COMPLETED),
            on_token_written=on_written,
        )

        assert len(callback_tokens) == 2
        assert callback_tokens[0].token_id == "tok-1"
        assert callback_tokens[1].token_id == "tok-2"


# =============================================================================
# Transform Output Canonical Validation Tests
# =============================================================================


class TestTransformCanonicalValidation:
    """Tests for canonical output validation at transform boundary.

    Per CLAUDE.md: plugin bugs must crash. Transforms that emit non-canonical
    data (NaN, Infinity, non-serializable types) have a bug that must be fixed.

    Bug: P3-2026-01-29-transform-output-canonical-validation
    """

    def test_transform_emitting_nan_raises_contract_violation(
        self,
        landscape_setup: tuple[LandscapeDB, LandscapeRecorder, Any],
        span_factory: SpanFactory,
    ) -> None:
        """Transform returning NaN in output raises PluginContractViolation."""
        from math import nan

        from elspeth.contracts.errors import PluginContractViolation

        _db, recorder, run = landscape_setup

        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="nan_transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        class NaNTransform(_TestTransformBase):
            """Transform that returns NaN - violates canonical contract."""

            name: ClassVar[str] = "nan_transform"

            def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
                return TransformResult.success(
                    {"value": nan},
                    success_reason={"action": "added_nan"},
                )

        transform = NaNTransform()
        transform.node_id = node.node_id

        ctx = PluginContext(
            run_id=run.run_id,
            config={},
            node_id=node.node_id,
            landscape=recorder,
        )

        token = TokenInfo(
            row_id="row-1",
            token_id="tok-1",
            row_data={"x": 1},
        )

        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=node.node_id,
            row_index=0,
            data=token.row_data,
            row_id=token.row_id,
        )
        recorder.create_token(row_id=row.row_id, token_id=token.token_id)

        executor = TransformExecutor(recorder, span_factory, run.run_id)

        with pytest.raises(PluginContractViolation, match="non-canonical data"):
            executor.execute_transform(
                transform=as_transform(transform),
                token=token,
                ctx=ctx,
                step_in_pipeline=1,
            )

    def test_transform_emitting_infinity_raises_contract_violation(
        self,
        landscape_setup: tuple[LandscapeDB, LandscapeRecorder, Any],
        span_factory: SpanFactory,
    ) -> None:
        """Transform returning Infinity in output raises PluginContractViolation."""
        from math import inf

        from elspeth.contracts.errors import PluginContractViolation

        _db, recorder, run = landscape_setup

        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="inf_transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        class InfTransform(_TestTransformBase):
            """Transform that returns Infinity - violates canonical contract."""

            name: ClassVar[str] = "inf_transform"

            def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
                return TransformResult.success(
                    {"value": inf},
                    success_reason={"action": "added_inf"},
                )

        transform = InfTransform()
        transform.node_id = node.node_id

        ctx = PluginContext(
            run_id=run.run_id,
            config={},
            node_id=node.node_id,
            landscape=recorder,
        )

        token = TokenInfo(
            row_id="row-1",
            token_id="tok-1",
            row_data={"x": 1},
        )

        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=node.node_id,
            row_index=0,
            data=token.row_data,
            row_id=token.row_id,
        )
        recorder.create_token(row_id=row.row_id, token_id=token.token_id)

        executor = TransformExecutor(recorder, span_factory, run.run_id)

        with pytest.raises(PluginContractViolation, match="non-canonical data"):
            executor.execute_transform(
                transform=as_transform(transform),
                token=token,
                ctx=ctx,
                step_in_pipeline=1,
            )

    def test_transform_emitting_valid_data_passes(
        self,
        landscape_setup: tuple[LandscapeDB, LandscapeRecorder, Any],
        span_factory: SpanFactory,
    ) -> None:
        """Transform returning valid JSON-serializable data succeeds."""
        _db, recorder, run = landscape_setup

        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="valid_transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        class ValidTransform(_TestTransformBase):
            """Transform that returns valid canonical data."""

            name: ClassVar[str] = "valid_transform"

            def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
                return TransformResult.success(
                    {
                        "string": "hello",
                        "int": 42,
                        "float": 3.14,
                        "bool": True,
                        "null": None,
                        "list": [1, 2, 3],
                        "nested": {"a": 1, "b": 2},
                    },
                    success_reason={"action": "transformed"},
                )

        transform = ValidTransform()
        transform.node_id = node.node_id

        ctx = PluginContext(
            run_id=run.run_id,
            config={},
            node_id=node.node_id,
            landscape=recorder,
        )

        token = TokenInfo(
            row_id="row-1",
            token_id="tok-1",
            row_data={"x": 1},
        )

        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=node.node_id,
            row_index=0,
            data=token.row_data,
            row_id=token.row_id,
        )
        recorder.create_token(row_id=row.row_id, token_id=token.token_id)

        executor = TransformExecutor(recorder, span_factory, run.run_id)

        # Should not raise - returns (result, updated_token, error_sink)
        result, updated_token, _error_sink = executor.execute_transform(
            transform=as_transform(transform),
            token=token,
            ctx=ctx,
            step_in_pipeline=1,
        )

        assert result.status == "success"
        assert updated_token.row_data["string"] == "hello"
        assert updated_token.row_data["int"] == 42
