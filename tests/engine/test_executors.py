# tests/engine/test_executors.py
"""Tests for plugin executors."""

from typing import Any

import pytest

from elspeth.contracts import RoutingMode
from elspeth.contracts.schema import SchemaConfig
from tests.conftest import as_gate, as_sink, as_transform

# Dynamic schema for tests that don't care about specific fields
DYNAMIC_SCHEMA = SchemaConfig.from_dict({"fields": "dynamic"})


class TestTransformExecutor:
    """Transform execution with audit."""

    def test_execute_transform_success(self) -> None:
        from elspeth.contracts import TokenInfo
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import TransformExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import TransformResult

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="double",
            node_type="transform",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # Mock transform plugin
        class DoubleTransform:
            name = "double"
            node_id = node.node_id

            def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
                return TransformResult.success({"value": row["value"] * 2})

        transform = DoubleTransform()
        ctx = PluginContext(run_id=run.run_id, config={})
        executor = TransformExecutor(recorder, SpanFactory())

        token = TokenInfo(
            row_id="row-1",
            token_id="token-1",
            row_data={"value": 21},
        )

        # Need to create row/token in landscape first
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=node.node_id,
            row_index=0,
            data=token.row_data,
            row_id=token.row_id,
        )
        recorder.create_token(row_id=row.row_id, token_id=token.token_id)

        result, _updated_token, _error_sink = executor.execute_transform(
            transform=as_transform(transform),
            token=token,
            ctx=ctx,
            step_in_pipeline=1,  # First transform is at step 1 (source=0)
        )

        assert result.status == "success"
        assert result.row == {"value": 42}
        # Audit fields populated
        assert result.input_hash is not None
        assert result.output_hash is not None
        assert result.duration_ms is not None

    def test_execute_transform_error(self) -> None:
        from elspeth.contracts import TokenInfo
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import TransformExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import TransformResult

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="failing",
            node_type="transform",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        class FailingTransform:
            name = "failing"
            node_id = node.node_id
            _on_error = "discard"  # Required for transforms that return errors

            def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
                return TransformResult.error({"message": "validation failed"})

        transform = FailingTransform()
        ctx = PluginContext(run_id=run.run_id, config={}, landscape=recorder)
        executor = TransformExecutor(recorder, SpanFactory())

        token = TokenInfo(
            row_id="row-1",
            token_id="token-1",
            row_data={"value": -1},
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=node.node_id,
            row_index=0,
            data=token.row_data,
            row_id=token.row_id,
        )
        recorder.create_token(row_id=row.row_id, token_id=token.token_id)

        result, _, _error_sink = executor.execute_transform(
            transform=as_transform(transform),
            token=token,
            ctx=ctx,
            step_in_pipeline=1,
        )

        assert result.status == "error"
        assert result.reason == {"message": "validation failed"}

    def test_execute_transform_exception_records_failure(self) -> None:
        """Transform raising exception still records audit state."""
        from elspeth.contracts import TokenInfo
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import TransformExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import TransformResult

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="exploding",
            node_type="transform",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        class ExplodingTransform:
            name = "exploding"
            node_id = node.node_id

            def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
                raise RuntimeError("kaboom!")

        transform = ExplodingTransform()
        ctx = PluginContext(run_id=run.run_id, config={})
        executor = TransformExecutor(recorder, SpanFactory())

        token = TokenInfo(
            row_id="row-1",
            token_id="token-1",
            row_data={"value": 99},
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=node.node_id,
            row_index=0,
            data=token.row_data,
            row_id=token.row_id,
        )
        recorder.create_token(row_id=row.row_id, token_id=token.token_id)

        with pytest.raises(RuntimeError, match="kaboom"):
            executor.execute_transform(
                transform=as_transform(transform),
                token=token,
                ctx=ctx,
                step_in_pipeline=1,
            )

        # Verify failure was recorded in landscape
        states = recorder.get_node_states_for_token(token.token_id)
        assert len(states) == 1
        state = states[0]
        assert state.status == "failed"
        # Type narrowing: failed status means NodeStateFailed which has duration_ms
        assert hasattr(state, "duration_ms") and state.duration_ms is not None

    def test_execute_transform_updates_token_row_data(self) -> None:
        """Updated token should have new row_data."""
        from elspeth.contracts import TokenInfo
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import TransformExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import TransformResult

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="enricher",
            node_type="transform",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        class EnrichTransform:
            name = "enricher"
            node_id = node.node_id

            def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
                return TransformResult.success({**row, "enriched": True})

        transform = EnrichTransform()
        ctx = PluginContext(run_id=run.run_id, config={})
        executor = TransformExecutor(recorder, SpanFactory())

        token = TokenInfo(
            row_id="row-1",
            token_id="token-1",
            row_data={"original": "data"},
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=node.node_id,
            row_index=0,
            data=token.row_data,
            row_id=token.row_id,
        )
        recorder.create_token(row_id=row.row_id, token_id=token.token_id)

        _result, updated_token, _error_sink = executor.execute_transform(
            transform=as_transform(transform),
            token=token,
            ctx=ctx,
            step_in_pipeline=1,
        )

        # Updated token has new row data
        assert updated_token.row_data == {"original": "data", "enriched": True}
        # Identity preserved
        assert updated_token.token_id == token.token_id
        assert updated_token.row_id == token.row_id

    def test_node_state_records_input_and_output(self) -> None:
        """Node state should record both input and output hashes."""
        from elspeth.contracts import TokenInfo
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import TransformExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import TransformResult

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="identity",
            node_type="transform",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        class IdentityTransform:
            name = "identity"
            node_id = node.node_id

            def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
                return TransformResult.success(row)

        transform = IdentityTransform()
        ctx = PluginContext(run_id=run.run_id, config={})
        executor = TransformExecutor(recorder, SpanFactory())

        token = TokenInfo(
            row_id="row-1",
            token_id="token-1",
            row_data={"key": "value"},
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=node.node_id,
            row_index=0,
            data=token.row_data,
            row_id=token.row_id,
        )
        recorder.create_token(row_id=row.row_id, token_id=token.token_id)

        executor.execute_transform(
            transform=as_transform(transform),
            token=token,
            ctx=ctx,
            step_in_pipeline=1,
        )

        # Check node state in landscape
        states = recorder.get_node_states_for_token(token.token_id)
        assert len(states) == 1
        state = states[0]
        assert state.status == "completed"
        # Type narrowing: completed status means NodeStateCompleted which has output_hash
        assert state.input_hash is not None
        assert hasattr(state, "output_hash") and state.output_hash is not None
        # Same input/output data means same hashes for identity transform
        assert state.input_hash == state.output_hash

    def test_execute_transform_returns_error_sink_on_discard(self) -> None:
        """When transform errors with on_error='discard', returns error_sink='discard'."""
        from elspeth.contracts import TokenInfo
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import TransformExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import TransformResult

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="discarding",
            node_type="transform",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        class DiscardingTransform:
            name = "discarding"
            node_id = node.node_id
            _on_error = "discard"

            def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
                return TransformResult.error({"message": "invalid input"})

        transform = DiscardingTransform()
        ctx = PluginContext(run_id=run.run_id, config={}, landscape=recorder)
        executor = TransformExecutor(recorder, SpanFactory())

        token = TokenInfo(
            row_id="row-1",
            token_id="token-1",
            row_data={"value": -1},
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=node.node_id,
            row_index=0,
            data=token.row_data,
            row_id=token.row_id,
        )
        recorder.create_token(row_id=row.row_id, token_id=token.token_id)

        result, _updated_token, error_sink = executor.execute_transform(
            transform=as_transform(transform),
            token=token,
            ctx=ctx,
            step_in_pipeline=1,
        )

        assert result.status == "error"
        assert error_sink == "discard"

    def test_execute_transform_returns_error_sink_name(self) -> None:
        """When transform errors with on_error=sink_name, returns that sink name."""
        from elspeth.contracts import TokenInfo
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import TransformExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import TransformResult

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="routing_to_error",
            node_type="transform",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        class ErrorRoutingTransform:
            name = "routing_to_error"
            node_id = node.node_id
            _on_error = "error_sink"  # Routes to named error sink

            def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
                return TransformResult.error({"message": "routing to error sink"})

        transform = ErrorRoutingTransform()
        ctx = PluginContext(run_id=run.run_id, config={}, landscape=recorder)
        executor = TransformExecutor(recorder, SpanFactory())

        token = TokenInfo(
            row_id="row-1",
            token_id="token-1",
            row_data={"value": "bad"},
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=node.node_id,
            row_index=0,
            data=token.row_data,
            row_id=token.row_id,
        )
        recorder.create_token(row_id=row.row_id, token_id=token.token_id)

        result, _updated_token, error_sink = executor.execute_transform(
            transform=as_transform(transform),
            token=token,
            ctx=ctx,
            step_in_pipeline=1,
        )

        assert result.status == "error"
        assert error_sink == "error_sink"

    def test_execute_transform_returns_none_error_sink_on_success(self) -> None:
        """On success, error_sink is None."""
        from elspeth.contracts import TokenInfo
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import TransformExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import TransformResult

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="successful",
            node_type="transform",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        class SuccessfulTransform:
            name = "successful"
            node_id = node.node_id

            def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
                return TransformResult.success({"result": "ok"})

        transform = SuccessfulTransform()
        ctx = PluginContext(run_id=run.run_id, config={})
        executor = TransformExecutor(recorder, SpanFactory())

        token = TokenInfo(
            row_id="row-1",
            token_id="token-1",
            row_data={"value": 42},
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=node.node_id,
            row_index=0,
            data=token.row_data,
            row_id=token.row_id,
        )
        recorder.create_token(row_id=row.row_id, token_id=token.token_id)

        result, _updated_token, error_sink = executor.execute_transform(
            transform=as_transform(transform),
            token=token,
            ctx=ctx,
            step_in_pipeline=1,
        )

        assert result.status == "success"
        assert error_sink is None

    def test_execute_transform_records_attempt_number(self) -> None:
        """Attempt number is passed to begin_node_state."""
        from unittest.mock import patch

        from elspeth.contracts import TokenInfo
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import TransformExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import TransformResult

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="attempt_test",
            node_type="transform",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        class SimpleTransform:
            name = "attempt_test"
            node_id = node.node_id

            def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
                return TransformResult.success(row)

        transform = SimpleTransform()
        ctx = PluginContext(run_id=run.run_id, config={})
        executor = TransformExecutor(recorder, SpanFactory())

        token = TokenInfo(
            row_id="row-1",
            token_id="token-1",
            row_data={"value": 42},
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=node.node_id,
            row_index=0,
            data=token.row_data,
            row_id=token.row_id,
        )
        recorder.create_token(row_id=row.row_id, token_id=token.token_id)

        # Patch begin_node_state to capture attempt
        with patch.object(recorder, "begin_node_state", wraps=recorder.begin_node_state) as mock:
            executor.execute_transform(
                transform=as_transform(transform),
                token=token,
                ctx=ctx,
                step_in_pipeline=0,
                attempt=2,  # Non-default attempt
            )

        # Verify attempt was passed
        mock.assert_called_once()
        call_kwargs = mock.call_args.kwargs
        assert call_kwargs.get("attempt") == 2


class TestGateExecutor:
    """Gate execution with audit and routing."""

    def test_execute_gate_continue(self) -> None:
        """Gate returns continue action - no routing event recorded."""
        from elspeth.contracts import TokenInfo
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import GateExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import GateResult, RoutingAction

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        gate_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="pass_through",
            node_type="gate",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # Mock gate that continues
        class PassThroughGate:
            name = "pass_through"
            node_id = gate_node.node_id

            def evaluate(self, row: dict[str, Any], ctx: PluginContext) -> GateResult:
                return GateResult(
                    row=row,
                    action=RoutingAction.continue_(),
                )

        gate = PassThroughGate()
        ctx = PluginContext(run_id=run.run_id, config={})
        executor = GateExecutor(recorder, SpanFactory())

        token = TokenInfo(
            row_id="row-1",
            token_id="token-1",
            row_data={"value": 42},
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=gate_node.node_id,
            row_index=0,
            data=token.row_data,
            row_id=token.row_id,
        )
        recorder.create_token(row_id=row.row_id, token_id=token.token_id)

        outcome = executor.execute_gate(
            gate=as_gate(gate),
            token=token,
            ctx=ctx,
            step_in_pipeline=1,
        )

        # Verify outcome
        assert outcome.result.action.kind == "continue"
        assert outcome.sink_name is None
        assert outcome.child_tokens == []
        assert outcome.updated_token.row_data == {"value": 42}

        # Verify audit fields populated
        assert outcome.result.input_hash is not None
        assert outcome.result.output_hash is not None
        assert outcome.result.duration_ms is not None

        # Verify node state recorded as completed
        states = recorder.get_node_states_for_token(token.token_id)
        assert len(states) == 1
        assert states[0].status == "completed"

        # Verify no routing events (continue doesn't create one)
        events = recorder.get_routing_events(states[0].state_id)
        assert len(events) == 0

    def test_execute_gate_route(self) -> None:
        """Gate routes to sink via route label - routing event recorded."""
        from elspeth.contracts import TokenInfo
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import GateExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import GateResult, RoutingAction

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        # Register gate and sink nodes
        gate_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="threshold_gate",
            node_type="gate",
            plugin_version="1.0",
            config={"threshold": 100},
            schema_config=DYNAMIC_SCHEMA,
        )
        sink_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="high_values",
            node_type="sink",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # Register edge from gate to sink using route label
        edge = recorder.register_edge(
            run_id=run.run_id,
            from_node_id=gate_node.node_id,
            to_node_id=sink_node.node_id,
            label="above",  # Route label, not sink name
            mode=RoutingMode.MOVE,
        )

        # Mock gate that routes high values using route label
        class ThresholdGate:
            name = "threshold_gate"
            node_id = gate_node.node_id

            def evaluate(self, row: dict[str, Any], ctx: PluginContext) -> GateResult:
                if row.get("value", 0) > 100:
                    return GateResult(
                        row=row,
                        action=RoutingAction.route(
                            "above",  # Route label
                            reason={"threshold_exceeded": True, "value": row["value"]},
                        ),
                    )
                return GateResult(row=row, action=RoutingAction.continue_())

        gate = ThresholdGate()
        ctx = PluginContext(run_id=run.run_id, config={})

        # Edge map: (node_id, label) -> edge_id
        edge_map = {(gate_node.node_id, "above"): edge.edge_id}
        # Route resolution map: (node_id, label) -> sink_name
        route_resolution_map = {(gate_node.node_id, "above"): "high_values"}
        executor = GateExecutor(recorder, SpanFactory(), edge_map, route_resolution_map)

        token = TokenInfo(
            row_id="row-1",
            token_id="token-1",
            row_data={"value": 150},  # Above threshold
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=gate_node.node_id,
            row_index=0,
            data=token.row_data,
            row_id=token.row_id,
        )
        recorder.create_token(row_id=row.row_id, token_id=token.token_id)

        outcome = executor.execute_gate(
            gate=as_gate(gate),
            token=token,
            ctx=ctx,
            step_in_pipeline=1,
        )

        # Verify outcome
        assert outcome.result.action.kind == "route"
        assert outcome.sink_name == "high_values"
        assert outcome.child_tokens == []

        # Verify node state recorded as completed (terminal state derived from events)
        states = recorder.get_node_states_for_token(token.token_id)
        assert len(states) == 1
        assert states[0].status == "completed"

        # Verify routing event recorded
        events = recorder.get_routing_events(states[0].state_id)
        assert len(events) == 1
        assert events[0].edge_id == edge.edge_id
        assert events[0].mode == "move"

    def test_missing_edge_raises_error(self) -> None:
        """Gate routing to unregistered route label raises MissingEdgeError."""
        from elspeth.contracts import TokenInfo
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import GateExecutor, MissingEdgeError
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import GateResult, RoutingAction

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        gate_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="broken_gate",
            node_type="gate",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # Mock gate that routes to a label that has no route resolution
        class BrokenGate:
            name = "broken_gate"
            node_id = gate_node.node_id

            def evaluate(self, row: dict[str, Any], ctx: PluginContext) -> GateResult:
                return GateResult(
                    row=row,
                    action=RoutingAction.route("nonexistent_label"),
                )

        gate = BrokenGate()
        ctx = PluginContext(run_id=run.run_id, config={})

        # Empty route resolution map - label not configured
        executor = GateExecutor(recorder, SpanFactory(), edge_map={}, route_resolution_map={})

        token = TokenInfo(
            row_id="row-1",
            token_id="token-1",
            row_data={"value": 42},
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=gate_node.node_id,
            row_index=0,
            data=token.row_data,
            row_id=token.row_id,
        )
        recorder.create_token(row_id=row.row_id, token_id=token.token_id)

        with pytest.raises(MissingEdgeError) as exc_info:
            executor.execute_gate(
                gate=as_gate(gate),
                token=token,
                ctx=ctx,
                step_in_pipeline=1,
            )

        # Verify error details
        assert exc_info.value.node_id == gate_node.node_id
        assert exc_info.value.label == "nonexistent_label"
        assert "Audit trail would be incomplete" in str(exc_info.value)

    def test_execute_gate_fork(self) -> None:
        """Gate forks to multiple paths - routing events and child tokens created."""
        from elspeth.contracts import TokenInfo
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import GateExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.engine.tokens import TokenManager
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import GateResult, RoutingAction

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        # Register gate and path nodes
        gate_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="splitter",
            node_type="gate",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        path_a_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="path_a",
            node_type="transform",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        path_b_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="path_b",
            node_type="transform",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # Register edges
        edge_a = recorder.register_edge(
            run_id=run.run_id,
            from_node_id=gate_node.node_id,
            to_node_id=path_a_node.node_id,
            label="path_a",
            mode=RoutingMode.COPY,
        )
        edge_b = recorder.register_edge(
            run_id=run.run_id,
            from_node_id=gate_node.node_id,
            to_node_id=path_b_node.node_id,
            label="path_b",
            mode=RoutingMode.COPY,
        )

        # Mock gate that forks to both paths
        class SplitterGate:
            name = "splitter"
            node_id = gate_node.node_id

            def evaluate(self, row: dict[str, Any], ctx: PluginContext) -> GateResult:
                return GateResult(
                    row=row,
                    action=RoutingAction.fork_to_paths(
                        ["path_a", "path_b"],
                        reason={"split_reason": "parallel processing"},
                    ),
                )

        gate = SplitterGate()
        ctx = PluginContext(run_id=run.run_id, config={})

        edge_map = {
            (gate_node.node_id, "path_a"): edge_a.edge_id,
            (gate_node.node_id, "path_b"): edge_b.edge_id,
        }
        executor = GateExecutor(recorder, SpanFactory(), edge_map)
        token_manager = TokenManager(recorder)

        token = TokenInfo(
            row_id="row-1",
            token_id="token-1",
            row_data={"value": 42},
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=gate_node.node_id,
            row_index=0,
            data=token.row_data,
            row_id=token.row_id,
        )
        recorder.create_token(row_id=row.row_id, token_id=token.token_id)

        outcome = executor.execute_gate(
            gate=as_gate(gate),
            token=token,
            ctx=ctx,
            step_in_pipeline=1,
            token_manager=token_manager,
        )

        # Verify outcome
        assert outcome.result.action.kind == "fork_to_paths"
        assert outcome.sink_name is None
        assert len(outcome.child_tokens) == 2

        # Verify child tokens have correct branch names
        branch_names = {t.branch_name for t in outcome.child_tokens}
        assert branch_names == {"path_a", "path_b"}

        # Verify all child tokens share the same row_id
        for child in outcome.child_tokens:
            assert child.row_id == token.row_id
            assert child.row_data == {"value": 42}

        # Verify routing events recorded
        states = recorder.get_node_states_for_token(token.token_id)
        events = recorder.get_routing_events(states[0].state_id)
        assert len(events) == 2

        # All events should share the same routing_group_id (fork group)
        group_ids = {e.routing_group_id for e in events}
        assert len(group_ids) == 1

    def test_fork_without_token_manager_raises_error(self) -> None:
        """Gate fork without token_manager raises RuntimeError for audit integrity."""
        from elspeth.contracts import TokenInfo
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import GateExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import GateResult, RoutingAction

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        # Register gate and path nodes
        gate_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="splitter",
            node_type="gate",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        path_a_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="path_a",
            node_type="transform",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        path_b_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="path_b",
            node_type="transform",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # Register edges
        edge_a = recorder.register_edge(
            run_id=run.run_id,
            from_node_id=gate_node.node_id,
            to_node_id=path_a_node.node_id,
            label="path_a",
            mode=RoutingMode.COPY,
        )
        edge_b = recorder.register_edge(
            run_id=run.run_id,
            from_node_id=gate_node.node_id,
            to_node_id=path_b_node.node_id,
            label="path_b",
            mode=RoutingMode.COPY,
        )

        # Mock gate that forks to multiple paths
        class SplitterGate:
            name = "splitter"
            node_id = gate_node.node_id

            def evaluate(self, row: dict[str, Any], ctx: PluginContext) -> GateResult:
                return GateResult(
                    row=row,
                    action=RoutingAction.fork_to_paths(["path_a", "path_b"]),
                )

        gate = SplitterGate()
        ctx = PluginContext(run_id=run.run_id, config={})

        edge_map = {
            (gate_node.node_id, "path_a"): edge_a.edge_id,
            (gate_node.node_id, "path_b"): edge_b.edge_id,
        }
        executor = GateExecutor(recorder, SpanFactory(), edge_map)

        token = TokenInfo(
            row_id="row-1",
            token_id="token-1",
            row_data={"value": 42},
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=gate_node.node_id,
            row_index=0,
            data=token.row_data,
            row_id=token.row_id,
        )
        recorder.create_token(row_id=row.row_id, token_id=token.token_id)

        # Call without token_manager - should raise RuntimeError
        with pytest.raises(RuntimeError, match="audit integrity would be compromised"):
            executor.execute_gate(
                gate=as_gate(gate),
                token=token,
                ctx=ctx,
                step_in_pipeline=1,
                token_manager=None,  # Explicitly None
            )

    def test_execute_gate_exception_records_failure(self) -> None:
        """Gate raising exception still records audit state."""
        from elspeth.contracts import TokenInfo
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import GateExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import GateResult

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        gate_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="exploding_gate",
            node_type="gate",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        class ExplodingGate:
            name = "exploding_gate"
            node_id = gate_node.node_id

            def evaluate(self, row: dict[str, Any], ctx: PluginContext) -> GateResult:
                raise RuntimeError("gate evaluation failed!")

        gate = ExplodingGate()
        ctx = PluginContext(run_id=run.run_id, config={})
        executor = GateExecutor(recorder, SpanFactory())

        token = TokenInfo(
            row_id="row-1",
            token_id="token-1",
            row_data={"value": 42},
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=gate_node.node_id,
            row_index=0,
            data=token.row_data,
            row_id=token.row_id,
        )
        recorder.create_token(row_id=row.row_id, token_id=token.token_id)

        with pytest.raises(RuntimeError, match="gate evaluation failed"):
            executor.execute_gate(
                gate=as_gate(gate),
                token=token,
                ctx=ctx,
                step_in_pipeline=1,
            )

        # Verify failure was recorded in landscape
        states = recorder.get_node_states_for_token(token.token_id)
        assert len(states) == 1
        state = states[0]
        assert state.status == "failed"
        # Type narrowing: failed status means NodeStateFailed which has duration_ms
        assert hasattr(state, "duration_ms") and state.duration_ms is not None


class TestConfigGateExecutor:
    """Config-driven gate execution with ExpressionParser."""

    def test_execute_config_gate_continue(self) -> None:
        """Config gate returns continue destination - no routing event recorded."""
        from elspeth.contracts import TokenInfo
        from elspeth.core.config import GateSettings
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import GateExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.context import PluginContext

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        gate_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="quality_check",
            node_type="gate",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # Config-driven gate that checks confidence
        gate_config = GateSettings(
            name="quality_check",
            condition="row['confidence'] >= 0.85",
            routes={"true": "continue", "false": "review_sink"},
        )

        ctx = PluginContext(run_id=run.run_id, config={})
        executor = GateExecutor(recorder, SpanFactory())

        token = TokenInfo(
            row_id="row-1",
            token_id="token-1",
            row_data={"confidence": 0.95},  # Above threshold
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=gate_node.node_id,
            row_index=0,
            data=token.row_data,
            row_id=token.row_id,
        )
        recorder.create_token(row_id=row.row_id, token_id=token.token_id)

        outcome = executor.execute_config_gate(
            gate_config=gate_config,
            node_id=gate_node.node_id,
            token=token,
            ctx=ctx,
            step_in_pipeline=1,
        )

        # Verify outcome
        assert outcome.result.action.kind == "continue"
        assert outcome.sink_name is None
        assert outcome.child_tokens == []
        assert outcome.updated_token.row_data == {"confidence": 0.95}

        # Verify audit fields populated
        assert outcome.result.input_hash is not None
        assert outcome.result.output_hash is not None
        assert outcome.result.duration_ms is not None

        # Verify node state recorded as completed
        states = recorder.get_node_states_for_token(token.token_id)
        assert len(states) == 1
        assert states[0].status == "completed"

        # Verify no routing events (continue doesn't create one)
        events = recorder.get_routing_events(states[0].state_id)
        assert len(events) == 0

    def test_execute_config_gate_route_to_sink(self) -> None:
        """Config gate routes to sink when condition evaluates to route label."""
        from elspeth.contracts import TokenInfo
        from elspeth.core.config import GateSettings
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import GateExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.context import PluginContext

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        gate_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="quality_check",
            node_type="gate",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        sink_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="review_sink",
            node_type="sink",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # Register edge for "false" route label
        edge = recorder.register_edge(
            run_id=run.run_id,
            from_node_id=gate_node.node_id,
            to_node_id=sink_node.node_id,
            label="false",
            mode=RoutingMode.MOVE,
        )

        gate_config = GateSettings(
            name="quality_check",
            condition="row['confidence'] >= 0.85",
            routes={"true": "continue", "false": "review_sink"},
        )

        edge_map = {(gate_node.node_id, "false"): edge.edge_id}
        ctx = PluginContext(run_id=run.run_id, config={})
        executor = GateExecutor(recorder, SpanFactory(), edge_map)

        token = TokenInfo(
            row_id="row-1",
            token_id="token-1",
            row_data={"confidence": 0.5},  # Below threshold
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=gate_node.node_id,
            row_index=0,
            data=token.row_data,
            row_id=token.row_id,
        )
        recorder.create_token(row_id=row.row_id, token_id=token.token_id)

        outcome = executor.execute_config_gate(
            gate_config=gate_config,
            node_id=gate_node.node_id,
            token=token,
            ctx=ctx,
            step_in_pipeline=1,
        )

        # Verify routing to sink
        assert outcome.result.action.kind == "route"
        assert outcome.sink_name == "review_sink"
        assert outcome.child_tokens == []

        # Verify routing event recorded
        states = recorder.get_node_states_for_token(token.token_id)
        events = recorder.get_routing_events(states[0].state_id)
        assert len(events) == 1
        assert events[0].edge_id == edge.edge_id

    def test_execute_config_gate_string_result(self) -> None:
        """Config gate using ternary expression that returns string route labels."""
        from elspeth.contracts import TokenInfo
        from elspeth.core.config import GateSettings
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import GateExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.context import PluginContext

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        gate_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="priority_router",
            node_type="gate",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        high_sink = recorder.register_node(
            run_id=run.run_id,
            plugin_name="high_priority_sink",
            node_type="sink",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        edge = recorder.register_edge(
            run_id=run.run_id,
            from_node_id=gate_node.node_id,
            to_node_id=high_sink.node_id,
            label="high",
            mode=RoutingMode.MOVE,
        )

        # Ternary expression returning string route labels
        gate_config = GateSettings(
            name="priority_router",
            condition="'high' if row['priority'] > 5 else 'low'",
            routes={"high": "high_priority_sink", "low": "continue"},
        )

        edge_map = {(gate_node.node_id, "high"): edge.edge_id}
        ctx = PluginContext(run_id=run.run_id, config={})
        executor = GateExecutor(recorder, SpanFactory(), edge_map)

        token = TokenInfo(
            row_id="row-1",
            token_id="token-1",
            row_data={"priority": 8},  # High priority
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=gate_node.node_id,
            row_index=0,
            data=token.row_data,
            row_id=token.row_id,
        )
        recorder.create_token(row_id=row.row_id, token_id=token.token_id)

        outcome = executor.execute_config_gate(
            gate_config=gate_config,
            node_id=gate_node.node_id,
            token=token,
            ctx=ctx,
            step_in_pipeline=1,
        )

        assert outcome.result.action.kind == "route"
        assert outcome.sink_name == "high_priority_sink"

    def test_execute_config_gate_fork(self) -> None:
        """Config gate forks to multiple paths."""
        from elspeth.contracts import TokenInfo
        from elspeth.core.config import GateSettings
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import GateExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.engine.tokens import TokenManager
        from elspeth.plugins.context import PluginContext

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        gate_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="parallel_analysis",
            node_type="gate",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        path_a = recorder.register_node(
            run_id=run.run_id,
            plugin_name="path_a",
            node_type="transform",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        path_b = recorder.register_node(
            run_id=run.run_id,
            plugin_name="path_b",
            node_type="transform",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        edge_a = recorder.register_edge(
            run_id=run.run_id,
            from_node_id=gate_node.node_id,
            to_node_id=path_a.node_id,
            label="path_a",
            mode=RoutingMode.COPY,
        )
        edge_b = recorder.register_edge(
            run_id=run.run_id,
            from_node_id=gate_node.node_id,
            to_node_id=path_b.node_id,
            label="path_b",
            mode=RoutingMode.COPY,
        )

        gate_config = GateSettings(
            name="parallel_analysis",
            condition="True",  # Always fork
            routes={"true": "fork", "false": "continue"},
            fork_to=["path_a", "path_b"],
        )

        edge_map = {
            (gate_node.node_id, "path_a"): edge_a.edge_id,
            (gate_node.node_id, "path_b"): edge_b.edge_id,
        }
        ctx = PluginContext(run_id=run.run_id, config={})
        executor = GateExecutor(recorder, SpanFactory(), edge_map)
        token_manager = TokenManager(recorder)

        token = TokenInfo(
            row_id="row-1",
            token_id="token-1",
            row_data={"value": 42},
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=gate_node.node_id,
            row_index=0,
            data=token.row_data,
            row_id=token.row_id,
        )
        recorder.create_token(row_id=row.row_id, token_id=token.token_id)

        outcome = executor.execute_config_gate(
            gate_config=gate_config,
            node_id=gate_node.node_id,
            token=token,
            ctx=ctx,
            step_in_pipeline=1,
            token_manager=token_manager,
        )

        # Verify fork
        assert outcome.result.action.kind == "fork_to_paths"
        assert outcome.sink_name is None
        assert len(outcome.child_tokens) == 2

        # Verify child tokens have correct branch names
        branch_names = {t.branch_name for t in outcome.child_tokens}
        assert branch_names == {"path_a", "path_b"}

        # Verify routing events
        states = recorder.get_node_states_for_token(token.token_id)
        events = recorder.get_routing_events(states[0].state_id)
        assert len(events) == 2

    def test_execute_config_gate_fork_without_token_manager_raises_error(self) -> None:
        """Config gate fork without token_manager raises RuntimeError."""
        from elspeth.contracts import TokenInfo
        from elspeth.core.config import GateSettings
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import GateExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.context import PluginContext

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        gate_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="fork_gate",
            node_type="gate",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        gate_config = GateSettings(
            name="fork_gate",
            condition="True",
            routes={"true": "fork", "false": "continue"},
            fork_to=["path_a", "path_b"],
        )

        ctx = PluginContext(run_id=run.run_id, config={})
        executor = GateExecutor(recorder, SpanFactory())

        token = TokenInfo(
            row_id="row-1",
            token_id="token-1",
            row_data={"value": 42},
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=gate_node.node_id,
            row_index=0,
            data=token.row_data,
            row_id=token.row_id,
        )
        recorder.create_token(row_id=row.row_id, token_id=token.token_id)

        with pytest.raises(RuntimeError, match="audit integrity would be compromised"):
            executor.execute_config_gate(
                gate_config=gate_config,
                node_id=gate_node.node_id,
                token=token,
                ctx=ctx,
                step_in_pipeline=1,
                token_manager=None,
            )

    def test_execute_config_gate_missing_route_label_raises_error(self) -> None:
        """Config gate condition returning unlisted label raises ValueError."""
        from elspeth.contracts import TokenInfo
        from elspeth.core.config import GateSettings
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import GateExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.context import PluginContext

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        gate_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="broken_gate",
            node_type="gate",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # Gate returns "maybe" but routes only define "true"/"false"
        gate_config = GateSettings(
            name="broken_gate",
            condition="'maybe'",  # Returns string not in routes
            routes={"true": "continue", "false": "error_sink"},
        )

        ctx = PluginContext(run_id=run.run_id, config={})
        executor = GateExecutor(recorder, SpanFactory())

        token = TokenInfo(
            row_id="row-1",
            token_id="token-1",
            row_data={"value": 42},
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=gate_node.node_id,
            row_index=0,
            data=token.row_data,
            row_id=token.row_id,
        )
        recorder.create_token(row_id=row.row_id, token_id=token.token_id)

        with pytest.raises(ValueError, match="which is not in routes"):
            executor.execute_config_gate(
                gate_config=gate_config,
                node_id=gate_node.node_id,
                token=token,
                ctx=ctx,
                step_in_pipeline=1,
            )

        # Verify failure was recorded
        states = recorder.get_node_states_for_token(token.token_id)
        assert len(states) == 1
        assert states[0].status == "failed"

    def test_execute_config_gate_expression_error_records_failure(self) -> None:
        """Config gate expression failure records audit state."""
        from elspeth.contracts import TokenInfo
        from elspeth.core.config import GateSettings
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import GateExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.context import PluginContext

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        gate_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="error_gate",
            node_type="gate",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # Expression accesses missing field
        gate_config = GateSettings(
            name="error_gate",
            condition="row['nonexistent'] > 0",
            routes={"true": "continue", "false": "continue"},
        )

        ctx = PluginContext(run_id=run.run_id, config={})
        executor = GateExecutor(recorder, SpanFactory())

        token = TokenInfo(
            row_id="row-1",
            token_id="token-1",
            row_data={"value": 42},  # No 'nonexistent' field
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=gate_node.node_id,
            row_index=0,
            data=token.row_data,
            row_id=token.row_id,
        )
        recorder.create_token(row_id=row.row_id, token_id=token.token_id)

        with pytest.raises(KeyError):
            executor.execute_config_gate(
                gate_config=gate_config,
                node_id=gate_node.node_id,
                token=token,
                ctx=ctx,
                step_in_pipeline=1,
            )

        # Verify failure was recorded
        states = recorder.get_node_states_for_token(token.token_id)
        assert len(states) == 1
        assert states[0].status == "failed"

    def test_execute_config_gate_missing_edge_raises_error(self) -> None:
        """Config gate routing to unregistered edge raises MissingEdgeError."""
        from elspeth.contracts import TokenInfo
        from elspeth.core.config import GateSettings
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import GateExecutor, MissingEdgeError
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.context import PluginContext

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        gate_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="routing_gate",
            node_type="gate",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        gate_config = GateSettings(
            name="routing_gate",
            condition="row['value'] < 0",
            routes={"true": "error_sink", "false": "continue"},
        )

        # No edge registered for "true" route
        ctx = PluginContext(run_id=run.run_id, config={})
        executor = GateExecutor(recorder, SpanFactory(), edge_map={})

        token = TokenInfo(
            row_id="row-1",
            token_id="token-1",
            row_data={"value": -5},  # Will trigger route to error_sink
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=gate_node.node_id,
            row_index=0,
            data=token.row_data,
            row_id=token.row_id,
        )
        recorder.create_token(row_id=row.row_id, token_id=token.token_id)

        with pytest.raises(MissingEdgeError) as exc_info:
            executor.execute_config_gate(
                gate_config=gate_config,
                node_id=gate_node.node_id,
                token=token,
                ctx=ctx,
                step_in_pipeline=1,
            )

        assert exc_info.value.node_id == gate_node.node_id
        assert exc_info.value.label == "true"

    def test_execute_config_gate_reason_includes_condition(self) -> None:
        """Config gate routing action reason includes condition and result."""
        from elspeth.contracts import TokenInfo
        from elspeth.core.config import GateSettings
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import GateExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.context import PluginContext

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        gate_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="audit_gate",
            node_type="gate",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        gate_config = GateSettings(
            name="audit_gate",
            condition="row['score'] > 100",
            routes={"true": "continue", "false": "continue"},
        )

        ctx = PluginContext(run_id=run.run_id, config={})
        executor = GateExecutor(recorder, SpanFactory())

        token = TokenInfo(
            row_id="row-1",
            token_id="token-1",
            row_data={"score": 150},
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=gate_node.node_id,
            row_index=0,
            data=token.row_data,
            row_id=token.row_id,
        )
        recorder.create_token(row_id=row.row_id, token_id=token.token_id)

        outcome = executor.execute_config_gate(
            gate_config=gate_config,
            node_id=gate_node.node_id,
            token=token,
            ctx=ctx,
            step_in_pipeline=1,
        )

        # Verify reason is recorded for audit trail
        reason = dict(outcome.result.action.reason)
        assert reason["condition"] == "row['score'] > 100"
        assert reason["result"] == "true"


class TestAggregationExecutorOldInterfaceDeleted:
    """Verify old accept()/flush() executor interface is deleted.

    OLD: TestAggregationExecutor tested executor.accept() and executor.flush()
         with plugin-level aggregation interface.
    NEW: Aggregation is engine-controlled via buffer_row()/execute_flush()
         with batch-aware transforms (is_batch_aware=True).
         See TestAggregationExecutorBuffering for new tests.
    """

    def test_old_accept_method_deleted(self) -> None:
        """Old accept() method should be deleted from AggregationExecutor."""
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import AggregationExecutor
        from elspeth.engine.spans import SpanFactory

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        executor = AggregationExecutor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
        )

        # Old accept() method should be deleted
        assert not hasattr(executor, "accept"), "accept() method should be deleted - use buffer_row() instead"

    def test_old_flush_method_deleted(self) -> None:
        """Old flush() method should be deleted from AggregationExecutor."""
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import AggregationExecutor
        from elspeth.engine.spans import SpanFactory

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        executor = AggregationExecutor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
        )

        # Old flush() method should be deleted (execute_flush() is the production method)
        # Note: flush() was the old method that called plugin.flush()
        # execute_flush() is the production method with full audit recording
        # _get_buffered_data() is internal-only for testing
        assert hasattr(executor, "execute_flush"), "execute_flush() should exist for production flush with audit"
        assert hasattr(executor, "_get_buffered_data"), "_get_buffered_data() should exist for testing buffer contents"


class TestAggregationExecutorTriggersDeleted:
    """Verify BaseAggregation-based trigger tests are deleted.

    OLD: TestAggregationExecutorTriggers tested trigger evaluation with
         BaseAggregation plugins (accept/flush interface).
    NEW: Trigger evaluation still exists but operates on engine buffers,
         not plugin state. See TestAggregationExecutorBuffering.
    """

    def test_base_aggregation_deleted(self) -> None:
        """BaseAggregation should be deleted (aggregation is structural)."""
        import elspeth.plugins.base as base

        assert not hasattr(base, "BaseAggregation"), "BaseAggregation should be deleted - use is_batch_aware=True on BaseTransform"


class TestSinkExecutor:
    """Sink execution with artifact recording."""

    def test_write_records_artifact(self) -> None:
        """Write tokens to sink records artifact in Landscape."""
        from elspeth.contracts import TokenInfo
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.engine.executors import SinkExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.context import PluginContext

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        sink_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="csv_output",
            node_type="sink",
            plugin_version="1.0",
            config={"path": "/tmp/output.csv"},
            schema_config=DYNAMIC_SCHEMA,
        )

        # Mock sink that writes rows and returns artifact info
        class CsvSink:
            name = "csv_output"
            node_id: str | None = sink_node.node_id

            def write(self, rows: list[dict[str, Any]], ctx: PluginContext) -> ArtifactDescriptor:
                # Simulate writing rows and return artifact info
                return ArtifactDescriptor.for_file(
                    path="/tmp/output.csv",
                    size_bytes=1024,
                    content_hash="abc123",
                )

        sink = CsvSink()
        ctx = PluginContext(run_id=run.run_id, config={})
        executor = SinkExecutor(recorder, SpanFactory(), run.run_id)

        # Create tokens
        tokens = []
        for i in range(3):
            token = TokenInfo(
                row_id=f"row-{i}",
                token_id=f"token-{i}",
                row_data={"value": i * 10},
            )
            row = recorder.create_row(
                run_id=run.run_id,
                source_node_id=sink_node.node_id,
                row_index=i,
                data=token.row_data,
                row_id=token.row_id,
            )
            recorder.create_token(row_id=row.row_id, token_id=token.token_id)
            tokens.append(token)

        # Write tokens to sink
        artifact = executor.write(
            sink=as_sink(sink),
            tokens=tokens,
            ctx=ctx,
            step_in_pipeline=5,
        )

        # Verify artifact returned with correct info
        assert artifact is not None
        assert artifact.path_or_uri == "file:///tmp/output.csv"
        assert artifact.size_bytes == 1024
        assert artifact.content_hash == "abc123"
        assert artifact.artifact_type == "file"
        assert artifact.sink_node_id == sink_node.node_id

        # Verify artifact recorded in Landscape
        artifacts = recorder.get_artifacts(run.run_id)
        assert len(artifacts) == 1
        assert artifacts[0].artifact_id == artifact.artifact_id

        # Verify node_state created for EACH token (COMPLETED terminal state derivation)
        for token in tokens:
            states = recorder.get_node_states_for_token(token.token_id)
            assert len(states) == 1
            state = states[0]
            assert state.status == "completed"
            assert state.node_id == sink_node.node_id
            # Type narrowing: completed status means NodeStateCompleted which has duration_ms
            assert hasattr(state, "duration_ms") and state.duration_ms is not None

    def test_write_empty_tokens_returns_none(self) -> None:
        """Write with empty tokens returns None without side effects."""
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.engine.executors import SinkExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.context import PluginContext

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        sink_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="empty_sink",
            node_type="sink",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        class EmptySink:
            name = "empty_sink"
            node_id: str | None = sink_node.node_id

            def write(self, rows: list[dict[str, Any]], ctx: PluginContext) -> ArtifactDescriptor:
                raise AssertionError("Should not be called for empty tokens")

        sink = EmptySink()
        ctx = PluginContext(run_id=run.run_id, config={})
        executor = SinkExecutor(recorder, SpanFactory(), run.run_id)

        # Write with empty tokens
        artifact = executor.write(
            sink=as_sink(sink),
            tokens=[],
            ctx=ctx,
            step_in_pipeline=5,
        )

        assert artifact is None

        # Verify no artifacts recorded
        artifacts = recorder.get_artifacts(run.run_id)
        assert len(artifacts) == 0

    def test_write_exception_records_failure(self) -> None:
        """Sink raising exception still records audit state for all tokens."""
        from elspeth.contracts import TokenInfo
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.engine.executors import SinkExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.context import PluginContext

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        sink_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="exploding_sink",
            node_type="sink",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        class ExplodingSink:
            name = "exploding_sink"
            node_id: str | None = sink_node.node_id

            def write(self, rows: list[dict[str, Any]], ctx: PluginContext) -> ArtifactDescriptor:
                raise RuntimeError("disk full!")

        sink = ExplodingSink()
        ctx = PluginContext(run_id=run.run_id, config={})
        executor = SinkExecutor(recorder, SpanFactory(), run.run_id)

        # Create tokens
        tokens = []
        for i in range(2):
            token = TokenInfo(
                row_id=f"row-{i}",
                token_id=f"token-{i}",
                row_data={"value": i},
            )
            row = recorder.create_row(
                run_id=run.run_id,
                source_node_id=sink_node.node_id,
                row_index=i,
                data=token.row_data,
                row_id=token.row_id,
            )
            recorder.create_token(row_id=row.row_id, token_id=token.token_id)
            tokens.append(token)

        # Write should raise
        with pytest.raises(RuntimeError, match="disk full"):
            executor.write(
                sink=as_sink(sink),
                tokens=tokens,
                ctx=ctx,
                step_in_pipeline=5,
            )

        # Verify failure recorded for ALL tokens
        for token in tokens:
            states = recorder.get_node_states_for_token(token.token_id)
            assert len(states) == 1
            state = states[0]
            assert state.status == "failed"
            # Type narrowing: failed status means NodeStateFailed which has duration_ms
            assert hasattr(state, "duration_ms") and state.duration_ms is not None

        # Verify no artifact recorded (write failed)
        artifacts = recorder.get_artifacts(run.run_id)
        assert len(artifacts) == 0

    def test_write_multiple_batches_creates_multiple_artifacts(self) -> None:
        """Multiple sink writes create separate artifacts."""
        from elspeth.contracts import TokenInfo
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.engine.executors import SinkExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.context import PluginContext

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        sink_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="batch_sink",
            node_type="sink",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        class BatchSink:
            name = "batch_sink"
            node_id: str | None = sink_node.node_id
            _batch_count: int = 0

            def write(self, rows: list[dict[str, Any]], ctx: PluginContext) -> ArtifactDescriptor:
                self._batch_count += 1
                return ArtifactDescriptor.for_file(
                    path=f"/tmp/batch_{self._batch_count}.json",
                    size_bytes=len(rows) * 100,
                    content_hash=f"hash_{self._batch_count}",
                )

        sink = BatchSink()
        ctx = PluginContext(run_id=run.run_id, config={})
        executor = SinkExecutor(recorder, SpanFactory(), run.run_id)

        artifacts = []
        # Write two batches
        for batch_num in range(2):
            tokens = []
            for i in range(2):
                idx = batch_num * 2 + i
                token = TokenInfo(
                    row_id=f"row-{idx}",
                    token_id=f"token-{idx}",
                    row_data={"batch": batch_num, "index": i},
                )
                row = recorder.create_row(
                    run_id=run.run_id,
                    source_node_id=sink_node.node_id,
                    row_index=idx,
                    data=token.row_data,
                    row_id=token.row_id,
                )
                recorder.create_token(row_id=row.row_id, token_id=token.token_id)
                tokens.append(token)

            artifact = executor.write(
                sink=as_sink(sink),
                tokens=tokens,
                ctx=ctx,
                step_in_pipeline=5,
            )
            artifacts.append(artifact)

        # Verify two distinct artifacts
        assert len(artifacts) == 2
        assert artifacts[0] is not None
        assert artifacts[1] is not None
        assert artifacts[0].artifact_id != artifacts[1].artifact_id
        assert artifacts[0].path_or_uri == "file:///tmp/batch_1.json"
        assert artifacts[1].path_or_uri == "file:///tmp/batch_2.json"

        # Verify both in Landscape
        all_artifacts = recorder.get_artifacts(run.run_id)
        assert len(all_artifacts) == 2

    def test_artifact_linked_to_first_state(self) -> None:
        """Artifact is linked to first token's state_id for audit lineage."""
        from elspeth.contracts import TokenInfo
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.artifacts import ArtifactDescriptor
        from elspeth.engine.executors import SinkExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.context import PluginContext

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        sink_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="linked_sink",
            node_type="sink",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        class LinkedSink:
            name = "linked_sink"
            node_id: str | None = sink_node.node_id

            def write(self, rows: list[dict[str, Any]], ctx: PluginContext) -> ArtifactDescriptor:
                return ArtifactDescriptor.for_file(path="/tmp/linked.csv", size_bytes=512, content_hash="xyz")

        sink = LinkedSink()
        ctx = PluginContext(run_id=run.run_id, config={})
        executor = SinkExecutor(recorder, SpanFactory(), run.run_id)

        # Create multiple tokens
        tokens = []
        for i in range(3):
            token = TokenInfo(
                row_id=f"row-{i}",
                token_id=f"token-{i}",
                row_data={"index": i},
            )
            row = recorder.create_row(
                run_id=run.run_id,
                source_node_id=sink_node.node_id,
                row_index=i,
                data=token.row_data,
                row_id=token.row_id,
            )
            recorder.create_token(row_id=row.row_id, token_id=token.token_id)
            tokens.append(token)

        artifact = executor.write(
            sink=as_sink(sink),
            tokens=tokens,
            ctx=ctx,
            step_in_pipeline=5,
        )

        # Get first token's state
        first_token_states = recorder.get_node_states_for_token(tokens[0].token_id)
        assert len(first_token_states) == 1
        first_state_id = first_token_states[0].state_id

        # Verify artifact is linked to first state
        assert artifact is not None
        assert artifact.produced_by_state_id == first_state_id


class TestAggregationExecutorRestore:
    """Tests for aggregation state restoration."""

    def test_restore_state_sets_internal_state(self) -> None:
        """restore_state() stores state for plugin access."""
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import AggregationExecutor
        from elspeth.engine.spans import SpanFactory

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        executor = AggregationExecutor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
        )

        state = {"buffer": [1, 2, 3], "sum": 6, "count": 3}

        executor.restore_state("agg_node", state)

        assert executor.get_restored_state("agg_node") == state

    def test_restore_state_returns_none_for_unknown_node(self) -> None:
        """get_restored_state() returns None for nodes without restored state."""
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import AggregationExecutor
        from elspeth.engine.spans import SpanFactory

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        executor = AggregationExecutor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
        )

        assert executor.get_restored_state("unknown_node") is None

    def test_restore_batch_sets_current_batch(self) -> None:
        """restore_batch() makes batch the current batch for its node."""
        from elspeth.contracts.enums import Determinism, NodeType
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import AggregationExecutor
        from elspeth.engine.spans import SpanFactory

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        # Register aggregation node
        recorder.register_node(
            run_id=run.run_id,
            node_id="agg_node",
            plugin_name="test",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=DYNAMIC_SCHEMA,
        )

        # Create a batch
        batch = recorder.create_batch(
            run_id=run.run_id,
            aggregation_node_id="agg_node",
        )

        # Create executor for this run
        executor = AggregationExecutor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
        )

        # Act
        executor.restore_batch(batch.batch_id)

        # Assert
        assert executor.get_batch_id("agg_node") == batch.batch_id

    def test_restore_batch_not_found_raises_error(self) -> None:
        """restore_batch() raises ValueError for unknown batch_id."""
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import AggregationExecutor
        from elspeth.engine.spans import SpanFactory

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        executor = AggregationExecutor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
        )

        with pytest.raises(ValueError, match="Batch not found"):
            executor.restore_batch("nonexistent-batch-id")

    def test_restore_batch_restores_member_count_deleted(self) -> None:
        """Test deleted - used old accept() interface.

        OLD: Tested that restoring a batch lets you call accept() with correct ordinals.
        NEW: Restore functionality now uses buffer_row() interface instead.
             See TestAggregationExecutorCheckpoint for new restore tests.
        """
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import AggregationExecutor
        from elspeth.engine.spans import SpanFactory

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        executor = AggregationExecutor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
        )

        # Verify old accept() method is deleted
        assert not hasattr(executor, "accept"), "accept() method should be deleted - use buffer_row() instead"


class TestAggregationExecutorBuffering:
    """Tests for engine-level row buffering in AggregationExecutor."""

    def test_executor_buffers_rows_internally(self) -> None:
        """Executor buffers rows without calling plugin.accept()."""
        from elspeth.contracts import TokenInfo
        from elspeth.core.config import AggregationSettings, TriggerConfig
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import AggregationExecutor
        from elspeth.engine.spans import SpanFactory

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        agg_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="buffer_test",
            node_type="aggregation",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        settings = AggregationSettings(
            name="test_agg",
            plugin="test",
            trigger=TriggerConfig(count=3),
        )

        executor = AggregationExecutor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            aggregation_settings={agg_node.node_id: settings},
        )

        # Buffer 3 rows
        for i in range(3):
            token = TokenInfo(
                row_id=f"row-{i}",
                token_id=f"token-{i}",
                row_data={"value": i},
            )
            row = recorder.create_row(
                run_id=run.run_id,
                source_node_id=agg_node.node_id,
                row_index=i,
                data=token.row_data,
                row_id=token.row_id,
            )
            recorder.create_token(row_id=row.row_id, token_id=token.token_id)
            executor.buffer_row(agg_node.node_id, token)

        # Check buffer
        buffered = executor.get_buffered_rows(agg_node.node_id)
        assert len(buffered) == 3
        assert [r["value"] for r in buffered] == [0, 1, 2]

    def test_get_buffered_data_does_not_clear_buffer(self) -> None:
        """_get_buffered_data() returns data without clearing (internal method)."""
        from elspeth.contracts import TokenInfo
        from elspeth.core.config import AggregationSettings, TriggerConfig
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import AggregationExecutor
        from elspeth.engine.spans import SpanFactory

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        agg_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="buffer_flush_test",
            node_type="aggregation",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        settings = AggregationSettings(
            name="test_agg",
            plugin="test",
            trigger=TriggerConfig(count=2),
        )

        executor = AggregationExecutor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            aggregation_settings={agg_node.node_id: settings},
        )

        # Buffer rows
        for i in range(2):
            token = TokenInfo(
                row_id=f"row-{i}",
                token_id=f"token-{i}",
                row_data={"value": i},
            )
            row = recorder.create_row(
                run_id=run.run_id,
                source_node_id=agg_node.node_id,
                row_index=i,
                data=token.row_data,
                row_id=token.row_id,
            )
            recorder.create_token(row_id=row.row_id, token_id=token.token_id)
            executor.buffer_row(agg_node.node_id, token)

        # _get_buffered_data() returns data WITHOUT clearing (internal method)
        buffered_rows, buffered_tokens = executor._get_buffered_data(agg_node.node_id)
        assert len(buffered_rows) == 2
        assert len(buffered_tokens) == 2

        # Buffer should still contain data (not cleared by _get_buffered_data)
        assert executor.get_buffered_rows(agg_node.node_id) == buffered_rows
        assert executor.get_buffered_tokens(agg_node.node_id) == buffered_tokens

    def test_buffered_tokens_are_tracked(self) -> None:
        """Executor tracks TokenInfo objects alongside buffered rows."""
        from elspeth.contracts import TokenInfo
        from elspeth.core.config import AggregationSettings, TriggerConfig
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import AggregationExecutor
        from elspeth.engine.spans import SpanFactory

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        agg_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="token_track_test",
            node_type="aggregation",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        settings = AggregationSettings(
            name="test_agg",
            plugin="test",
            trigger=TriggerConfig(count=2),
        )

        executor = AggregationExecutor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            aggregation_settings={agg_node.node_id: settings},
        )

        # Buffer 2 rows
        for i in range(2):
            token = TokenInfo(
                row_id=f"row-{i}",
                token_id=f"token-{i}",
                row_data={"value": i * 10},
            )
            row = recorder.create_row(
                run_id=run.run_id,
                source_node_id=agg_node.node_id,
                row_index=i,
                data=token.row_data,
                row_id=token.row_id,
            )
            recorder.create_token(row_id=row.row_id, token_id=token.token_id)
            executor.buffer_row(agg_node.node_id, token)

        # Check buffered tokens
        tokens = executor.get_buffered_tokens(agg_node.node_id)
        assert len(tokens) == 2
        assert tokens[0].token_id == "token-0"
        assert tokens[1].token_id == "token-1"

    def test_buffer_creates_batch_on_first_row(self) -> None:
        """buffer_row() creates a batch on first row just like accept()."""
        from elspeth.contracts import TokenInfo
        from elspeth.core.config import AggregationSettings, TriggerConfig
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import AggregationExecutor
        from elspeth.engine.spans import SpanFactory

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        agg_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="batch_create_test",
            node_type="aggregation",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        settings = AggregationSettings(
            name="test_agg",
            plugin="test",
            trigger=TriggerConfig(count=5),
        )

        executor = AggregationExecutor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            aggregation_settings={agg_node.node_id: settings},
        )

        # No batch yet
        assert executor.get_batch_id(agg_node.node_id) is None

        # Buffer first row
        token = TokenInfo(
            row_id="row-0",
            token_id="token-0",
            row_data={"value": 42},
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=agg_node.node_id,
            row_index=0,
            data=token.row_data,
            row_id=token.row_id,
        )
        recorder.create_token(row_id=row.row_id, token_id=token.token_id)
        executor.buffer_row(agg_node.node_id, token)

        # Batch should now exist
        batch_id = executor.get_batch_id(agg_node.node_id)
        assert batch_id is not None

        # Batch should be in landscape
        batch = recorder.get_batch(batch_id)
        assert batch is not None
        assert batch.aggregation_node_id == agg_node.node_id

    def test_buffer_updates_trigger_evaluator(self) -> None:
        """buffer_row() updates trigger evaluator count."""
        from elspeth.contracts import TokenInfo
        from elspeth.core.config import AggregationSettings, TriggerConfig
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import AggregationExecutor
        from elspeth.engine.spans import SpanFactory

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        agg_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="trigger_update_test",
            node_type="aggregation",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        settings = AggregationSettings(
            name="test_agg",
            plugin="test",
            trigger=TriggerConfig(count=3),  # Trigger at 3
        )

        executor = AggregationExecutor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            aggregation_settings={agg_node.node_id: settings},
        )

        # Buffer 2 rows - should not trigger
        for i in range(2):
            token = TokenInfo(
                row_id=f"row-{i}",
                token_id=f"token-{i}",
                row_data={"value": i},
            )
            row = recorder.create_row(
                run_id=run.run_id,
                source_node_id=agg_node.node_id,
                row_index=i,
                data=token.row_data,
                row_id=token.row_id,
            )
            recorder.create_token(row_id=row.row_id, token_id=token.token_id)
            executor.buffer_row(agg_node.node_id, token)

        assert executor.should_flush(agg_node.node_id) is False

        # Buffer 3rd row - should trigger
        token = TokenInfo(
            row_id="row-2",
            token_id="token-2",
            row_data={"value": 2},
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=agg_node.node_id,
            row_index=2,
            data=token.row_data,
            row_id=token.row_id,
        )
        recorder.create_token(row_id=row.row_id, token_id=token.token_id)
        executor.buffer_row(agg_node.node_id, token)

        assert executor.should_flush(agg_node.node_id) is True

    def test_get_buffered_data_returns_both_rows_and_tokens(self) -> None:
        """_get_buffered_data() returns tuple of (rows, tokens) for passthrough mode."""
        from elspeth.contracts import TokenInfo
        from elspeth.core.config import AggregationSettings, TriggerConfig
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import AggregationExecutor
        from elspeth.engine.spans import SpanFactory

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        agg_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="flush_returns_tokens_test",
            node_type="aggregation",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        settings = AggregationSettings(
            name="test_agg",
            plugin="test",
            trigger=TriggerConfig(count=2),
        )

        executor = AggregationExecutor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            aggregation_settings={agg_node.node_id: settings},
        )

        # Buffer 2 rows with distinct tokens
        token1 = TokenInfo(
            row_id="row-1",
            token_id="token-1",
            row_data={"x": 1},
        )
        token2 = TokenInfo(
            row_id="row-2",
            token_id="token-2",
            row_data={"x": 2},
        )

        for i, token in enumerate([token1, token2]):
            row = recorder.create_row(
                run_id=run.run_id,
                source_node_id=agg_node.node_id,
                row_index=i,
                data=token.row_data,
                row_id=token.row_id,
            )
            recorder.create_token(row_id=row.row_id, token_id=token.token_id)
            executor.buffer_row(agg_node.node_id, token)

        # _get_buffered_data() returns tuple of (rows, tokens) without clearing
        rows, tokens = executor._get_buffered_data(agg_node.node_id)

        # Verify rows
        assert len(rows) == 2
        assert rows[0] == {"x": 1}
        assert rows[1] == {"x": 2}

        # Verify tokens
        assert len(tokens) == 2
        assert tokens[0].token_id == "token-1"
        assert tokens[1].token_id == "token-2"

        # Buffer should NOT be cleared (_get_buffered_data is internal, doesn't clear)
        assert executor.get_buffered_rows(agg_node.node_id) == rows
        assert executor.get_buffered_tokens(agg_node.node_id) == tokens


class TestAggregationExecutorCheckpoint:
    """Tests for buffer serialization/deserialization for crash recovery."""

    def test_get_checkpoint_state_returns_buffer_contents(self) -> None:
        """get_checkpoint_state() returns serializable buffer state."""
        import json

        from elspeth.contracts import TokenInfo
        from elspeth.core.config import AggregationSettings, TriggerConfig
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import AggregationExecutor
        from elspeth.engine.spans import SpanFactory

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        agg_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="checkpoint_test",
            node_type="aggregation",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        settings = AggregationSettings(
            name="test_agg",
            plugin="test",
            trigger=TriggerConfig(count=10),  # High count so we don't trigger
        )

        executor = AggregationExecutor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            aggregation_settings={agg_node.node_id: settings},
        )

        # Buffer some rows
        for i in range(3):
            token = TokenInfo(
                row_id=f"row-{i}",
                token_id=f"token-{i}",
                row_data={"value": i},
            )
            row = recorder.create_row(
                run_id=run.run_id,
                source_node_id=agg_node.node_id,
                row_index=i,
                data=token.row_data,
                row_id=token.row_id,
            )
            recorder.create_token(row_id=row.row_id, token_id=token.token_id)
            executor.buffer_row(agg_node.node_id, token)

        # Get checkpoint state
        state = executor.get_checkpoint_state()

        assert agg_node.node_id in state
        assert state[agg_node.node_id]["rows"] == [
            {"value": 0},
            {"value": 1},
            {"value": 2},
        ]
        assert state[agg_node.node_id]["token_ids"] == [
            "token-0",
            "token-1",
            "token-2",
        ]
        assert state[agg_node.node_id]["batch_id"] is not None

        # Must be JSON serializable
        json_str = json.dumps(state)
        restored = json.loads(json_str)
        assert restored == state

    def test_get_checkpoint_state_excludes_empty_buffers(self) -> None:
        """get_checkpoint_state() only includes non-empty buffers."""
        from elspeth.core.config import AggregationSettings, TriggerConfig
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import AggregationExecutor
        from elspeth.engine.spans import SpanFactory

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        agg_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="empty_buffer_test",
            node_type="aggregation",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        settings = AggregationSettings(
            name="test_agg",
            plugin="test",
            trigger=TriggerConfig(count=10),
        )

        executor = AggregationExecutor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            aggregation_settings={agg_node.node_id: settings},
        )

        # Don't buffer anything
        state = executor.get_checkpoint_state()
        assert state == {}  # Empty buffers not included

    def test_restore_from_checkpoint_restores_buffers(self) -> None:
        """restore_from_checkpoint() restores buffer state."""
        from elspeth.core.config import AggregationSettings, TriggerConfig
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import AggregationExecutor
        from elspeth.engine.spans import SpanFactory

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        agg_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="restore_test",
            node_type="aggregation",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        settings = AggregationSettings(
            name="test_agg",
            plugin="test",
            trigger=TriggerConfig(count=10),
        )

        executor = AggregationExecutor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            aggregation_settings={agg_node.node_id: settings},
        )

        # Simulate checkpoint state from previous run
        checkpoint_state = {
            agg_node.node_id: {
                "rows": [{"value": 0}, {"value": 1}, {"value": 2}],
                "token_ids": ["token-0", "token-1", "token-2"],
                "batch_id": "batch-123",
            }
        }

        executor.restore_from_checkpoint(checkpoint_state)

        # Buffer should be restored
        buffered = executor.get_buffered_rows(agg_node.node_id)
        assert len(buffered) == 3
        assert buffered == [{"value": 0}, {"value": 1}, {"value": 2}]

        # Batch ID should be restored
        assert executor.get_batch_id(agg_node.node_id) == "batch-123"

    def test_restore_from_checkpoint_restores_trigger_count(self) -> None:
        """restore_from_checkpoint() restores trigger evaluator count."""
        from elspeth.core.config import AggregationSettings, TriggerConfig
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import AggregationExecutor
        from elspeth.engine.spans import SpanFactory

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        agg_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="trigger_restore_test",
            node_type="aggregation",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        settings = AggregationSettings(
            name="test_agg",
            plugin="test",
            trigger=TriggerConfig(count=5),  # Trigger at 5
        )

        executor = AggregationExecutor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            aggregation_settings={agg_node.node_id: settings},
        )

        # Simulate checkpoint state with 4 rows buffered
        checkpoint_state = {
            agg_node.node_id: {
                "rows": [{"value": i} for i in range(4)],
                "token_ids": [f"token-{i}" for i in range(4)],
                "batch_id": "batch-123",
            }
        }

        executor.restore_from_checkpoint(checkpoint_state)

        # Trigger evaluator should reflect restored count (4 rows)
        # Should NOT trigger yet (need 5)
        assert executor.should_flush(agg_node.node_id) is False

        # Trigger evaluator internal count should be 4
        evaluator = executor._trigger_evaluators[agg_node.node_id]
        assert evaluator.batch_count == 4

    def test_checkpoint_roundtrip(self) -> None:
        """Buffer state survives checkpoint/restore cycle."""
        import json

        from elspeth.contracts import TokenInfo
        from elspeth.core.config import AggregationSettings, TriggerConfig
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import AggregationExecutor
        from elspeth.engine.spans import SpanFactory

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        agg_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="roundtrip_test",
            node_type="aggregation",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        settings = AggregationSettings(
            name="test_agg",
            plugin="test",
            trigger=TriggerConfig(count=10),
        )

        # First executor - buffer some rows
        executor1 = AggregationExecutor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            aggregation_settings={agg_node.node_id: settings},
        )

        for i in range(3):
            token = TokenInfo(
                row_id=f"row-{i}",
                token_id=f"token-{i}",
                row_data={"value": i * 10},
            )
            row = recorder.create_row(
                run_id=run.run_id,
                source_node_id=agg_node.node_id,
                row_index=i,
                data=token.row_data,
                row_id=token.row_id,
            )
            recorder.create_token(row_id=row.row_id, token_id=token.token_id)
            executor1.buffer_row(agg_node.node_id, token)

        # Get checkpoint state and serialize (simulates crash)
        state = executor1.get_checkpoint_state()
        serialized = json.dumps(state)

        # Second executor - restore from checkpoint
        executor2 = AggregationExecutor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            aggregation_settings={agg_node.node_id: settings},
        )

        restored_state = json.loads(serialized)
        executor2.restore_from_checkpoint(restored_state)

        # Verify buffer restored correctly
        buffered = executor2.get_buffered_rows(agg_node.node_id)
        assert buffered == [{"value": 0}, {"value": 10}, {"value": 20}]

        # Verify trigger count restored
        evaluator = executor2._trigger_evaluators[agg_node.node_id]
        assert evaluator.batch_count == 3
