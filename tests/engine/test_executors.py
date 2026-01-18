# tests/engine/test_executors.py
"""Tests for plugin executors."""

from typing import Any

import pytest

from elspeth.contracts import RoutingMode
from elspeth.contracts.schema import SchemaConfig

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

            def process(
                self, row: dict[str, Any], ctx: PluginContext
            ) -> TransformResult:
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
            transform=transform,
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

            def process(
                self, row: dict[str, Any], ctx: PluginContext
            ) -> TransformResult:
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
            transform=transform,
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

            def process(
                self, row: dict[str, Any], ctx: PluginContext
            ) -> TransformResult:
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
                transform=transform,
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

            def process(
                self, row: dict[str, Any], ctx: PluginContext
            ) -> TransformResult:
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
            transform=transform,
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

            def process(
                self, row: dict[str, Any], ctx: PluginContext
            ) -> TransformResult:
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
            transform=transform,
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

            def process(
                self, row: dict[str, Any], ctx: PluginContext
            ) -> TransformResult:
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
            transform=transform,
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

            def process(
                self, row: dict[str, Any], ctx: PluginContext
            ) -> TransformResult:
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
            transform=transform,
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

            def process(
                self, row: dict[str, Any], ctx: PluginContext
            ) -> TransformResult:
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
            transform=transform,
            token=token,
            ctx=ctx,
            step_in_pipeline=1,
        )

        assert result.status == "success"
        assert error_sink is None


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
            gate=gate,
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
            gate=gate,
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
        executor = GateExecutor(
            recorder, SpanFactory(), edge_map={}, route_resolution_map={}
        )

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
                gate=gate,
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
            gate=gate,
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
                gate=gate,
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
                gate=gate,
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
            routes={"true": "fork"},
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
            routes={"true": "fork"},
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


class TestAggregationExecutor:
    """Aggregation execution with batch tracking."""

    def test_accept_creates_batch(self) -> None:
        """First accept creates batch and sets batch_id."""
        from elspeth.contracts import TokenInfo
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import AggregationExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import AcceptResult

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        agg_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="sum_aggregator",
            node_type="aggregation",
            plugin_version="1.0",
            config={"batch_size": 2},
            schema_config=DYNAMIC_SCHEMA,
        )

        # Mock aggregation that collects values
        class SumAggregation:
            name = "sum_aggregator"
            node_id = agg_node.node_id
            _values: list[int]

            def __init__(self) -> None:
                self._values = []

            def accept(self, row: dict[str, Any], ctx: PluginContext) -> AcceptResult:
                self._values.append(row["value"])
                return AcceptResult(accepted=True)

            def flush(self, ctx: PluginContext) -> list[dict[str, Any]]:
                total = sum(self._values)
                self._values = []
                return [{"sum": total}]

        aggregation = SumAggregation()
        ctx = PluginContext(run_id=run.run_id, config={})
        executor = AggregationExecutor(recorder, SpanFactory(), run.run_id)

        # Create token
        token = TokenInfo(
            row_id="row-1",
            token_id="token-1",
            row_data={"value": 10},
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=agg_node.node_id,
            row_index=0,
            data=token.row_data,
            row_id=token.row_id,
        )
        recorder.create_token(row_id=row.row_id, token_id=token.token_id)

        # First accept should create batch
        result = executor.accept(
            aggregation=aggregation,
            token=token,
            ctx=ctx,
            step_in_pipeline=1,
        )

        # Verify result
        assert result.accepted is True
        assert result.batch_id is not None
        assert executor.get_batch_id(aggregation.node_id) == result.batch_id

        # Verify batch was created in landscape
        batch = recorder.get_batch(result.batch_id)
        assert batch is not None
        assert batch.status == "draft"
        assert batch.aggregation_node_id == agg_node.node_id

        # Verify batch member recorded
        members = recorder.get_batch_members(result.batch_id)
        assert len(members) == 1
        assert members[0].token_id == token.token_id
        assert members[0].ordinal == 0

        # Verify node state recorded
        states = recorder.get_node_states_for_token(token.token_id)
        assert len(states) == 1
        assert states[0].status == "completed"

    def test_accept_adds_to_existing_batch(self) -> None:
        """Subsequent accepts add to existing batch."""
        from elspeth.contracts import TokenInfo
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import AggregationExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import AcceptResult

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        agg_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="count_aggregator",
            node_type="aggregation",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # Mock aggregation
        class CountAggregation:
            name = "count_aggregator"
            node_id = agg_node.node_id
            _count: int = 0

            def accept(self, row: dict[str, Any], ctx: PluginContext) -> AcceptResult:
                self._count += 1
                return AcceptResult(accepted=True)

            def flush(self, ctx: PluginContext) -> list[dict[str, Any]]:
                result = [{"count": self._count}]
                self._count = 0
                return result

        aggregation = CountAggregation()
        ctx = PluginContext(run_id=run.run_id, config={})
        executor = AggregationExecutor(recorder, SpanFactory(), run.run_id)

        # Create and accept two tokens
        tokens = []
        for i in range(2):
            token = TokenInfo(
                row_id=f"row-{i}",
                token_id=f"token-{i}",
                row_data={"index": i},
            )
            row = recorder.create_row(
                run_id=run.run_id,
                source_node_id=agg_node.node_id,
                row_index=i,
                data=token.row_data,
                row_id=token.row_id,
            )
            recorder.create_token(row_id=row.row_id, token_id=token.token_id)
            tokens.append(token)

        # Accept first token - creates batch
        result1 = executor.accept(aggregation, tokens[0], ctx, step_in_pipeline=1)
        batch_id = result1.batch_id
        assert batch_id is not None  # First accept always creates batch

        # Accept second token - adds to same batch
        result2 = executor.accept(aggregation, tokens[1], ctx, step_in_pipeline=1)

        # Both should have same batch_id
        assert result1.batch_id == result2.batch_id
        assert executor.get_batch_id(aggregation.node_id) == batch_id

        # Verify both members recorded with correct ordinals
        members = recorder.get_batch_members(batch_id)
        assert len(members) == 2
        assert members[0].token_id == tokens[0].token_id
        assert members[0].ordinal == 0
        assert members[1].token_id == tokens[1].token_id
        assert members[1].ordinal == 1

    def test_flush_with_audit(self) -> None:
        """Flush transitions batch and returns outputs."""
        from elspeth.contracts import TokenInfo
        from elspeth.core.config import TriggerConfig
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import AggregationExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.engine.triggers import TriggerEvaluator
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import AcceptResult

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        agg_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="avg_aggregator",
            node_type="aggregation",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # Mock aggregation that computes average
        class AvgAggregation:
            name = "avg_aggregator"
            node_id = agg_node.node_id
            _values: list[float]

            def __init__(self) -> None:
                self._values = []

            def accept(self, row: dict[str, Any], ctx: PluginContext) -> AcceptResult:
                self._values.append(row["value"])
                return AcceptResult(accepted=True)

            def flush(self, ctx: PluginContext) -> list[dict[str, Any]]:
                avg = sum(self._values) / len(self._values) if self._values else 0
                self._values = []
                return [{"average": avg}]

        aggregation = AvgAggregation()
        ctx = PluginContext(run_id=run.run_id, config={})
        executor = AggregationExecutor(recorder, SpanFactory(), run.run_id)
        # Engine uses TriggerEvaluator to manage trigger conditions (WP-06)
        trigger_evaluator = TriggerEvaluator(TriggerConfig(count=2))

        # Accept two rows
        batch_id: str | None = None
        for i, value in enumerate([10.0, 20.0]):
            token = TokenInfo(
                row_id=f"row-{i}",
                token_id=f"token-{i}",
                row_data={"value": value},
            )
            row = recorder.create_row(
                run_id=run.run_id,
                source_node_id=agg_node.node_id,
                row_index=i,
                data=token.row_data,
                row_id=token.row_id,
            )
            recorder.create_token(row_id=row.row_id, token_id=token.token_id)

            result = executor.accept(aggregation, token, ctx, step_in_pipeline=1)
            batch_id = result.batch_id
            trigger_evaluator.record_accept()
            if trigger_evaluator.should_trigger():
                break

        assert batch_id is not None  # Trigger condition reached after 2 rows

        # Flush the batch
        outputs = executor.flush(
            aggregation=aggregation,
            ctx=ctx,
            trigger_reason="count_reached",
            step_in_pipeline=1,
        )

        # Verify outputs
        assert len(outputs) == 1
        assert outputs[0]["average"] == 15.0

        # Verify batch status is completed
        batch = recorder.get_batch(batch_id)
        assert batch is not None
        assert batch.status == "completed"
        assert batch.trigger_reason == "count_reached"

        # Verify batch_id is reset in executor
        assert executor.get_batch_id(aggregation.node_id) is None

    def test_flush_without_batch_raises_error(self) -> None:
        """Flush without prior accept raises ValueError."""
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import AggregationExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import AcceptResult

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        agg_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="no_batch",
            node_type="aggregation",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        class NoBatchAggregation:
            name = "no_batch"
            node_id = agg_node.node_id

            def accept(self, row: dict[str, Any], ctx: PluginContext) -> AcceptResult:
                return AcceptResult(accepted=True)

            def flush(self, ctx: PluginContext) -> list[dict[str, Any]]:
                return []

        aggregation = NoBatchAggregation()
        ctx = PluginContext(run_id=run.run_id, config={})
        executor = AggregationExecutor(recorder, SpanFactory(), run.run_id)

        # Flush without accept should raise
        with pytest.raises(ValueError, match="No batch to flush"):
            executor.flush(
                aggregation=aggregation,
                ctx=ctx,
                trigger_reason="manual",
                step_in_pipeline=1,
            )

    def test_accept_exception_records_failure(self) -> None:
        """Exception in accept() records failure state."""
        from elspeth.contracts import TokenInfo
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import AggregationExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import AcceptResult

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        agg_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="exploding_agg",
            node_type="aggregation",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        class ExplodingAggregation:
            name = "exploding_agg"
            node_id = agg_node.node_id

            def accept(self, row: dict[str, Any], ctx: PluginContext) -> AcceptResult:
                raise RuntimeError("accept failed!")

            def flush(self, ctx: PluginContext) -> list[dict[str, Any]]:
                return []

        aggregation = ExplodingAggregation()
        ctx = PluginContext(run_id=run.run_id, config={})
        executor = AggregationExecutor(recorder, SpanFactory(), run.run_id)

        token = TokenInfo(
            row_id="row-1",
            token_id="token-1",
            row_data={"value": 1},
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=agg_node.node_id,
            row_index=0,
            data=token.row_data,
            row_id=token.row_id,
        )
        recorder.create_token(row_id=row.row_id, token_id=token.token_id)

        with pytest.raises(RuntimeError, match="accept failed"):
            executor.accept(aggregation, token, ctx, step_in_pipeline=1)

        # Verify failure was recorded
        states = recorder.get_node_states_for_token(token.token_id)
        assert len(states) == 1
        state = states[0]
        assert state.status == "failed"
        # Type narrowing: failed status means NodeStateFailed which has duration_ms
        assert hasattr(state, "duration_ms") and state.duration_ms is not None

    def test_flush_exception_marks_batch_failed(self) -> None:
        """Exception in flush() marks batch as failed."""
        from elspeth.contracts import TokenInfo
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import AggregationExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import AcceptResult

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        agg_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="exploding_flush",
            node_type="aggregation",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        class ExplodingFlushAggregation:
            name = "exploding_flush"
            node_id = agg_node.node_id

            def accept(self, row: dict[str, Any], ctx: PluginContext) -> AcceptResult:
                return AcceptResult(accepted=True)

            def flush(self, ctx: PluginContext) -> list[dict[str, Any]]:
                raise RuntimeError("flush failed!")

        aggregation = ExplodingFlushAggregation()
        ctx = PluginContext(run_id=run.run_id, config={})
        executor = AggregationExecutor(recorder, SpanFactory(), run.run_id)

        token = TokenInfo(
            row_id="row-1",
            token_id="token-1",
            row_data={"value": 1},
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=agg_node.node_id,
            row_index=0,
            data=token.row_data,
            row_id=token.row_id,
        )
        recorder.create_token(row_id=row.row_id, token_id=token.token_id)

        # Accept succeeds
        result = executor.accept(aggregation, token, ctx, step_in_pipeline=1)
        batch_id = result.batch_id
        assert batch_id is not None  # Trigger always fires on first accept

        # Flush fails
        with pytest.raises(RuntimeError, match="flush failed"):
            executor.flush(
                aggregation=aggregation,
                ctx=ctx,
                trigger_reason="trigger",
                step_in_pipeline=1,
            )

        # Verify batch is marked failed
        batch = recorder.get_batch(batch_id)
        assert batch is not None
        assert batch.status == "failed"

    def test_multiple_batches_sequential(self) -> None:
        """After flush, new batch is created for subsequent accepts."""
        from elspeth.contracts import TokenInfo
        from elspeth.core.config import TriggerConfig
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import AggregationExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.engine.triggers import TriggerEvaluator
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import AcceptResult

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        agg_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="batch_counter",
            node_type="aggregation",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        class BatchCounterAggregation:
            name = "batch_counter"
            node_id = agg_node.node_id
            _count: int = 0
            _batch_num: int = 0

            def accept(self, row: dict[str, Any], ctx: PluginContext) -> AcceptResult:
                self._count += 1
                return AcceptResult(accepted=True)

            def flush(self, ctx: PluginContext) -> list[dict[str, Any]]:
                self._batch_num += 1
                result = [{"batch": self._batch_num, "count": self._count}]
                self._count = 0
                return result

        aggregation = BatchCounterAggregation()
        ctx = PluginContext(run_id=run.run_id, config={})
        executor = AggregationExecutor(recorder, SpanFactory(), run.run_id)
        # Engine uses TriggerEvaluator to manage trigger conditions (WP-06)
        trigger_evaluator = TriggerEvaluator(TriggerConfig(count=2))

        batch_ids: list[str] = []

        # Process 4 tokens -> 2 batches
        for i in range(4):
            token = TokenInfo(
                row_id=f"row-{i}",
                token_id=f"token-{i}",
                row_data={"index": i},
            )
            row = recorder.create_row(
                run_id=run.run_id,
                source_node_id=agg_node.node_id,
                row_index=i,
                data=token.row_data,
                row_id=token.row_id,
            )
            recorder.create_token(row_id=row.row_id, token_id=token.token_id)

            result = executor.accept(aggregation, token, ctx, step_in_pipeline=1)
            trigger_evaluator.record_accept()
            if trigger_evaluator.should_trigger():
                assert result.batch_id is not None  # Batch exists when trigger fires
                batch_ids.append(result.batch_id)
                _outputs = executor.flush(
                    aggregation=aggregation,
                    ctx=ctx,
                    trigger_reason="count_reached",
                    step_in_pipeline=1,
                )
                trigger_evaluator.reset()

        # Should have 2 completed batches with different IDs
        assert len(batch_ids) == 2
        assert batch_ids[0] != batch_ids[1]

        # Both batches should be completed
        for batch_id in batch_ids:
            batch = recorder.get_batch(batch_id)
            assert batch is not None
            assert batch.status == "completed"

        # Each batch should have 2 members
        for batch_id in batch_ids:
            members = recorder.get_batch_members(batch_id)
            assert len(members) == 2


class TestAggregationExecutorTriggers:
    """Tests for config-driven trigger evaluation in AggregationExecutor."""

    def test_executor_evaluates_count_trigger(self) -> None:
        """Executor evaluates count trigger and returns should_flush."""
        from elspeth.contracts.identity import TokenInfo
        from elspeth.core.config import AggregationSettings, TriggerConfig
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import AggregationExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.base import BaseAggregation
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import AcceptResult

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        agg_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="test",
            node_type="aggregation",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # Create a simple mock aggregation
        class SimpleAggregation(BaseAggregation):
            name = "test"
            input_schema = None
            output_schema = None

            def __init__(self) -> None:
                super().__init__({})
                self._buffer: list[dict[str, Any]] = []

            def accept(self, row: dict[str, Any], ctx: PluginContext) -> AcceptResult:
                self._buffer.append(row)
                return AcceptResult(accepted=True)

            def flush(self, ctx: PluginContext) -> list[dict[str, Any]]:
                result = list(self._buffer)
                self._buffer = []
                return result

        aggregation = SimpleAggregation()
        aggregation.node_id = agg_node.node_id

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

        ctx = PluginContext(run_id=run.run_id, config={})

        # First two accepts - should not trigger
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
            executor.accept(aggregation, token, ctx, step_in_pipeline=1)
            assert executor.should_flush(agg_node.node_id) is False

        # Third accept - should trigger
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
        executor.accept(aggregation, token, ctx, step_in_pipeline=1)
        assert executor.should_flush(agg_node.node_id) is True

    def test_executor_reset_trigger_after_flush(self) -> None:
        """Executor resets trigger state after flush."""
        from elspeth.contracts.identity import TokenInfo
        from elspeth.core.config import AggregationSettings, TriggerConfig
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import AggregationExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.base import BaseAggregation
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import AcceptResult

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        agg_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="test",
            node_type="aggregation",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        class SimpleAggregation(BaseAggregation):
            name = "test"
            input_schema = None
            output_schema = None

            def __init__(self) -> None:
                super().__init__({})
                self._buffer: list[dict[str, Any]] = []

            def accept(self, row: dict[str, Any], ctx: PluginContext) -> AcceptResult:
                self._buffer.append(row)
                return AcceptResult(accepted=True)

            def flush(self, ctx: PluginContext) -> list[dict[str, Any]]:
                result = list(self._buffer)
                self._buffer = []
                return result

        aggregation = SimpleAggregation()
        aggregation.node_id = agg_node.node_id

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

        ctx = PluginContext(run_id=run.run_id, config={})

        # Accept until trigger
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
            executor.accept(aggregation, token, ctx, step_in_pipeline=1)

        assert executor.should_flush(agg_node.node_id) is True

        # Flush
        executor.flush(aggregation, ctx, "count", step_in_pipeline=1)

        # After flush, trigger should be reset
        assert executor.should_flush(agg_node.node_id) is False


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

            def write(
                self, rows: list[dict[str, Any]], ctx: PluginContext
            ) -> ArtifactDescriptor:
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
            sink=sink,
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

            def write(
                self, rows: list[dict[str, Any]], ctx: PluginContext
            ) -> ArtifactDescriptor:
                raise AssertionError("Should not be called for empty tokens")

        sink = EmptySink()
        ctx = PluginContext(run_id=run.run_id, config={})
        executor = SinkExecutor(recorder, SpanFactory(), run.run_id)

        # Write with empty tokens
        artifact = executor.write(
            sink=sink,
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

            def write(
                self, rows: list[dict[str, Any]], ctx: PluginContext
            ) -> ArtifactDescriptor:
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
                sink=sink,
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

            def write(
                self, rows: list[dict[str, Any]], ctx: PluginContext
            ) -> ArtifactDescriptor:
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
                sink=sink,
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

            def write(
                self, rows: list[dict[str, Any]], ctx: PluginContext
            ) -> ArtifactDescriptor:
                return ArtifactDescriptor.for_file(
                    path="/tmp/linked.csv", size_bytes=512, content_hash="xyz"
                )

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
            sink=sink,
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
