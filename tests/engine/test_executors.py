# tests/engine/test_executors.py
"""Tests for plugin executors."""

from typing import Any

import pytest


class TestTransformExecutor:
    """Transform execution with audit."""

    def test_execute_transform_success(self) -> None:
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import TransformExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.engine.tokens import TokenInfo
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

        result, _updated_token = executor.execute_transform(
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
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import TransformExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.engine.tokens import TokenInfo
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
        )

        class FailingTransform:
            name = "failing"
            node_id = node.node_id

            def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
                return TransformResult.error({"message": "validation failed"})

        transform = FailingTransform()
        ctx = PluginContext(run_id=run.run_id, config={})
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

        result, _ = executor.execute_transform(
            transform=transform,
            token=token,
            ctx=ctx,
            step_in_pipeline=1,
        )

        assert result.status == "error"
        assert result.reason == {"message": "validation failed"}

    def test_execute_transform_exception_records_failure(self) -> None:
        """Transform raising exception still records audit state."""
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import TransformExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.engine.tokens import TokenInfo
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
                transform=transform,
                token=token,
                ctx=ctx,
                step_in_pipeline=1,
            )

        # Verify failure was recorded in landscape
        states = recorder.get_node_states_for_token(token.token_id)
        assert len(states) == 1
        assert states[0].status == "failed"
        assert states[0].duration_ms is not None

    def test_execute_transform_updates_token_row_data(self) -> None:
        """Updated token should have new row_data."""
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import TransformExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.engine.tokens import TokenInfo
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

        _result, updated_token = executor.execute_transform(
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
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import TransformExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.engine.tokens import TokenInfo
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
            transform=transform,
            token=token,
            ctx=ctx,
            step_in_pipeline=1,
        )

        # Check node state in landscape
        states = recorder.get_node_states_for_token(token.token_id)
        assert len(states) == 1
        state = states[0]
        assert state.input_hash is not None
        assert state.output_hash is not None
        assert state.status == "completed"
        # Same input/output data means same hashes for identity transform
        assert state.input_hash == state.output_hash


class TestGateExecutor:
    """Gate execution with audit and routing."""

    def test_execute_gate_continue(self) -> None:
        """Gate returns continue action - no routing event recorded."""
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import GateExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.engine.tokens import TokenInfo
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
        """Gate routes to sink - routing event recorded."""
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import GateExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.engine.tokens import TokenInfo
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
        )
        sink_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="high_values",
            node_type="sink",
            plugin_version="1.0",
            config={},
        )

        # Register edge from gate to sink
        edge = recorder.register_edge(
            run_id=run.run_id,
            from_node_id=gate_node.node_id,
            to_node_id=sink_node.node_id,
            label="high_values",
            mode="move",
        )

        # Mock gate that routes high values
        class ThresholdGate:
            name = "threshold_gate"
            node_id = gate_node.node_id

            def evaluate(self, row: dict[str, Any], ctx: PluginContext) -> GateResult:
                if row.get("value", 0) > 100:
                    return GateResult(
                        row=row,
                        action=RoutingAction.route_to_sink(
                            "high_values",
                            reason={"threshold_exceeded": True, "value": row["value"]},
                        ),
                    )
                return GateResult(row=row, action=RoutingAction.continue_())

        gate = ThresholdGate()
        ctx = PluginContext(run_id=run.run_id, config={})

        # Edge map: (node_id, label) -> edge_id
        edge_map = {(gate_node.node_id, "high_values"): edge.edge_id}
        executor = GateExecutor(recorder, SpanFactory(), edge_map)

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
        assert outcome.result.action.kind == "route_to_sink"
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
        """Gate routing to unregistered sink raises MissingEdgeError."""
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import GateExecutor, MissingEdgeError
        from elspeth.engine.spans import SpanFactory
        from elspeth.engine.tokens import TokenInfo
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
        )

        # Mock gate that routes to a sink that has no edge registered
        class BrokenGate:
            name = "broken_gate"
            node_id = gate_node.node_id

            def evaluate(self, row: dict[str, Any], ctx: PluginContext) -> GateResult:
                return GateResult(
                    row=row,
                    action=RoutingAction.route_to_sink("nonexistent_sink"),
                )

        gate = BrokenGate()
        ctx = PluginContext(run_id=run.run_id, config={})

        # Empty edge map - no edges registered
        executor = GateExecutor(recorder, SpanFactory(), edge_map={})

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
        assert exc_info.value.label == "nonexistent_sink"
        assert "Audit trail would be incomplete" in str(exc_info.value)

    def test_execute_gate_fork(self) -> None:
        """Gate forks to multiple paths - routing events and child tokens created."""
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import GateExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.engine.tokens import TokenInfo, TokenManager
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
        )
        path_a_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="path_a",
            node_type="transform",
            plugin_version="1.0",
            config={},
        )
        path_b_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="path_b",
            node_type="transform",
            plugin_version="1.0",
            config={},
        )

        # Register edges
        edge_a = recorder.register_edge(
            run_id=run.run_id,
            from_node_id=gate_node.node_id,
            to_node_id=path_a_node.node_id,
            label="path_a",
            mode="copy",
        )
        edge_b = recorder.register_edge(
            run_id=run.run_id,
            from_node_id=gate_node.node_id,
            to_node_id=path_b_node.node_id,
            label="path_b",
            mode="copy",
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
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import GateExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.engine.tokens import TokenInfo
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
        )
        path_a_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="path_a",
            node_type="transform",
            plugin_version="1.0",
            config={},
        )
        path_b_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="path_b",
            node_type="transform",
            plugin_version="1.0",
            config={},
        )

        # Register edges
        edge_a = recorder.register_edge(
            run_id=run.run_id,
            from_node_id=gate_node.node_id,
            to_node_id=path_a_node.node_id,
            label="path_a",
            mode="copy",
        )
        edge_b = recorder.register_edge(
            run_id=run.run_id,
            from_node_id=gate_node.node_id,
            to_node_id=path_b_node.node_id,
            label="path_b",
            mode="copy",
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
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import GateExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.engine.tokens import TokenInfo
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
        assert states[0].status == "failed"
        assert states[0].duration_ms is not None


class TestAggregationExecutor:
    """Aggregation execution with batch tracking."""

    def test_accept_creates_batch(self) -> None:
        """First accept creates batch and sets batch_id."""
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import AggregationExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.engine.tokens import TokenInfo
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
        )

        # Mock aggregation that collects values
        class SumAggregation:
            name = "sum_aggregator"
            node_id = agg_node.node_id
            _batch_id: str | None = None
            _values: list[int]

            def __init__(self) -> None:
                self._values = []

            def accept(self, row: dict[str, Any], ctx: PluginContext) -> AcceptResult:
                self._values.append(row["value"])
                trigger = len(self._values) >= 2
                return AcceptResult(accepted=True, trigger=trigger)

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
        assert aggregation._batch_id == result.batch_id

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
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import AggregationExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.engine.tokens import TokenInfo
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
        )

        # Mock aggregation
        class CountAggregation:
            name = "count_aggregator"
            node_id = agg_node.node_id
            _batch_id: str | None = None
            _count: int = 0

            def accept(self, row: dict[str, Any], ctx: PluginContext) -> AcceptResult:
                self._count += 1
                return AcceptResult(accepted=True, trigger=self._count >= 3)

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

        # Accept second token - adds to same batch
        result2 = executor.accept(aggregation, tokens[1], ctx, step_in_pipeline=1)

        # Both should have same batch_id
        assert result1.batch_id == result2.batch_id
        assert aggregation._batch_id == batch_id

        # Verify both members recorded with correct ordinals
        members = recorder.get_batch_members(batch_id)
        assert len(members) == 2
        assert members[0].token_id == tokens[0].token_id
        assert members[0].ordinal == 0
        assert members[1].token_id == tokens[1].token_id
        assert members[1].ordinal == 1

    def test_flush_with_audit(self) -> None:
        """Flush transitions batch and returns outputs."""
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import AggregationExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.engine.tokens import TokenInfo
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
        )

        # Mock aggregation that computes average
        class AvgAggregation:
            name = "avg_aggregator"
            node_id = agg_node.node_id
            _batch_id: str | None = None
            _values: list[float]

            def __init__(self) -> None:
                self._values = []

            def accept(self, row: dict[str, Any], ctx: PluginContext) -> AcceptResult:
                self._values.append(row["value"])
                return AcceptResult(accepted=True, trigger=len(self._values) >= 2)

            def flush(self, ctx: PluginContext) -> list[dict[str, Any]]:
                avg = sum(self._values) / len(self._values) if self._values else 0
                self._values = []
                return [{"average": avg}]

        aggregation = AvgAggregation()
        ctx = PluginContext(run_id=run.run_id, config={})
        executor = AggregationExecutor(recorder, SpanFactory(), run.run_id)

        # Accept two rows
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
            if result.trigger:
                batch_id = result.batch_id
                break

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

        # Verify aggregation._batch_id is reset
        assert aggregation._batch_id is None

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
        )

        class NoBatchAggregation:
            name = "no_batch"
            node_id = agg_node.node_id
            _batch_id: str | None = None

            def accept(self, row: dict[str, Any], ctx: PluginContext) -> AcceptResult:
                return AcceptResult(accepted=True, trigger=False)

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
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import AggregationExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.engine.tokens import TokenInfo
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
        )

        class ExplodingAggregation:
            name = "exploding_agg"
            node_id = agg_node.node_id
            _batch_id: str | None = None

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
        assert states[0].status == "failed"
        assert states[0].duration_ms is not None

    def test_flush_exception_marks_batch_failed(self) -> None:
        """Exception in flush() marks batch as failed."""
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import AggregationExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.engine.tokens import TokenInfo
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
        )

        class ExplodingFlushAggregation:
            name = "exploding_flush"
            node_id = agg_node.node_id
            _batch_id: str | None = None

            def accept(self, row: dict[str, Any], ctx: PluginContext) -> AcceptResult:
                return AcceptResult(accepted=True, trigger=True)

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
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import AggregationExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.engine.tokens import TokenInfo
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
        )

        class BatchCounterAggregation:
            name = "batch_counter"
            node_id = agg_node.node_id
            _batch_id: str | None = None
            _count: int = 0
            _batch_num: int = 0

            def accept(self, row: dict[str, Any], ctx: PluginContext) -> AcceptResult:
                self._count += 1
                return AcceptResult(accepted=True, trigger=self._count >= 2)

            def flush(self, ctx: PluginContext) -> list[dict[str, Any]]:
                self._batch_num += 1
                result = [{"batch": self._batch_num, "count": self._count}]
                self._count = 0
                return result

        aggregation = BatchCounterAggregation()
        ctx = PluginContext(run_id=run.run_id, config={})
        executor = AggregationExecutor(recorder, SpanFactory(), run.run_id)

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
            if result.trigger:
                batch_ids.append(result.batch_id)
                outputs = executor.flush(
                    aggregation=aggregation,
                    ctx=ctx,
                    trigger_reason="count_reached",
                    step_in_pipeline=1,
                )

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
