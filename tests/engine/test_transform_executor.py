# tests/engine/test_transform_executor.py
"""Tests for transform executor."""

from typing import Any

import pytest

from elspeth.contracts import NodeStateCompleted, NodeStateFailed, NodeType
from elspeth.contracts.schema import SchemaConfig
from tests.conftest import as_transform

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
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # Mock transform plugin
        class DoubleTransform:
            name = "double"
            node_id = node.node_id

            def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
                return TransformResult.success({"value": row["value"] * 2}, success_reason={"action": "double"})

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
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        class FailingTransform:
            name = "failing"
            node_id = node.node_id
            _on_error = "discard"  # Required for transforms that return errors

            def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
                return TransformResult.error({"reason": "validation_failed", "message": "validation failed"})

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

        result, _, error_sink = executor.execute_transform(
            transform=as_transform(transform),
            token=token,
            ctx=ctx,
            step_in_pipeline=1,
        )

        assert result.status == "error"
        assert result.reason == {"reason": "validation_failed", "message": "validation failed"}

        # Verify audit trail records the error
        states = recorder.get_node_states_for_token(token.token_id)
        assert len(states) == 1
        state = states[0]
        assert state.status == "failed"
        assert isinstance(state, NodeStateFailed), "Failed state should be NodeStateFailed"
        assert state.duration_ms is not None

        # Verify transform error is recorded with correct attribution
        errors = recorder.get_transform_errors_for_token(token.token_id)
        assert len(errors) == 1
        assert errors[0].transform_id == node.node_id
        assert errors[0].destination == error_sink

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
            node_type=NodeType.TRANSFORM,
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
        # NodeStateFailed has duration_ms - access directly (no hasattr guards)
        assert isinstance(state, NodeStateFailed), "Failed state should be NodeStateFailed"
        assert state.duration_ms is not None

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
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        class EnrichTransform:
            name = "enricher"
            node_id = node.node_id

            def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
                return TransformResult.success({**row, "enriched": True}, success_reason={"action": "enrich"})

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
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        class IdentityTransform:
            name = "identity"
            node_id = node.node_id

            def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
                return TransformResult.success(row, success_reason={"action": "identity"})

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
        # NodeStateCompleted has input_hash and output_hash - access directly (no hasattr guards)
        assert isinstance(state, NodeStateCompleted), "Completed state should be NodeStateCompleted"
        assert state.input_hash is not None
        assert state.output_hash is not None
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
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        class DiscardingTransform:
            name = "discarding"
            node_id = node.node_id
            _on_error = "discard"

            def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
                return TransformResult.error({"reason": "invalid_input", "message": "invalid input"})

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

        # Verify audit trail records the error
        states = recorder.get_node_states_for_token(token.token_id)
        assert len(states) == 1
        state = states[0]
        assert state.status == "failed"
        assert isinstance(state, NodeStateFailed), "Failed state should be NodeStateFailed"
        assert state.duration_ms is not None

        # Verify transform error is recorded with correct destination
        errors = recorder.get_transform_errors_for_token(token.token_id)
        assert len(errors) == 1
        assert errors[0].transform_id == node.node_id
        assert errors[0].destination == "discard"

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
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        class ErrorRoutingTransform:
            name = "routing_to_error"
            node_id = node.node_id
            _on_error = "error_sink"  # Routes to named error sink

            def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
                return TransformResult.error({"reason": "validation_failed", "message": "routing to error sink"})

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
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        class SuccessfulTransform:
            name = "successful"
            node_id = node.node_id

            def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
                return TransformResult.success({"result": "ok"}, success_reason={"action": "test"})

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
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        class SimpleTransform:
            name = "attempt_test"
            node_id = node.node_id

            def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
                return TransformResult.success(row, success_reason={"action": "test"})

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

    def test_execute_transform_passes_context_after_to_recorder(self) -> None:
        """TransformResult.context_after should be passed to complete_node_state.

        P3-2026-02-02: Pooling metadata flows through context_after to the audit trail.
        This test verifies the executor passes context_after to the recorder.
        """
        import json

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
            plugin_name="pool_metadata",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # Transform that returns context_after (simulating pooled executor)
        pool_context = {
            "pool_config": {"pool_size": 4, "dispatch_delay_at_completion_ms": 50},
            "pool_stats": {"max_concurrent_reached": 4, "capacity_retries": 2},
            "query_ordering": [
                {"submit_index": 0, "complete_index": 1, "buffer_wait_ms": 15.3},
                {"submit_index": 1, "complete_index": 0, "buffer_wait_ms": 0.5},
            ],
        }

        class PooledTransform:
            name = "pool_metadata"
            node_id = node.node_id

            def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
                return TransformResult.success(
                    row,
                    success_reason={"action": "enriched"},
                    context_after=pool_context,
                )

        transform = PooledTransform()
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

        executor.execute_transform(
            transform=as_transform(transform),
            token=token,
            ctx=ctx,
            step_in_pipeline=1,
        )

        # Verify context_after was recorded in node state
        states = recorder.get_node_states_for_token(token.token_id)
        assert len(states) == 1
        state = states[0]
        assert isinstance(state, NodeStateCompleted)
        assert state.context_after_json is not None, (
            "context_after should be recorded in node state. "
            "TransformExecutor.execute_transform must pass context_after to complete_node_state."
        )

        # Verify the content
        context_after = json.loads(state.context_after_json)
        assert context_after["pool_config"]["pool_size"] == 4
        assert context_after["pool_stats"]["max_concurrent_reached"] == 4
        assert len(context_after["query_ordering"]) == 2


class TestTransformErrorIdRegression:
    """Regression tests for P2-2026-01-19-transform-errors-ambiguous-transform-id.

    Transform errors must be recorded with node_id (unique DAG identifier),
    not name (plugin type which can be reused multiple times).
    """

    def test_transform_error_uses_node_id_not_name(self) -> None:
        """Transform errors are attributed to node_id, not ambiguous plugin name.

        Bug: When a pipeline has two instances of the same plugin (e.g., two
        field_mappers), errors were recorded with transform.name which is the
        plugin type - making it impossible to determine which node failed.

        Fix: Use transform.node_id which is unique per DAG node.
        """
        from elspeth.contracts import TokenInfo
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import TransformExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import TransformResult

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        # Register TWO nodes with SAME plugin name but DIFFERENT node_ids
        # This simulates a pipeline with two instances of the same transform plugin
        node1 = recorder.register_node(
            run_id=run.run_id,
            plugin_name="field_mapper",  # Same name as node2!
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={"field": "email"},
            schema_config=DYNAMIC_SCHEMA,
        )
        node2 = recorder.register_node(
            run_id=run.run_id,
            plugin_name="field_mapper",  # Same name as node1!
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={"field": "phone"},
            schema_config=DYNAMIC_SCHEMA,
        )

        # Verify precondition: both have same name but different node_ids
        assert node1.node_id != node2.node_id
        shared_plugin_name = "field_mapper"

        # Create a transform that fails - using node2's identity
        class FailingFieldMapper:
            name = shared_plugin_name  # This is the plugin name (not unique)
            node_id = node2.node_id  # This is the unique DAG node ID
            _on_error = "error_sink"

            def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
                return TransformResult.error({"reason": "validation_failed", "error": "invalid phone format"})

        transform = FailingFieldMapper()
        ctx = PluginContext(run_id=run.run_id, config={}, landscape=recorder)
        executor = TransformExecutor(recorder, SpanFactory())

        token = TokenInfo(
            row_id="row-1",
            token_id="token-1",
            row_data={"phone": "invalid"},
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=node1.node_id,
            row_index=0,
            data=token.row_data,
            row_id=token.row_id,
        )
        recorder.create_token(row_id=row.row_id, token_id=token.token_id)

        # Execute the transform - this should record an error
        result, _, error_sink = executor.execute_transform(
            transform=as_transform(transform),
            token=token,
            ctx=ctx,
            step_in_pipeline=1,
        )

        assert result.status == "error"
        assert error_sink == "error_sink"

        # REGRESSION CHECK: Verify the recorded error uses node_id, not name
        errors = recorder.get_transform_errors_for_token(token.token_id)
        assert len(errors) == 1

        recorded_error = errors[0]
        # The transform_id should be the unique node_id, NOT the plugin name
        assert recorded_error.transform_id == node2.node_id, (
            f"Transform error should use node_id ({node2.node_id}) not name ({shared_plugin_name}). Got: {recorded_error.transform_id}"
        )
        # Explicitly verify it's NOT the ambiguous name
        assert recorded_error.transform_id != shared_plugin_name, (
            "Transform error should NOT use plugin name (ambiguous when same plugin used multiple times)"
        )


class TestTransformExecutorMaxWorkers:
    """Tests for max_workers concurrency limiting."""

    def test_max_workers_limits_batch_adapter_max_pending(self) -> None:
        """max_workers should limit max_pending passed to connect_output.

        When max_workers is configured lower than the transform's _pool_size,
        the executor should cap max_pending to enforce the concurrency limit.
        """
        from unittest.mock import MagicMock

        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import TransformExecutor
        from elspeth.engine.spans import SpanFactory

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        # Executor with max_workers=5 (lower than default pool_size of 30)
        executor = TransformExecutor(recorder, SpanFactory(), max_workers=5)

        # Mock transform with high pool_size
        mock_transform = MagicMock()
        mock_transform._pool_size = 30  # Higher than max_workers
        mock_transform.connect_output = MagicMock()
        # Simulate no adapter attached yet
        del mock_transform._executor_batch_adapter

        # Call _get_batch_adapter to trigger connect_output
        executor._get_batch_adapter(mock_transform)

        # Verify max_pending was capped to max_workers
        mock_transform.connect_output.assert_called_once()
        call_kwargs = mock_transform.connect_output.call_args.kwargs
        assert call_kwargs["max_pending"] == 5, f"max_pending should be capped to max_workers (5), got {call_kwargs['max_pending']}"

    def test_max_workers_none_uses_transform_pool_size(self) -> None:
        """Without max_workers, transform's _pool_size is used directly."""
        from unittest.mock import MagicMock

        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import TransformExecutor
        from elspeth.engine.spans import SpanFactory

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        # Executor with no max_workers limit
        executor = TransformExecutor(recorder, SpanFactory(), max_workers=None)

        # Mock transform with pool_size
        mock_transform = MagicMock()
        mock_transform._pool_size = 30
        mock_transform.connect_output = MagicMock()
        del mock_transform._executor_batch_adapter

        executor._get_batch_adapter(mock_transform)

        # Verify transform's pool_size is used
        call_kwargs = mock_transform.connect_output.call_args.kwargs
        assert call_kwargs["max_pending"] == 30

    def test_max_workers_higher_than_pool_size_uses_pool_size(self) -> None:
        """When max_workers > pool_size, use pool_size (no point exceeding it)."""
        from unittest.mock import MagicMock

        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import TransformExecutor
        from elspeth.engine.spans import SpanFactory

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        # Executor with high max_workers
        executor = TransformExecutor(recorder, SpanFactory(), max_workers=100)

        # Mock transform with lower pool_size
        mock_transform = MagicMock()
        mock_transform._pool_size = 20  # Lower than max_workers
        mock_transform.connect_output = MagicMock()
        del mock_transform._executor_batch_adapter

        executor._get_batch_adapter(mock_transform)

        # Verify pool_size is used (not max_workers)
        call_kwargs = mock_transform.connect_output.call_args.kwargs
        assert call_kwargs["max_pending"] == 20
