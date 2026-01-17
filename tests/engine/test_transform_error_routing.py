"""Tests for transform error routing in engine.

These tests verify that TransformResult.error() rows are routed correctly:
- To configured sink (on_error = "sink_name")
- Discarded silently (on_error = "discard")
- RuntimeError if no on_error configured

Transform bugs (exceptions) still crash - only explicit errors are routed.
"""

from typing import Any

import pytest

from elspeth.contracts import TokenInfo
from elspeth.contracts.schema import SchemaConfig
from elspeth.plugins.context import PluginContext, TransformErrorToken
from elspeth.plugins.results import TransformResult

# Dynamic schema for tests that don't care about specific fields
DYNAMIC_SCHEMA = SchemaConfig.from_dict({"fields": "dynamic"})


class MockTransform:
    """Mock transform for testing error routing."""

    name = "mock_transform"
    node_id: str | None = None

    def __init__(self, result: TransformResult, on_error: str | None = None) -> None:
        self._result = result
        self._on_error = on_error

    def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
        return self._result


class TestTransformErrorRouting:
    """Tests for routing TransformResult.error() rows."""

    @pytest.fixture
    def setup_landscape(self):
        """Set up LandscapeDB, recorder, and run for tests."""
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        return db, recorder, run

    def test_success_result_returns_row_unchanged(self, setup_landscape) -> None:
        """TransformResult.success() returns the transformed row normally."""
        from elspeth.engine.executors import TransformExecutor
        from elspeth.engine.spans import SpanFactory

        db, recorder, run = setup_landscape

        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="mock_transform",
            node_type="transform",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        transform = MockTransform(TransformResult.success({"value": 42}))
        transform.node_id = node.node_id

        ctx = PluginContext(
            run_id=run.run_id,
            config={},
            node_id=node.node_id,
            landscape=recorder,
        )
        executor = TransformExecutor(recorder, SpanFactory())

        token = TokenInfo(
            row_id="row-1",
            token_id="tok_123",
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

        result, updated_token, _ = executor.execute_transform(
            transform=transform,
            token=token,
            ctx=ctx,
            step_in_pipeline=1,
        )

        assert result.status == "success"
        assert result.row == {"value": 42}
        assert updated_token.row_data == {"value": 42}

    def test_error_result_with_on_error_routes_to_sink(self, setup_landscape) -> None:
        """TransformResult.error() with on_error routes to configured sink."""
        from elspeth.engine.executors import TransformExecutor
        from elspeth.engine.spans import SpanFactory

        db, recorder, run = setup_landscape

        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="mock_transform",
            node_type="transform",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        transform = MockTransform(
            TransformResult.error({"reason": "Cannot process"}),
            on_error="error_sink",
        )
        transform.node_id = node.node_id

        # Track route_to_sink calls
        routed: list[dict[str, Any]] = []

        ctx = PluginContext(
            run_id=run.run_id,
            config={},
            node_id=node.node_id,
            landscape=recorder,
        )
        # Override route_to_sink to track calls
        ctx.route_to_sink = lambda sink_name, row, metadata=None: routed.append(
            {"sink": sink_name, "row": row, "metadata": metadata}
        )

        executor = TransformExecutor(recorder, SpanFactory())

        token = TokenInfo(
            row_id="row-1",
            token_id="tok_123",
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

        result, _, _error_sink = executor.execute_transform(
            transform=transform,
            token=token,
            ctx=ctx,
            step_in_pipeline=1,
        )

        assert result.status == "error"
        assert len(routed) == 1
        assert routed[0]["sink"] == "error_sink"
        assert routed[0]["row"] == {"input": 1}

    def test_error_result_with_discard_does_not_route(self, setup_landscape) -> None:
        """TransformResult.error() with discard does NOT call route_to_sink."""
        from elspeth.engine.executors import TransformExecutor
        from elspeth.engine.spans import SpanFactory

        db, recorder, run = setup_landscape

        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="mock_transform",
            node_type="transform",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        transform = MockTransform(
            TransformResult.error({"reason": "Cannot process"}),
            on_error="discard",
        )
        transform.node_id = node.node_id

        routed: list[str] = []
        ctx = PluginContext(
            run_id=run.run_id,
            config={},
            node_id=node.node_id,
            landscape=recorder,
        )
        ctx.route_to_sink = lambda sink_name, row, metadata=None: routed.append(
            sink_name
        )

        executor = TransformExecutor(recorder, SpanFactory())

        token = TokenInfo(
            row_id="row-1",
            token_id="tok_123",
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

        result, _, _error_sink = executor.execute_transform(
            transform=transform,
            token=token,
            ctx=ctx,
            step_in_pipeline=1,
        )

        assert result.status == "error"
        assert routed == []  # Nothing routed for discard

    def test_error_without_on_error_raises_runtime_error(self, setup_landscape) -> None:
        """TransformResult.error() without on_error raises RuntimeError."""
        from elspeth.engine.executors import TransformExecutor
        from elspeth.engine.spans import SpanFactory

        db, recorder, run = setup_landscape

        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="mock_transform",
            node_type="transform",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        transform = MockTransform(
            TransformResult.error({"reason": "Cannot process"}),
            on_error=None,  # Not configured
        )
        transform.node_id = node.node_id

        ctx = PluginContext(
            run_id=run.run_id,
            config={},
            node_id=node.node_id,
            landscape=recorder,
        )
        executor = TransformExecutor(recorder, SpanFactory())

        token = TokenInfo(
            row_id="row-1",
            token_id="tok_123",
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

        with pytest.raises(RuntimeError, match="on_error"):
            executor.execute_transform(
                transform=transform,
                token=token,
                ctx=ctx,
                step_in_pipeline=1,
            )

    def test_error_event_recorded_for_sink_destination(self, setup_landscape) -> None:
        """record_transform_error called when routing to sink."""
        from elspeth.engine.executors import TransformExecutor
        from elspeth.engine.spans import SpanFactory

        db, recorder, run = setup_landscape

        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="mock_transform",
            node_type="transform",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        transform = MockTransform(
            TransformResult.error({"reason": "Test error"}),
            on_error="error_sink",
        )
        transform.node_id = node.node_id

        recorded: list[dict[str, Any]] = []

        def capture_record(
            token_id: str,
            transform_id: str,
            row: dict[str, Any],
            error_details: dict[str, Any],
            destination: str,
        ) -> TransformErrorToken:
            recorded.append(
                {
                    "token_id": token_id,
                    "transform_id": transform_id,
                    "row": row,
                    "error_details": error_details,
                    "destination": destination,
                }
            )
            return TransformErrorToken(
                token_id=token_id,
                transform_id=transform_id,
                destination=destination,
            )

        ctx = PluginContext(
            run_id=run.run_id,
            config={},
            node_id=node.node_id,
            landscape=recorder,
        )
        ctx.record_transform_error = capture_record
        ctx.route_to_sink = lambda sink_name, row, metadata=None: None

        executor = TransformExecutor(recorder, SpanFactory())

        token = TokenInfo(
            row_id="row-1",
            token_id="tok_123",
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

        executor.execute_transform(
            transform=transform,
            token=token,
            ctx=ctx,
            step_in_pipeline=1,
        )

        assert len(recorded) == 1
        assert recorded[0]["destination"] == "error_sink"
        assert recorded[0]["transform_id"] == "mock_transform"
        assert recorded[0]["token_id"] == "tok_123"
        assert recorded[0]["error_details"] == {"reason": "Test error"}

    def test_error_event_recorded_for_discard(self, setup_landscape) -> None:
        """record_transform_error called even when discarding."""
        from elspeth.engine.executors import TransformExecutor
        from elspeth.engine.spans import SpanFactory

        db, recorder, run = setup_landscape

        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="mock_transform",
            node_type="transform",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        transform = MockTransform(
            TransformResult.error({"reason": "Test error"}),
            on_error="discard",
        )
        transform.node_id = node.node_id

        recorded: list[dict[str, Any]] = []

        def capture_record(
            token_id: str,
            transform_id: str,
            row: dict[str, Any],
            error_details: dict[str, Any],
            destination: str,
        ) -> TransformErrorToken:
            recorded.append({"destination": destination})
            return TransformErrorToken(
                token_id=token_id,
                transform_id=transform_id,
                destination=destination,
            )

        ctx = PluginContext(
            run_id=run.run_id,
            config={},
            node_id=node.node_id,
            landscape=recorder,
        )
        ctx.record_transform_error = capture_record

        executor = TransformExecutor(recorder, SpanFactory())

        token = TokenInfo(
            row_id="row-1",
            token_id="tok_123",
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

        executor.execute_transform(
            transform=transform,
            token=token,
            ctx=ctx,
            step_in_pipeline=1,
        )

        # Error recorded even for discard
        assert len(recorded) == 1
        assert recorded[0]["destination"] == "discard"

    def test_exception_in_transform_propagates(self, setup_landscape) -> None:
        """Exception in transform propagates (does NOT route to on_error).

        This enforces CLAUDE.md's rule: bugs crash, they don't get silently routed.
        """
        from elspeth.engine.executors import TransformExecutor
        from elspeth.engine.spans import SpanFactory

        db, recorder, run = setup_landscape

        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="buggy",
            node_type="transform",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        class BuggyTransform:
            """Transform with a bug that raises an exception."""

            name = "buggy"
            node_id: str | None = node.node_id
            _on_error = "error_sink"  # Configured but should NOT be used for bugs

            def process(
                self, row: dict[str, Any], ctx: PluginContext
            ) -> TransformResult:
                raise KeyError("nonexistent_field")  # BUG!

        transform = BuggyTransform()
        ctx = PluginContext(
            run_id=run.run_id,
            config={},
            node_id=node.node_id,
            landscape=recorder,
        )
        executor = TransformExecutor(recorder, SpanFactory())

        token = TokenInfo(
            row_id="row-1",
            token_id="tok_123",
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

        # Exception propagates - not caught and routed
        with pytest.raises(KeyError):
            executor.execute_transform(
                transform=transform,
                token=token,
                ctx=ctx,
                step_in_pipeline=1,
            )

    def test_error_routing_preserves_original_row(self, setup_landscape) -> None:
        """Error routing sends the original input row, not None."""
        from elspeth.engine.executors import TransformExecutor
        from elspeth.engine.spans import SpanFactory

        db, recorder, run = setup_landscape

        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="mock_transform",
            node_type="transform",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        original_row = {"field1": "value1", "field2": 42, "nested": {"a": 1}}

        transform = MockTransform(
            TransformResult.error({"reason": "Validation failed"}),
            on_error="quarantine_sink",
        )
        transform.node_id = node.node_id

        routed_rows: list[dict[str, Any]] = []
        ctx = PluginContext(
            run_id=run.run_id,
            config={},
            node_id=node.node_id,
            landscape=recorder,
        )
        ctx.route_to_sink = lambda sink_name, row, metadata=None: routed_rows.append(
            row
        )

        executor = TransformExecutor(recorder, SpanFactory())

        token = TokenInfo(
            row_id="row-1",
            token_id="tok_123",
            row_data=original_row,
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

        assert len(routed_rows) == 1
        assert routed_rows[0] == original_row

    def test_error_metadata_includes_transform_error_details(
        self, setup_landscape
    ) -> None:
        """Routed error includes metadata with transform error reason."""
        from elspeth.engine.executors import TransformExecutor
        from elspeth.engine.spans import SpanFactory

        db, recorder, run = setup_landscape

        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="mock_transform",
            node_type="transform",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        error_reason = {"reason": "Division by zero", "field": "divisor", "value": 0}

        transform = MockTransform(
            TransformResult.error(error_reason),
            on_error="error_sink",
        )
        transform.node_id = node.node_id

        routed_metadata: list[dict[str, Any] | None] = []
        ctx = PluginContext(
            run_id=run.run_id,
            config={},
            node_id=node.node_id,
            landscape=recorder,
        )
        ctx.route_to_sink = (
            lambda sink_name, row, metadata=None: routed_metadata.append(metadata)
        )

        executor = TransformExecutor(recorder, SpanFactory())

        token = TokenInfo(
            row_id="row-1",
            token_id="tok_123",
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

        executor.execute_transform(
            transform=transform,
            token=token,
            ctx=ctx,
            step_in_pipeline=1,
        )

        assert len(routed_metadata) == 1
        assert routed_metadata[0] is not None
        assert "transform_error" in routed_metadata[0]
        assert routed_metadata[0]["transform_error"] == error_reason
