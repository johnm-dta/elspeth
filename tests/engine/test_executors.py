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
