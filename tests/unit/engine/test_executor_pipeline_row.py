# tests/unit/engine/test_executor_pipeline_row.py
"""Tests for executor PipelineRow handling.

Tests that TransformExecutor:
1. Passes PipelineRow (not dict) to transform.process()
2. Extracts dict from PipelineRow for Landscape recording
3. Sets ctx.contract from token.row_data.contract
4. Creates new PipelineRow from result using correct contract
5. Crashes if no contract available (B6 fix)
"""

from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import pytest

from elspeth.contracts import TransformResult
from elspeth.contracts.identity import TokenInfo
from elspeth.contracts.schema_contract import PipelineRow, SchemaContract
from elspeth.testing import make_field, make_row
from tests.unit.engine.conftest import make_test_step_resolver as _make_step_resolver


def _make_contract() -> SchemaContract:
    """Create a simple test contract."""
    return SchemaContract(
        fields=(
            make_field(
                "value",
                python_type=str,
                original_name="'value'",
                required=True,
                source="declared",
            ),
        ),
        mode="FLEXIBLE",
        locked=True,
    )


def _make_output_contract() -> SchemaContract:
    """Create a contract for transform output (different from input)."""
    return SchemaContract(
        fields=(
            make_field(
                "value",
                python_type=str,
                original_name="'value'",
                required=True,
                source="declared",
            ),
            make_field(
                "processed",
                python_type=bool,
                original_name="'processed'",
                required=True,
                source="declared",
            ),
        ),
        mode="FLEXIBLE",
        locked=True,
    )


class TestTransformExecutorPipelineRow:
    """Tests for TransformExecutor with PipelineRow."""

    def test_execute_transform_passes_pipeline_row_to_plugin(self) -> None:
        """TransformExecutor should pass PipelineRow to transform.process()."""
        from elspeth.contracts.plugin_context import PluginContext
        from elspeth.engine.executors import TransformExecutor
        from elspeth.engine.spans import SpanFactory

        # Setup token with PipelineRow
        contract = _make_contract()
        row = make_row({"value": "test"}, contract=contract)
        token = TokenInfo(row_id="row_001", token_id="token_001", row_data=row)

        # Mock transform - configure spec to avoid MagicMock auto-creating 'accept' attr
        mock_transform = MagicMock()
        mock_transform.name = "test_transform"
        mock_transform.node_id = "transform_001"
        mock_transform.on_error = "discard"
        # Delete accept to prevent batch transform detection
        del mock_transform.accept
        mock_transform.process.return_value = TransformResult.success(
            make_row({"value": "processed"}, contract=contract),
            success_reason={"action": "test"},
        )

        # Mock recorder
        mock_recorder = MagicMock()
        mock_state = MagicMock()
        mock_state.state_id = "state_001"
        mock_recorder.begin_node_state.return_value = mock_state

        # Mock span factory - use nullcontext for proper context manager behavior
        mock_span_factory = MagicMock(spec=SpanFactory)
        mock_span_factory.transform_span.return_value = nullcontext()

        # Create executor
        executor = TransformExecutor(mock_recorder, mock_span_factory, _make_step_resolver())

        # Create context
        ctx = PluginContext(run_id="run_001", config={})

        # Execute
        _result, _updated_token, _error_sink = executor.execute_transform(
            transform=mock_transform,
            token=token,
            ctx=ctx,
        )

        # Verify PipelineRow was passed to transform.process()
        mock_transform.process.assert_called_once()
        call_args = mock_transform.process.call_args
        passed_row = call_args[0][0]
        assert isinstance(passed_row, PipelineRow), f"Expected PipelineRow, got {type(passed_row)}"
        assert passed_row["value"] == "test"

    def test_execute_transform_extracts_dict_for_landscape(self) -> None:
        """TransformExecutor should extract dict for Landscape recording."""
        from elspeth.contracts.plugin_context import PluginContext
        from elspeth.engine.executors import TransformExecutor
        from elspeth.engine.spans import SpanFactory

        # Setup token with PipelineRow
        contract = _make_contract()
        row = make_row({"value": "test"}, contract=contract)
        token = TokenInfo(row_id="row_001", token_id="token_001", row_data=row)

        # Mock transform - configure spec to avoid MagicMock auto-creating 'accept' attr
        mock_transform = MagicMock()
        mock_transform.name = "test_transform"
        mock_transform.node_id = "transform_001"
        mock_transform.on_error = "discard"
        # Delete accept to prevent batch transform detection
        del mock_transform.accept
        mock_transform.process.return_value = TransformResult.success(
            make_row({"value": "processed"}, contract=contract),
            success_reason={"action": "test"},
        )

        # Mock recorder - capture what gets passed
        mock_recorder = MagicMock()
        mock_state = MagicMock()
        mock_state.state_id = "state_001"
        mock_recorder.begin_node_state.return_value = mock_state

        # Mock span factory - use nullcontext for proper context manager behavior
        mock_span_factory = MagicMock(spec=SpanFactory)
        mock_span_factory.transform_span.return_value = nullcontext()

        # Create executor
        executor = TransformExecutor(mock_recorder, mock_span_factory, _make_step_resolver())

        # Create context
        ctx = PluginContext(run_id="run_001", config={})

        # Execute
        executor.execute_transform(
            transform=mock_transform,
            token=token,
            ctx=ctx,
        )

        # Verify dict was passed to begin_node_state (for Landscape recording)
        mock_recorder.begin_node_state.assert_called_once()
        call_kwargs = mock_recorder.begin_node_state.call_args[1]
        input_data = call_kwargs["input_data"]
        assert isinstance(input_data, dict), f"Expected dict for Landscape, got {type(input_data)}"
        assert input_data == {"value": "test"}

    def test_execute_transform_sets_ctx_contract(self) -> None:
        """TransformExecutor should set ctx.contract from token.row_data.contract."""
        from elspeth.contracts.plugin_context import PluginContext
        from elspeth.engine.executors import TransformExecutor
        from elspeth.engine.spans import SpanFactory

        # Setup token with PipelineRow
        contract = _make_contract()
        row = make_row({"value": "test"}, contract=contract)
        token = TokenInfo(row_id="row_001", token_id="token_001", row_data=row)

        # Mock transform - capture the context passed to process()
        captured_ctx = None

        def capture_ctx(row_data, ctx):
            nonlocal captured_ctx
            captured_ctx = ctx
            return TransformResult.success(
                make_row({"value": "processed"}, contract=contract),
                success_reason={"action": "test"},
            )

        mock_transform = MagicMock()
        mock_transform.name = "test_transform"
        mock_transform.node_id = "transform_001"
        mock_transform.on_error = "discard"
        # Delete accept to prevent batch transform detection
        del mock_transform.accept
        mock_transform.process = capture_ctx

        # Mock recorder
        mock_recorder = MagicMock()
        mock_state = MagicMock()
        mock_state.state_id = "state_001"
        mock_recorder.begin_node_state.return_value = mock_state

        # Mock span factory - use nullcontext for proper context manager behavior
        mock_span_factory = MagicMock(spec=SpanFactory)
        mock_span_factory.transform_span.return_value = nullcontext()

        # Create executor
        executor = TransformExecutor(mock_recorder, mock_span_factory, _make_step_resolver())

        # Create context - initially no contract
        ctx = PluginContext(run_id="run_001", config={})
        assert ctx.contract is None

        # Execute
        executor.execute_transform(
            transform=mock_transform,
            token=token,
            ctx=ctx,
        )

        # Verify ctx.contract was set from token.row_data.contract
        assert captured_ctx is not None
        assert captured_ctx.contract is contract

    def test_execute_transform_creates_pipeline_row_from_result(self) -> None:
        """TransformExecutor should create PipelineRow from result dict + contract."""
        from elspeth.contracts.plugin_context import PluginContext
        from elspeth.engine.executors import TransformExecutor
        from elspeth.engine.spans import SpanFactory

        # Setup token with PipelineRow
        input_contract = _make_contract()
        output_contract = _make_output_contract()
        row = make_row({"value": "test"}, contract=input_contract)
        token = TokenInfo(row_id="row_001", token_id="token_001", row_data=row)

        # Mock transform - returns different contract on output
        mock_transform = MagicMock()
        mock_transform.name = "test_transform"
        mock_transform.node_id = "transform_001"
        mock_transform.on_error = "discard"
        # Delete accept to prevent batch transform detection
        del mock_transform.accept
        mock_transform.process.return_value = TransformResult.success(
            make_row({"value": "processed", "processed": True}, contract=output_contract),
            success_reason={"action": "test"},
        )

        # Mock recorder
        mock_recorder = MagicMock()
        mock_state = MagicMock()
        mock_state.state_id = "state_001"
        mock_recorder.begin_node_state.return_value = mock_state

        # Mock span factory - use nullcontext for proper context manager behavior
        mock_span_factory = MagicMock(spec=SpanFactory)
        mock_span_factory.transform_span.return_value = nullcontext()

        # Create executor
        executor = TransformExecutor(mock_recorder, mock_span_factory, _make_step_resolver())

        # Create context
        ctx = PluginContext(run_id="run_001", config={})

        # Execute
        _result, updated_token, _error_sink = executor.execute_transform(
            transform=mock_transform,
            token=token,
            ctx=ctx,
        )

        # Verify updated token has PipelineRow with output contract
        assert isinstance(updated_token.row_data, PipelineRow)
        assert updated_token.row_data["value"] == "processed"
        assert updated_token.row_data["processed"] is True
        assert updated_token.row_data.contract is output_contract

    def test_execute_transform_error_preserves_token(self) -> None:
        """When transform returns error, token should be unchanged."""
        from elspeth.contracts.plugin_context import PluginContext
        from elspeth.engine.executors import TransformExecutor
        from elspeth.engine.spans import SpanFactory

        # Setup token with PipelineRow
        contract = _make_contract()
        row = make_row({"value": "test"}, contract=contract)
        token = TokenInfo(row_id="row_001", token_id="token_001", row_data=row)

        # Mock transform - returns error
        mock_transform = MagicMock()
        mock_transform.name = "test_transform"
        mock_transform.node_id = "transform_001"
        mock_transform.on_error = "discard"  # Has error handler
        # Delete accept to prevent batch transform detection
        del mock_transform.accept
        mock_transform.process.return_value = TransformResult.error(
            reason={"reason": "test_error"},
        )

        # Mock recorder
        mock_recorder = MagicMock()
        mock_state = MagicMock()
        mock_state.state_id = "state_001"
        mock_recorder.begin_node_state.return_value = mock_state

        # Mock span factory - use nullcontext for proper context manager behavior
        mock_span_factory = MagicMock(spec=SpanFactory)
        mock_span_factory.transform_span.return_value = nullcontext()

        # Create executor
        executor = TransformExecutor(mock_recorder, mock_span_factory, _make_step_resolver())

        # Create context
        ctx = PluginContext(run_id="run_001", config={})

        # Execute
        _result, updated_token, _error_sink = executor.execute_transform(
            transform=mock_transform,
            token=token,
            ctx=ctx,
        )

        # Verify token is unchanged on error
        assert updated_token is token
        assert updated_token.row_data is row

    def test_execute_transform_hashes_dict_not_pipeline_row(self) -> None:
        """stable_hash should be called with dict, not PipelineRow."""
        from elspeth.contracts.plugin_context import PluginContext
        from elspeth.engine.executors import TransformExecutor
        from elspeth.engine.spans import SpanFactory

        # Setup token with PipelineRow
        contract = _make_contract()
        row = make_row({"value": "test"}, contract=contract)
        token = TokenInfo(row_id="row_001", token_id="token_001", row_data=row)

        # Mock transform - configure spec to avoid MagicMock auto-creating 'accept' attr
        mock_transform = MagicMock()
        mock_transform.name = "test_transform"
        mock_transform.node_id = "transform_001"
        mock_transform.on_error = "discard"
        # Delete accept to prevent batch transform detection
        del mock_transform.accept
        mock_transform.process.return_value = TransformResult.success(
            make_row({"value": "processed"}, contract=contract),
            success_reason={"action": "test"},
        )

        # Mock recorder
        mock_recorder = MagicMock()
        mock_state = MagicMock()
        mock_state.state_id = "state_001"
        mock_recorder.begin_node_state.return_value = mock_state

        # Mock span factory - use nullcontext for proper context manager behavior
        mock_span_factory = MagicMock(spec=SpanFactory)
        mock_span_factory.transform_span.return_value = nullcontext()

        # Create executor
        executor = TransformExecutor(mock_recorder, mock_span_factory, _make_step_resolver())

        # Create context
        ctx = PluginContext(run_id="run_001", config={})

        # Patch stable_hash to verify what gets passed
        with patch("elspeth.engine.executors.transform.stable_hash") as mock_hash:
            mock_hash.return_value = "test_hash"

            executor.execute_transform(
                transform=mock_transform,
                token=token,
                ctx=ctx,
            )

            # First call should be for input hash - should receive dict
            first_call_arg = mock_hash.call_args_list[0][0][0]
            assert isinstance(first_call_arg, dict), f"stable_hash should receive dict, got {type(first_call_arg)}"


class TestSinkExecutorPipelineRow:
    """Tests for SinkExecutor with PipelineRow.

    Sinks receive list[dict], NOT list[PipelineRow]. The SinkExecutor must
    extract dicts from PipelineRow before calling sink.write().

    Contract metadata is preserved in Landscape audit trail, not sink output.
    """

    def test_execute_sink_extracts_dicts_from_pipeline_rows(self) -> None:
        """SinkExecutor should extract dicts before calling sink.write()."""
        from elspeth.contracts import ArtifactDescriptor, PendingOutcome, RowOutcome
        from elspeth.contracts.plugin_context import PluginContext
        from elspeth.engine.executors import SinkExecutor
        from elspeth.engine.spans import SpanFactory

        # Setup tokens with PipelineRow
        contract = _make_contract()
        tokens = [
            TokenInfo(row_id="r1", token_id="t1", row_data=make_row({"value": "a"}, contract=contract)),
            TokenInfo(row_id="r2", token_id="t2", row_data=make_row({"value": "b"}, contract=contract)),
        ]

        # Mock sink - capture what gets passed to write()
        mock_sink = MagicMock()
        mock_sink.name = "test_sink"
        mock_sink.node_id = "sink_001"
        mock_sink.write.return_value = ArtifactDescriptor(
            artifact_type="file",
            path_or_uri="/output/test.csv",
            content_hash="abc123",
            size_bytes=100,
        )

        # Mock recorder
        mock_recorder = MagicMock()
        mock_state = MagicMock()
        mock_state.state_id = "state_001"
        mock_recorder.begin_node_state.return_value = mock_state
        mock_artifact = MagicMock()
        mock_recorder.register_artifact.return_value = mock_artifact

        # Mock span factory
        mock_span_factory = MagicMock(spec=SpanFactory)
        mock_span_factory.sink_span.return_value = nullcontext()

        # Create executor
        executor = SinkExecutor(mock_recorder, mock_span_factory, run_id="run_001")

        # Create context
        ctx = PluginContext(run_id="run_001", config={})

        # Execute
        executor.write(
            sink=mock_sink,
            tokens=tokens,
            ctx=ctx,
            step_in_pipeline=5,
            sink_name="test_sink",
            pending_outcome=PendingOutcome(outcome=RowOutcome.COMPLETED),
        )

        # Verify dicts (not PipelineRow) were passed to sink.write()
        mock_sink.write.assert_called_once()
        call_args = mock_sink.write.call_args
        rows = call_args[0][0]
        assert len(rows) == 2
        assert all(isinstance(r, dict) for r in rows), f"Expected dicts, got {[type(r) for r in rows]}"
        assert all(not isinstance(r, PipelineRow) for r in rows), "Sink should not receive PipelineRow objects"
        assert rows == [{"value": "a"}, {"value": "b"}]

    def test_execute_sink_extracts_dict_for_landscape_input(self) -> None:
        """SinkExecutor should extract dict for Landscape input_data recording."""
        from elspeth.contracts import ArtifactDescriptor, PendingOutcome, RowOutcome
        from elspeth.contracts.plugin_context import PluginContext
        from elspeth.engine.executors import SinkExecutor
        from elspeth.engine.spans import SpanFactory

        # Setup token with PipelineRow
        contract = _make_contract()
        token = TokenInfo(row_id="r1", token_id="t1", row_data=make_row({"value": "test"}, contract=contract))

        # Mock sink
        mock_sink = MagicMock()
        mock_sink.name = "test_sink"
        mock_sink.node_id = "sink_001"
        mock_sink.write.return_value = ArtifactDescriptor(
            artifact_type="file",
            path_or_uri="/output/test.csv",
            content_hash="abc123",
            size_bytes=100,
        )

        # Mock recorder - capture what gets passed
        mock_recorder = MagicMock()
        mock_state = MagicMock()
        mock_state.state_id = "state_001"
        mock_recorder.begin_node_state.return_value = mock_state
        mock_artifact = MagicMock()
        mock_recorder.register_artifact.return_value = mock_artifact

        # Mock span factory
        mock_span_factory = MagicMock(spec=SpanFactory)
        mock_span_factory.sink_span.return_value = nullcontext()

        # Create executor
        executor = SinkExecutor(mock_recorder, mock_span_factory, run_id="run_001")

        # Create context
        ctx = PluginContext(run_id="run_001", config={})

        # Execute
        executor.write(
            sink=mock_sink,
            tokens=[token],
            ctx=ctx,
            step_in_pipeline=5,
            sink_name="test_sink",
            pending_outcome=PendingOutcome(outcome=RowOutcome.COMPLETED),
        )

        # Verify dict was passed to begin_node_state (for Landscape recording)
        mock_recorder.begin_node_state.assert_called_once()
        call_kwargs = mock_recorder.begin_node_state.call_args[1]
        input_data = call_kwargs["input_data"]
        assert isinstance(input_data, dict), f"Expected dict for Landscape input_data, got {type(input_data)}"
        # PipelineRow subclasses dict, so type() check is needed, not isinstance()
        assert type(input_data) is not PipelineRow, "Landscape should not receive PipelineRow objects"  # type: ignore[comparison-overlap, unreachable]
        assert input_data == {"value": "test"}  # type: ignore[unreachable]

    def test_execute_sink_extracts_dict_for_landscape_output(self) -> None:
        """SinkExecutor should extract dict for Landscape output_data recording."""
        from elspeth.contracts import ArtifactDescriptor, PendingOutcome, RowOutcome
        from elspeth.contracts.plugin_context import PluginContext
        from elspeth.engine.executors import SinkExecutor
        from elspeth.engine.spans import SpanFactory

        # Setup token with PipelineRow
        contract = _make_contract()
        token = TokenInfo(row_id="r1", token_id="t1", row_data=make_row({"value": "test"}, contract=contract))

        # Mock sink
        mock_sink = MagicMock()
        mock_sink.name = "test_sink"
        mock_sink.node_id = "sink_001"
        mock_sink.write.return_value = ArtifactDescriptor(
            artifact_type="file",
            path_or_uri="/output/test.csv",
            content_hash="abc123",
            size_bytes=100,
        )

        # Mock recorder - capture what gets passed to complete_node_state
        mock_recorder = MagicMock()
        mock_state = MagicMock()
        mock_state.state_id = "state_001"
        mock_recorder.begin_node_state.return_value = mock_state
        mock_artifact = MagicMock()
        mock_recorder.register_artifact.return_value = mock_artifact

        # Mock span factory
        mock_span_factory = MagicMock(spec=SpanFactory)
        mock_span_factory.sink_span.return_value = nullcontext()

        # Create executor
        executor = SinkExecutor(mock_recorder, mock_span_factory, run_id="run_001")

        # Create context
        ctx = PluginContext(run_id="run_001", config={})

        # Execute
        executor.write(
            sink=mock_sink,
            tokens=[token],
            ctx=ctx,
            step_in_pipeline=5,
            sink_name="test_sink",
            pending_outcome=PendingOutcome(outcome=RowOutcome.COMPLETED),
        )

        # Verify dict was passed to complete_node_state output_data
        # complete_node_state is called once for successful sink write
        complete_calls = [c for c in mock_recorder.complete_node_state.call_args_list if c[1].get("output_data") is not None]
        assert len(complete_calls) == 1
        output_data = complete_calls[0][1]["output_data"]
        row_in_output = output_data["row"]
        assert isinstance(row_in_output, dict), f"Expected dict in output_data['row'], got {type(row_in_output)}"
        # PipelineRow subclasses dict, so type() check is needed, not isinstance()
        assert type(row_in_output) is not PipelineRow, "Landscape output should not contain PipelineRow"  # type: ignore[comparison-overlap, unreachable]
        assert row_in_output == {"value": "test"}  # type: ignore[unreachable]

    def test_execute_sink_preserves_all_fields_in_dict(self) -> None:
        """Sink should receive all fields, including extras not in contract."""
        from elspeth.contracts import ArtifactDescriptor, PendingOutcome, RowOutcome
        from elspeth.contracts.plugin_context import PluginContext
        from elspeth.engine.executors import SinkExecutor
        from elspeth.engine.spans import SpanFactory

        # Setup token with PipelineRow that has extra fields not in contract
        contract = _make_contract()  # Only declares 'value' field
        # PipelineRow allows extra fields in FLEXIBLE mode
        row_data = {"value": "test", "extra_field": "extra_value", "another": 123}
        token = TokenInfo(row_id="r1", token_id="t1", row_data=make_row(row_data, contract=contract))

        # Mock sink - capture what gets passed
        mock_sink = MagicMock()
        mock_sink.name = "test_sink"
        mock_sink.node_id = "sink_001"
        mock_sink.write.return_value = ArtifactDescriptor(
            artifact_type="file",
            path_or_uri="/output/test.csv",
            content_hash="abc123",
            size_bytes=100,
        )

        # Mock recorder
        mock_recorder = MagicMock()
        mock_state = MagicMock()
        mock_state.state_id = "state_001"
        mock_recorder.begin_node_state.return_value = mock_state
        mock_artifact = MagicMock()
        mock_recorder.register_artifact.return_value = mock_artifact

        # Mock span factory
        mock_span_factory = MagicMock(spec=SpanFactory)
        mock_span_factory.sink_span.return_value = nullcontext()

        # Create executor
        executor = SinkExecutor(mock_recorder, mock_span_factory, run_id="run_001")

        # Create context
        ctx = PluginContext(run_id="run_001", config={})

        # Execute
        executor.write(
            sink=mock_sink,
            tokens=[token],
            ctx=ctx,
            step_in_pipeline=5,
            sink_name="test_sink",
            pending_outcome=PendingOutcome(outcome=RowOutcome.COMPLETED),
        )

        # Verify ALL fields are passed to sink, not just contract fields
        mock_sink.write.assert_called_once()
        call_args = mock_sink.write.call_args
        rows = call_args[0][0]
        assert len(rows) == 1
        assert rows[0] == {"value": "test", "extra_field": "extra_value", "another": 123}

    def test_execute_sink_updates_ctx_contract_from_tokens(self) -> None:
        """SinkExecutor should synchronize ctx.contract to sink token contracts."""
        from elspeth.contracts import ArtifactDescriptor, PendingOutcome, RowOutcome
        from elspeth.contracts.plugin_context import PluginContext
        from elspeth.engine.executors import SinkExecutor
        from elspeth.engine.spans import SpanFactory

        stale_contract = _make_contract()
        sink_contract = SchemaContract(
            fields=(
                make_field(
                    "value",
                    python_type=str,
                    original_name="Value Renamed",
                    required=True,
                    source="declared",
                ),
            ),
            mode="FLEXIBLE",
            locked=True,
        )
        token = TokenInfo(row_id="r1", token_id="t1", row_data=make_row({"value": "test"}, contract=sink_contract))

        mock_sink = MagicMock()
        mock_sink.name = "test_sink"
        mock_sink.node_id = "sink_001"

        captured_contract = None

        def capture_write(_rows, call_ctx):
            nonlocal captured_contract
            captured_contract = call_ctx.contract
            return ArtifactDescriptor(
                artifact_type="file",
                path_or_uri="/output/test.csv",
                content_hash="abc123",
                size_bytes=100,
            )

        mock_sink.write.side_effect = capture_write

        mock_recorder = MagicMock()
        mock_state = MagicMock()
        mock_state.state_id = "state_001"
        mock_recorder.begin_node_state.return_value = mock_state
        mock_recorder.register_artifact.return_value = MagicMock()

        mock_span_factory = MagicMock(spec=SpanFactory)
        mock_span_factory.sink_span.return_value = nullcontext()

        executor = SinkExecutor(mock_recorder, mock_span_factory, run_id="run_001")

        ctx = PluginContext(run_id="run_001", config={})
        ctx.contract = stale_contract

        executor.write(
            sink=mock_sink,
            tokens=[token],
            ctx=ctx,
            step_in_pipeline=5,
            sink_name="test_sink",
            pending_outcome=PendingOutcome(outcome=RowOutcome.COMPLETED),
        )

        assert captured_contract is sink_contract
        assert ctx.contract is sink_contract

    def test_execute_sink_merges_mixed_token_contracts_for_context(self) -> None:
        """SinkExecutor should merge mixed token contracts before sink.write()."""
        from elspeth.contracts import ArtifactDescriptor, PendingOutcome, RowOutcome
        from elspeth.contracts.plugin_context import PluginContext
        from elspeth.engine.executors import SinkExecutor
        from elspeth.engine.spans import SpanFactory

        contract_a = SchemaContract(
            fields=(
                make_field(
                    "value",
                    python_type=str,
                    original_name="Value",
                    required=True,
                    source="declared",
                ),
            ),
            mode="FLEXIBLE",
            locked=True,
        )
        contract_b = SchemaContract(
            fields=(
                make_field(
                    "value",
                    python_type=str,
                    original_name="Value",
                    required=True,
                    source="declared",
                ),
                make_field(
                    "extra",
                    python_type=str,
                    original_name="Extra Field",
                    required=False,
                    source="inferred",
                ),
            ),
            mode="FLEXIBLE",
            locked=True,
        )
        expected_merged = contract_a.merge(contract_b)

        tokens = [
            TokenInfo(row_id="r1", token_id="t1", row_data=make_row({"value": "a"}, contract=contract_a)),
            TokenInfo(row_id="r2", token_id="t2", row_data=make_row({"value": "b", "extra": "x"}, contract=contract_b)),
        ]

        mock_sink = MagicMock()
        mock_sink.name = "test_sink"
        mock_sink.node_id = "sink_001"

        captured_contract = None

        def capture_write(_rows, call_ctx):
            nonlocal captured_contract
            captured_contract = call_ctx.contract
            return ArtifactDescriptor(
                artifact_type="file",
                path_or_uri="/output/test.csv",
                content_hash="abc123",
                size_bytes=100,
            )

        mock_sink.write.side_effect = capture_write

        mock_recorder = MagicMock()
        mock_state = MagicMock()
        mock_state.state_id = "state_001"
        mock_recorder.begin_node_state.return_value = mock_state
        mock_recorder.register_artifact.return_value = MagicMock()

        mock_span_factory = MagicMock(spec=SpanFactory)
        mock_span_factory.sink_span.return_value = nullcontext()

        executor = SinkExecutor(mock_recorder, mock_span_factory, run_id="run_001")

        ctx = PluginContext(run_id="run_001", config={})
        ctx.contract = _make_contract()

        executor.write(
            sink=mock_sink,
            tokens=tokens,
            ctx=ctx,
            step_in_pipeline=5,
            sink_name="test_sink",
            pending_outcome=PendingOutcome(outcome=RowOutcome.COMPLETED),
        )

        assert captured_contract is not None
        assert captured_contract == expected_merged
        assert captured_contract.get_field("extra") is not None
        assert ctx.contract == expected_merged


class TestAggregationExecutorPipelineRow:
    """Tests for AggregationExecutor with PipelineRow.

    AggregationExecutor buffers rows for batch aggregations. The buffer stores
    dicts (JSON-serializable for checkpoints), not PipelineRow objects.
    """

    def test_buffer_row_stores_dict_not_pipeline_row(self) -> None:
        """AggregationExecutor should store dicts in buffer, not PipelineRow.

        Internal buffer must be JSON-serializable for checkpoints.
        """
        from elspeth.contracts.types import NodeID
        from elspeth.core.config import AggregationSettings, TriggerConfig
        from elspeth.engine.executors import AggregationExecutor
        from elspeth.engine.spans import SpanFactory

        # Setup token with PipelineRow
        contract = _make_contract()
        row = make_row({"value": "test"}, contract=contract)
        token = TokenInfo(row_id="row_001", token_id="token_001", row_data=row)

        # Mock recorder
        mock_recorder = MagicMock()
        mock_batch = MagicMock()
        mock_batch.batch_id = "batch_001"
        mock_recorder.create_batch.return_value = mock_batch

        # Mock span factory
        mock_span_factory = MagicMock(spec=SpanFactory)

        # Create aggregation settings with a count trigger
        agg_settings = AggregationSettings(
            name="test_agg",
            plugin="batch_stats",
            input="default",
            trigger=TriggerConfig(count=10),
        )
        node_id = NodeID("agg_001")

        # Create executor with aggregation_settings
        executor = AggregationExecutor(
            mock_recorder,
            mock_span_factory,
            _make_step_resolver(),
            run_id="run_001",
            aggregation_settings={node_id: agg_settings},
        )

        # Buffer the row
        executor.buffer_row(node_id, token)

        # Verify buffer contains dict, NOT PipelineRow
        buffered = executor.get_buffered_rows(node_id)
        assert len(buffered) == 1
        assert isinstance(buffered[0], dict), f"Expected dict in buffer, got {type(buffered[0])}"
        # PipelineRow subclasses dict, so type() check is needed, not isinstance()
        assert type(buffered[0]) is not PipelineRow, "Buffer should not contain PipelineRow"  # type: ignore[comparison-overlap, unreachable]
        assert buffered[0] == {"value": "test"}  # type: ignore[unreachable]

    def test_buffer_row_extracts_dict_from_pipeline_row(self) -> None:
        """buffer_row should call to_dict() on token.row_data."""
        from elspeth.contracts.types import NodeID
        from elspeth.core.config import AggregationSettings, TriggerConfig
        from elspeth.engine.executors import AggregationExecutor
        from elspeth.engine.spans import SpanFactory

        # Setup token with PipelineRow containing extra fields
        contract = _make_contract()
        row_data = {"value": "test", "extra": "field", "number": 42}
        row = make_row(row_data, contract=contract)
        token = TokenInfo(row_id="row_001", token_id="token_001", row_data=row)

        # Mock recorder
        mock_recorder = MagicMock()
        mock_batch = MagicMock()
        mock_batch.batch_id = "batch_001"
        mock_recorder.create_batch.return_value = mock_batch

        # Mock span factory
        mock_span_factory = MagicMock(spec=SpanFactory)

        # Create aggregation settings
        agg_settings = AggregationSettings(
            name="test_agg",
            plugin="batch_stats",
            input="default",
            trigger=TriggerConfig(count=10),
        )
        node_id = NodeID("agg_001")

        # Create executor
        executor = AggregationExecutor(
            mock_recorder,
            mock_span_factory,
            _make_step_resolver(),
            run_id="run_001",
            aggregation_settings={node_id: agg_settings},
        )

        # Buffer the row
        executor.buffer_row(node_id, token)

        # Verify all fields are preserved in the buffer
        buffered = executor.get_buffered_rows(node_id)
        assert buffered[0] == {"value": "test", "extra": "field", "number": 42}

    def test_buffer_tokens_preserves_pipeline_row(self) -> None:
        """TokenInfo in buffer_tokens should keep PipelineRow as row_data."""
        from elspeth.contracts.types import NodeID
        from elspeth.core.config import AggregationSettings, TriggerConfig
        from elspeth.engine.executors import AggregationExecutor
        from elspeth.engine.spans import SpanFactory

        # Setup token with PipelineRow
        contract = _make_contract()
        row = make_row({"value": "test"}, contract=contract)
        token = TokenInfo(row_id="row_001", token_id="token_001", row_data=row)

        # Mock recorder
        mock_recorder = MagicMock()
        mock_batch = MagicMock()
        mock_batch.batch_id = "batch_001"
        mock_recorder.create_batch.return_value = mock_batch

        # Mock span factory
        mock_span_factory = MagicMock(spec=SpanFactory)

        # Create aggregation settings
        agg_settings = AggregationSettings(
            name="test_agg",
            plugin="batch_stats",
            input="default",
            trigger=TriggerConfig(count=10),
        )
        node_id = NodeID("agg_001")

        # Create executor
        executor = AggregationExecutor(
            mock_recorder,
            mock_span_factory,
            _make_step_resolver(),
            run_id="run_001",
            aggregation_settings={node_id: agg_settings},
        )

        # Buffer the row
        executor.buffer_row(node_id, token)

        # Verify buffer_tokens preserves TokenInfo with PipelineRow
        buffered_tokens = executor.get_buffered_tokens(node_id)
        assert len(buffered_tokens) == 1
        assert isinstance(buffered_tokens[0].row_data, PipelineRow)
        assert buffered_tokens[0].row_data.contract is contract

    def test_checkpoint_contains_dicts_not_pipeline_row(self) -> None:
        """get_checkpoint_state() should return JSON-serializable dicts."""
        import json

        from elspeth.contracts.types import NodeID
        from elspeth.core.config import AggregationSettings, TriggerConfig
        from elspeth.engine.executors import AggregationExecutor
        from elspeth.engine.spans import SpanFactory

        # Setup token with PipelineRow
        contract = _make_contract()
        row = make_row({"value": "test"}, contract=contract)
        token = TokenInfo(row_id="row_001", token_id="token_001", row_data=row)

        # Mock recorder
        mock_recorder = MagicMock()
        mock_batch = MagicMock()
        mock_batch.batch_id = "batch_001"
        mock_recorder.create_batch.return_value = mock_batch

        # Mock span factory
        mock_span_factory = MagicMock(spec=SpanFactory)

        # Create aggregation settings
        agg_settings = AggregationSettings(
            name="test_agg",
            plugin="batch_stats",
            input="default",
            trigger=TriggerConfig(count=10),
        )
        node_id = NodeID("agg_001")

        # Create executor
        executor = AggregationExecutor(
            mock_recorder,
            mock_span_factory,
            _make_step_resolver(),
            run_id="run_001",
            aggregation_settings={node_id: agg_settings},
        )

        # Buffer the row
        executor.buffer_row(node_id, token)

        # Get checkpoint state
        checkpoint = executor.get_checkpoint_state()

        # Verify checkpoint is JSON-serializable
        try:
            serialized = json.dumps(checkpoint)
            assert len(serialized) > 0
        except (TypeError, ValueError) as e:
            pytest.fail(f"Checkpoint should be JSON-serializable but got error: {e}")

        # Verify row_data is stored as dict in checkpoint
        node_checkpoint = checkpoint[str(node_id)]
        assert "tokens" in node_checkpoint
        token_data = node_checkpoint["tokens"][0]
        assert "row_data" in token_data
        # row_data should be a dict, not PipelineRow
        assert isinstance(token_data["row_data"], dict)
        assert token_data["row_data"] == {"value": "test"}

    def test_checkpoint_includes_contract_for_restore(self) -> None:
        """Checkpoint should include contract info to enable PipelineRow restoration."""
        from elspeth.contracts.types import NodeID
        from elspeth.core.config import AggregationSettings, TriggerConfig
        from elspeth.engine.executors import AggregationExecutor
        from elspeth.engine.spans import SpanFactory

        # Setup token with PipelineRow
        contract = _make_contract()
        row = make_row({"value": "test"}, contract=contract)
        token = TokenInfo(row_id="row_001", token_id="token_001", row_data=row)

        # Mock recorder
        mock_recorder = MagicMock()
        mock_batch = MagicMock()
        mock_batch.batch_id = "batch_001"
        mock_recorder.create_batch.return_value = mock_batch

        # Mock span factory
        mock_span_factory = MagicMock(spec=SpanFactory)

        # Create aggregation settings
        agg_settings = AggregationSettings(
            name="test_agg",
            plugin="batch_stats",
            input="default",
            trigger=TriggerConfig(count=10),
        )
        node_id = NodeID("agg_001")

        # Create executor
        executor = AggregationExecutor(
            mock_recorder,
            mock_span_factory,
            _make_step_resolver(),
            run_id="run_001",
            aggregation_settings={node_id: agg_settings},
        )

        # Buffer the row
        executor.buffer_row(node_id, token)

        # Get checkpoint state
        checkpoint = executor.get_checkpoint_state()

        # Verify contract info is stored (either per-token or per-node)
        node_checkpoint = checkpoint[str(node_id)]
        # Contract should be stored somewhere in checkpoint
        # Either as "contract" at node level or "contract_version" per token
        assert "contract" in node_checkpoint or any("contract_version" in t for t in node_checkpoint["tokens"]), (
            "Checkpoint must include contract info for PipelineRow restoration"
        )

    def test_restore_from_checkpoint_creates_pipeline_row(self) -> None:
        """restore_from_checkpoint should reconstruct TokenInfo with PipelineRow."""
        from elspeth.contracts.types import NodeID
        from elspeth.core.config import AggregationSettings, TriggerConfig
        from elspeth.engine.executors import AggregationExecutor
        from elspeth.engine.spans import SpanFactory

        # Setup contract and token
        contract = _make_contract()
        row = make_row({"value": "test"}, contract=contract)
        token = TokenInfo(row_id="row_001", token_id="token_001", row_data=row)

        # Mock recorder
        mock_recorder = MagicMock()
        mock_batch = MagicMock()
        mock_batch.batch_id = "batch_001"
        mock_recorder.create_batch.return_value = mock_batch

        # Mock span factory
        mock_span_factory = MagicMock(spec=SpanFactory)

        # Create aggregation settings
        agg_settings = AggregationSettings(
            name="test_agg",
            plugin="batch_stats",
            input="default",
            trigger=TriggerConfig(count=10),
        )
        node_id = NodeID("agg_001")

        # Create executor and buffer row
        executor = AggregationExecutor(
            mock_recorder,
            mock_span_factory,
            _make_step_resolver(),
            run_id="run_001",
            aggregation_settings={node_id: agg_settings},
        )
        executor.buffer_row(node_id, token)

        # Get checkpoint
        checkpoint = executor.get_checkpoint_state()

        # Create new executor and restore from checkpoint
        new_executor = AggregationExecutor(
            mock_recorder,
            mock_span_factory,
            _make_step_resolver(),
            run_id="run_001",
            aggregation_settings={node_id: agg_settings},
        )
        new_executor.restore_from_checkpoint(checkpoint)

        # Verify restored tokens have PipelineRow
        restored_tokens = new_executor.get_buffered_tokens(node_id)
        assert len(restored_tokens) == 1
        assert isinstance(restored_tokens[0].row_data, PipelineRow)
        assert restored_tokens[0].row_data["value"] == "test"
        # Verify contract is restored
        assert restored_tokens[0].row_data.contract is not None
        assert restored_tokens[0].row_data.contract.mode == "FLEXIBLE"

    def test_restore_from_checkpoint_buffer_has_dicts(self) -> None:
        """After restore, _buffers should contain dicts (not PipelineRow)."""
        from elspeth.contracts.types import NodeID
        from elspeth.core.config import AggregationSettings, TriggerConfig
        from elspeth.engine.executors import AggregationExecutor
        from elspeth.engine.spans import SpanFactory

        # Setup contract and token
        contract = _make_contract()
        row = make_row({"value": "test"}, contract=contract)
        token = TokenInfo(row_id="row_001", token_id="token_001", row_data=row)

        # Mock recorder
        mock_recorder = MagicMock()
        mock_batch = MagicMock()
        mock_batch.batch_id = "batch_001"
        mock_recorder.create_batch.return_value = mock_batch

        # Mock span factory
        mock_span_factory = MagicMock(spec=SpanFactory)

        # Create aggregation settings
        agg_settings = AggregationSettings(
            name="test_agg",
            plugin="batch_stats",
            input="default",
            trigger=TriggerConfig(count=10),
        )
        node_id = NodeID("agg_001")

        # Create executor and buffer row
        executor = AggregationExecutor(
            mock_recorder,
            mock_span_factory,
            _make_step_resolver(),
            run_id="run_001",
            aggregation_settings={node_id: agg_settings},
        )
        executor.buffer_row(node_id, token)

        # Get checkpoint
        checkpoint = executor.get_checkpoint_state()

        # Create new executor and restore from checkpoint
        new_executor = AggregationExecutor(
            mock_recorder,
            mock_span_factory,
            _make_step_resolver(),
            run_id="run_001",
            aggregation_settings={node_id: agg_settings},
        )
        new_executor.restore_from_checkpoint(checkpoint)

        # Verify _buffers contains dicts, not PipelineRow
        restored_rows = new_executor.get_buffered_rows(node_id)
        assert len(restored_rows) == 1
        assert isinstance(restored_rows[0], dict)
        # PipelineRow subclasses dict, so type() check is needed, not isinstance()
        assert type(restored_rows[0]) is not PipelineRow  # type: ignore[comparison-overlap, unreachable]
        assert restored_rows[0] == {"value": "test"}  # type: ignore[unreachable]
