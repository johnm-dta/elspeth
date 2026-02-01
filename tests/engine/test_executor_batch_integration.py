# tests/engine/test_executor_batch_integration.py
"""Integration tests for TransformExecutor with batch transforms.

These tests verify the SharedBatchAdapter integration that allows
batch transforms (using BatchTransformMixin) to work with the engine.

The key fix: TransformExecutor now uses one SharedBatchAdapter per transform
(instead of one BlockingResultAdapter per row), with multiple waiters
registered by token_id. This prevents the deadlock where only the first
row's adapter was connected to the transform's output port.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any
from unittest.mock import Mock, patch

import pytest

from elspeth.contracts import NodeStateCompleted, TransformResult
from elspeth.contracts.enums import NodeType
from elspeth.contracts.identity import TokenInfo
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.recorder import LandscapeRecorder
from elspeth.engine.executors import TransformExecutor
from elspeth.engine.spans import SpanFactory
from elspeth.plugins.context import PluginContext
from elspeth.plugins.llm.azure import AzureLLMTransform

DYNAMIC_SCHEMA = {"fields": "dynamic"}


@contextmanager
def mock_azure_openai(
    responses: list[str] | None = None,
    default_response: str = "Response",
) -> Generator[Mock, None, None]:
    """Context manager to mock openai.AzureOpenAI for batch processing.

    Args:
        responses: List of responses to return in order (cycles if exhausted)
        default_response: Default response if responses not provided

    Yields:
        Mock client instance for assertions
    """
    response_iter = iter(responses or [default_response])
    call_count = [0]

    def make_response(**kwargs: Any) -> Mock:
        try:
            content = next(response_iter)
        except StopIteration:
            # Cycle back to first response
            content = (responses or [default_response])[call_count[0] % len(responses or [default_response])]

        call_count[0] += 1

        mock_usage = Mock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5

        mock_message = Mock()
        mock_message.content = content

        mock_choice = Mock()
        mock_choice.message = mock_message

        mock_response = Mock()
        mock_response.choices = [mock_choice]
        mock_response.model = "gpt-4o"
        mock_response.usage = mock_usage
        mock_response.model_dump = Mock(return_value={"model": "gpt-4o"})

        return mock_response

    with patch("openai.AzureOpenAI") as mock_azure_class:
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = make_response
        mock_azure_class.return_value = mock_client
        yield mock_client


@pytest.fixture
def recorder() -> LandscapeRecorder:
    """Create recorder with in-memory DB.

    Mock record_call to avoid calls table issues - the calls table
    may not exist in the schema for some in-memory test DBs.
    """
    db = LandscapeDB.in_memory()
    rec = LandscapeRecorder(db)
    rec.record_call = Mock()  # type: ignore[method-assign]
    return rec


@pytest.fixture
def executor(recorder: LandscapeRecorder) -> TransformExecutor:
    """Create TransformExecutor for testing."""
    spans = SpanFactory()  # No tracer = no-op spans
    return TransformExecutor(recorder, spans)


@pytest.fixture
def run_id(recorder: LandscapeRecorder) -> str:
    """Create a run for testing."""
    run = recorder.begin_run(config={}, canonical_version="v1")
    return run.run_id


@pytest.fixture
def node_id(recorder: LandscapeRecorder, run_id: str) -> str:
    """Create a node for testing."""
    schema = SchemaConfig.from_dict(DYNAMIC_SCHEMA)
    node = recorder.register_node(
        run_id=run_id,
        plugin_name="azure_llm",
        node_type=NodeType.TRANSFORM,
        plugin_version="1.0",
        config={},
        schema_config=schema,
    )
    return node.node_id


def create_token_in_recorder(
    recorder: LandscapeRecorder,
    run_id: str,
    node_id: str,
    row_id: str,
    token_id: str,
    row_data: dict[str, Any],
    row_index: int = 0,
) -> TokenInfo:
    """Create row and token in recorder, return TokenInfo."""
    row = recorder.create_row(
        run_id=run_id,
        source_node_id=node_id,
        row_index=row_index,
        data=row_data,
        row_id=row_id,
    )
    recorder.create_token(row_id=row.row_id, token_id=token_id)
    return TokenInfo(row_id=row_id, token_id=token_id, row_data=row_data)


class TestExecutorBatchIntegration:
    """Tests for TransformExecutor with batch transforms."""

    def test_executor_single_row_through_batch_transform(
        self,
        recorder: LandscapeRecorder,
        executor: TransformExecutor,
        run_id: str,
        node_id: str,
    ) -> None:
        """Verify a single row processes correctly through batch transform."""
        with mock_azure_openai(responses=["positive"]) as mock_client:
            transform = AzureLLMTransform(
                {
                    "deployment_name": "gpt-4o",
                    "endpoint": "https://test.openai.azure.com",
                    "api_key": "test-key",
                    "template": "Classify: {{ row.text }}",
                    "schema": DYNAMIC_SCHEMA,
                    "required_input_fields": [],  # Explicit opt-out for this test
                    "pool_size": 5,
                }
            )
            transform.node_id = node_id

            # Create context with required fields
            ctx = PluginContext(
                run_id=run_id,
                config={},
                landscape=recorder,
            )

            # Call on_start to initialize recorder reference
            transform.on_start(ctx)

            # Create token in recorder
            token = create_token_in_recorder(
                recorder,
                run_id,
                node_id,
                row_id="row-1",
                token_id="token-1",
                row_data={"text": "Great product!"},
            )

            # Execute through executor
            result, _updated_token, error_sink = executor.execute_transform(
                transform=transform,
                token=token,
                ctx=ctx,
                step_in_pipeline=0,
            )

            # Verify result
            assert result.status == "success"
            assert result.row is not None
            assert result.row["llm_response"] == "positive"
            assert error_sink is None

            # Verify LLM was called
            assert mock_client.chat.completions.create.call_count == 1

            # Cleanup
            transform.close()

    def test_executor_multiple_rows_no_deadlock(
        self,
        recorder: LandscapeRecorder,
        executor: TransformExecutor,
        run_id: str,
        node_id: str,
    ) -> None:
        """Verify multiple rows process without deadlock.

        This is the key test - the old BlockingResultAdapter caused deadlock
        on the second row because only the first adapter was connected.
        """
        responses = ["positive", "negative", "neutral"]

        with mock_azure_openai(responses=responses) as mock_client:
            transform = AzureLLMTransform(
                {
                    "deployment_name": "gpt-4o",
                    "endpoint": "https://test.openai.azure.com",
                    "api_key": "test-key",
                    "template": "Classify: {{ row.text }}",
                    "schema": DYNAMIC_SCHEMA,
                    "required_input_fields": [],  # Explicit opt-out for this test
                    "pool_size": 5,
                }
            )
            transform.node_id = node_id

            ctx = PluginContext(
                run_id=run_id,
                config={},
                landscape=recorder,
            )
            transform.on_start(ctx)

            # Process 3 rows
            rows = [
                {"text": "Great!"},
                {"text": "Terrible!"},
                {"text": "Okay"},
            ]
            results: list[TransformResult] = []

            for i, row_data in enumerate(rows):
                token = create_token_in_recorder(
                    recorder,
                    run_id,
                    node_id,
                    row_id=f"row-{i}",
                    token_id=f"token-{i}",
                    row_data=row_data,
                    row_index=i,
                )

                result, _, _error_sink = executor.execute_transform(
                    transform=transform,
                    token=token,
                    ctx=ctx,
                    step_in_pipeline=0,
                )
                results.append(result)

            # All 3 should succeed (this would hang with the old bug)
            assert all(r.status == "success" for r in results)

            # Results should have correct responses
            assert results[0].row is not None
            assert results[1].row is not None
            assert results[2].row is not None
            assert results[0].row["llm_response"] == "positive"
            assert results[1].row["llm_response"] == "negative"
            assert results[2].row["llm_response"] == "neutral"

            # Verify LLM was called 3 times
            assert mock_client.chat.completions.create.call_count == 3

            transform.close()

    def test_executor_error_handling_in_batch_transform(
        self,
        recorder: LandscapeRecorder,
        executor: TransformExecutor,
        run_id: str,
        node_id: str,
    ) -> None:
        """Verify error handling works through batch transform."""
        with patch("openai.AzureOpenAI") as mock_azure_class:
            mock_client = Mock()
            mock_client.chat.completions.create.side_effect = Exception("API error")
            mock_azure_class.return_value = mock_client

            transform = AzureLLMTransform(
                {
                    "deployment_name": "gpt-4o",
                    "endpoint": "https://test.openai.azure.com",
                    "api_key": "test-key",
                    "template": "Classify: {{ row.text }}",
                    "schema": DYNAMIC_SCHEMA,
                    "required_input_fields": [],  # Explicit opt-out for this test
                    "pool_size": 5,
                    "on_error": "error_sink",  # Configure error routing
                }
            )
            transform.node_id = node_id

            ctx = PluginContext(
                run_id=run_id,
                config={},
                landscape=recorder,
            )
            transform.on_start(ctx)

            token = create_token_in_recorder(
                recorder,
                run_id,
                node_id,
                row_id="row-1",
                token_id="token-1",
                row_data={"text": "test"},
            )

            result, _, error_sink = executor.execute_transform(
                transform=transform,
                token=token,
                ctx=ctx,
                step_in_pipeline=0,
            )

            # Verify error result
            assert result.status == "error"
            assert result.reason is not None
            assert error_sink == "error_sink"

            transform.close()

    def test_executor_reuses_adapter_across_rows(
        self,
        recorder: LandscapeRecorder,
        executor: TransformExecutor,
        run_id: str,
        node_id: str,
    ) -> None:
        """Verify the same SharedBatchAdapter is reused for all rows.

        The fix stores the adapter on the transform instance and reuses it.
        """
        with mock_azure_openai(responses=["a", "b"]):
            transform = AzureLLMTransform(
                {
                    "deployment_name": "gpt-4o",
                    "endpoint": "https://test.openai.azure.com",
                    "api_key": "test-key",
                    "template": "{{ row.text }}",
                    "schema": DYNAMIC_SCHEMA,
                    "required_input_fields": [],  # Explicit opt-out for this test
                    "pool_size": 5,
                }
            )
            transform.node_id = node_id

            ctx = PluginContext(
                run_id=run_id,
                config={},
                landscape=recorder,
            )
            transform.on_start(ctx)

            # Process first row
            token1 = create_token_in_recorder(
                recorder,
                run_id,
                node_id,
                row_id="r1",
                token_id="t1",
                row_data={"text": "A"},
                row_index=0,
            )
            executor.execute_transform(transform, token1, ctx, step_in_pipeline=0)

            # Capture adapter reference (dynamically added attribute)
            adapter1 = getattr(transform, "_executor_batch_adapter", None)

            # Process second row
            token2 = create_token_in_recorder(
                recorder,
                run_id,
                node_id,
                row_id="r2",
                token_id="t2",
                row_data={"text": "B"},
                row_index=1,
            )
            executor.execute_transform(transform, token2, ctx, step_in_pipeline=0)

            # Adapter should be the same instance (dynamically added attribute)
            adapter2 = getattr(transform, "_executor_batch_adapter", None)
            assert adapter1 is not None
            assert adapter2 is not None
            assert adapter1 is adapter2

            transform.close()

    def test_executor_audit_trail_recorded(
        self,
        recorder: LandscapeRecorder,
        executor: TransformExecutor,
        run_id: str,
        node_id: str,
    ) -> None:
        """Verify audit trail is properly recorded for batch transforms."""
        with mock_azure_openai(responses=["result"]):
            transform = AzureLLMTransform(
                {
                    "deployment_name": "gpt-4o",
                    "endpoint": "https://test.openai.azure.com",
                    "api_key": "test-key",
                    "template": "{{ row.text }}",
                    "schema": DYNAMIC_SCHEMA,
                    "required_input_fields": [],  # Explicit opt-out for this test
                    "pool_size": 5,
                }
            )
            transform.node_id = node_id

            ctx = PluginContext(
                run_id=run_id,
                config={},
                landscape=recorder,
            )
            transform.on_start(ctx)

            token = create_token_in_recorder(
                recorder,
                run_id,
                node_id,
                row_id="row-1",
                token_id="token-1",
                row_data={"text": "test input"},
            )

            result, _, _ = executor.execute_transform(
                transform=transform,
                token=token,
                ctx=ctx,
                step_in_pipeline=0,
            )

            # Verify node_states recorded
            states = recorder.get_node_states_for_token(token_id="token-1")
            assert len(states) == 1

            state = states[0]
            assert state.node_id == node_id
            assert state.status == "completed"
            # Type narrow to NodeStateCompleted for attribute access
            assert isinstance(state, NodeStateCompleted)
            # NodeStateCompleted has input_hash and output_hash (hashes only, data in payload store)
            assert state.input_hash is not None
            assert state.output_hash is not None

            # Verify result has expected response
            assert result.status == "success"
            assert result.row is not None
            assert result.row["llm_response"] == "result"

            transform.close()


class TestBatchTransformTimeoutEviction:
    """Tests for eviction on timeout.

    When a batch transform times out, the executor must call evict_submission()
    to prevent FIFO blocking on retry attempts.
    """

    def test_executor_calls_evict_on_timeout(
        self,
        real_landscape_recorder: LandscapeRecorder,
    ) -> None:
        """Verify executor calls evict_submission when waiter times out.

        This ensures that timeout cleanup properly evicts the buffer entry,
        allowing retries to proceed without FIFO blocking.
        """
        recorder = real_landscape_recorder
        spans = SpanFactory()  # No tracer = no-op spans
        executor = TransformExecutor(recorder, spans)

        # Create run and node
        run = recorder.begin_run(config={}, canonical_version="v1")
        run_id = run.run_id

        schema = SchemaConfig.from_dict(DYNAMIC_SCHEMA)
        node = recorder.register_node(
            run_id=run_id,
            plugin_name="mock_batch_transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=schema,
        )
        node_id = node.node_id

        # Create a mock transform that supports eviction
        mock_transform = Mock()
        mock_transform.name = "mock_batch_transform"
        mock_transform.node_id = node_id  # Required for audit trail

        # Transform has accept() so executor treats it as batch transform
        mock_transform.accept = Mock()

        # Transform supports eviction
        mock_transform.evict_submission = Mock(return_value=True)

        # Mock the adapter to timeout on wait
        mock_waiter = Mock()
        mock_waiter.wait = Mock(side_effect=TimeoutError("No result received within timeout"))

        mock_adapter = Mock()
        mock_adapter.register = Mock(return_value=mock_waiter)

        # Inject the mock adapter on the TRANSFORM (not executor)
        # The executor stores adapters on transform._executor_batch_adapter
        mock_transform._executor_batch_adapter = mock_adapter
        mock_transform._batch_initialized = True  # Prevent connect_output being called

        token = create_token_in_recorder(
            recorder,
            run_id,
            node_id,
            row_id="row-timeout",
            token_id="token-timeout",
            row_data={"text": "timeout test"},
        )

        ctx = PluginContext(
            run_id=run_id,
            config={},
            landscape=recorder,
            state_id="state-attempt-1",
        )
        ctx.token = token

        # Execute - should timeout and call evict_submission
        with pytest.raises(TimeoutError):
            executor.execute_transform(
                transform=mock_transform,
                token=token,
                ctx=ctx,
                step_in_pipeline=0,
            )

        # Verify evict_submission was called with correct args
        mock_transform.evict_submission.assert_called_once_with(token.token_id, ctx.state_id)
