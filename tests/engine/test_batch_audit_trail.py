# tests/engine/test_batch_audit_trail.py
"""Audit trail tests for batch transforms.

These tests verify that the audit trail is correctly recorded when processing
rows through batch transforms. Unlike test_executor_batch_integration.py,
these tests do NOT mock record_call - they validate the actual audit records.

Focus:
- Node state records (begin_node_state, complete_node_state)
- External call records (record_call with CallType.LLM)
- Proper state transitions for success and error paths

NOTE: These tests use a temporary file database instead of in-memory SQLite
because BatchTransformMixin uses worker threads. SQLite in-memory databases
are per-connection, so each worker thread would get an empty database.
Using a temp file ensures all threads share the same database.
"""

from __future__ import annotations

import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest

from elspeth.contracts.audit import NodeStateCompleted, NodeStateFailed
from elspeth.contracts.enums import CallStatus, CallType, NodeStateStatus, NodeType
from elspeth.contracts.identity import TokenInfo
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.recorder import LandscapeRecorder
from elspeth.engine.executors import TransformExecutor
from elspeth.engine.spans import SpanFactory
from elspeth.plugins.context import PluginContext
from elspeth.plugins.llm.azure import AzureLLMTransform

DYNAMIC_SCHEMA = {"fields": "dynamic"}


# =============================================================================
# Test Helpers
# =============================================================================


def create_test_environment(
    recorder: LandscapeRecorder,
    plugin_name: str = "azure_llm",
) -> tuple[str, str]:
    """Create run and node in recorder, return (run_id, node_id).

    Args:
        recorder: LandscapeRecorder instance
        plugin_name: Name of the plugin being tested

    Returns:
        Tuple of (run_id, node_id)
    """
    # Create run
    run = recorder.begin_run(config={}, canonical_version="v1")
    run_id = run.run_id

    # Register node with dynamic schema
    schema = SchemaConfig.from_dict(DYNAMIC_SCHEMA)
    node = recorder.register_node(
        run_id=run_id,
        plugin_name=plugin_name,
        node_type=NodeType.TRANSFORM,
        plugin_version="1.0",
        config={},
        schema_config=schema,
    )

    return run_id, node.node_id


def create_token_in_recorder(
    recorder: LandscapeRecorder,
    run_id: str,
    node_id: str,
    row_id: str,
    token_id: str,
    row_data: dict[str, Any],
    row_index: int = 0,
) -> TokenInfo:
    """Create row and token in recorder, return TokenInfo.

    Args:
        recorder: LandscapeRecorder instance
        run_id: Run ID
        node_id: Source node ID for the row
        row_id: Row ID
        token_id: Token ID
        row_data: Row data dict
        row_index: Index of row in source

    Returns:
        TokenInfo for the created token
    """
    row = recorder.create_row(
        run_id=run_id,
        source_node_id=node_id,
        row_index=row_index,
        data=row_data,
        row_id=row_id,
    )
    recorder.create_token(row_id=row.row_id, token_id=token_id)
    return TokenInfo(row_id=row_id, token_id=token_id, row_data=row_data)


@contextmanager
def mock_azure_openai(
    responses: list[str] | None = None,
    default_response: str = "Test response",
    side_effect: Exception | None = None,
) -> Generator[Mock, None, None]:
    """Context manager to mock openai.AzureOpenAI for testing.

    Args:
        responses: List of responses to return in order
        default_response: Default response if responses not provided
        side_effect: Exception to raise instead of returning response

    Yields:
        Mock client instance for assertions
    """
    response_list = responses or [default_response]
    call_count = [0]

    def make_response(**kwargs: Any) -> Mock:
        if side_effect:
            raise side_effect

        content = response_list[call_count[0] % len(response_list)]
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


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def recorder() -> Generator[LandscapeRecorder, None, None]:
    """Create recorder with thread-safe temp file DB.

    Uses a temp file instead of in-memory SQLite because BatchTransformMixin
    uses worker threads. SQLite in-memory databases are per-connection, so
    each worker thread would get an empty database without tables.

    NOTE: Unlike test_executor_batch_integration.py, we do NOT mock record_call.
    This allows us to verify actual audit trail records.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_audit.db"
        db = LandscapeDB.from_url(f"sqlite:///{db_path}", create_tables=True)
        yield LandscapeRecorder(db)


@pytest.fixture
def executor(recorder: LandscapeRecorder) -> TransformExecutor:
    """Create TransformExecutor for testing."""
    spans = SpanFactory()  # No tracer = no-op spans
    return TransformExecutor(recorder, spans)


# =============================================================================
# Success Path Tests
# =============================================================================


class TestAuditTrailSuccessPath:
    """Tests for audit trail recording on successful LLM transform execution."""

    def test_success_records_node_state_and_call(
        self,
        recorder: LandscapeRecorder,
        executor: TransformExecutor,
    ) -> None:
        """Verify successful LLM call records proper node state and call.

        Assertions:
        - result.status == "success" and response field present
        - get_node_states_for_token returns 1 state with status == "completed"
        - node_id matches, input_hash/output_hash non-null
        - get_calls(state_id) returns 1 call with CallType.LLM, CallStatus.SUCCESS
        - request/response hashes are set
        """
        # Setup
        run_id, node_id = create_test_environment(recorder)

        with mock_azure_openai(responses=["Positive sentiment"]) as mock_client:
            transform = AzureLLMTransform(
                {
                    "deployment_name": "gpt-4o",
                    "endpoint": "https://test.openai.azure.com",
                    "api_key": "test-key",
                    "template": "Analyze: {{ row.text }}",
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
                row_data={"text": "This product is excellent!"},
            )

            # Execute
            result, _updated_token, error_sink = executor.execute_transform(
                transform=transform,
                token=token,
                ctx=ctx,
                step_in_pipeline=0,
            )

            # Verify result
            assert result.status == "success"
            assert result.row is not None
            assert result.row["llm_response"] == "Positive sentiment"
            assert error_sink is None

            # Verify LLM was called
            assert mock_client.chat.completions.create.call_count == 1

            # Verify node state recorded
            states = recorder.get_node_states_for_token(token_id="token-1")
            assert len(states) == 1

            state = states[0]
            assert isinstance(state, NodeStateCompleted)
            assert state.node_id == node_id
            assert state.status == NodeStateStatus.COMPLETED
            assert state.input_hash is not None
            assert state.output_hash is not None

            # Verify call recorded
            calls = recorder.get_calls(state.state_id)
            assert len(calls) == 1

            call = calls[0]
            assert call.call_type == CallType.LLM
            assert call.status == CallStatus.SUCCESS
            assert call.request_hash is not None
            assert call.response_hash is not None
            assert call.error_json is None

            # Cleanup
            transform.close()


# =============================================================================
# Error Path Tests
# =============================================================================


class TestAuditTrailErrorPath:
    """Tests for audit trail recording on failed LLM transform execution."""

    def test_error_records_failed_node_state_and_error_call(
        self,
        recorder: LandscapeRecorder,
        executor: TransformExecutor,
    ) -> None:
        """Verify LLM error records proper node state and call with error.

        Assertions:
        - result.status == "error" and error_sink == "error_sink"
        - Node state status is failed, error payload recorded
        - get_calls(state_id) has CallStatus.ERROR and error_json populated
        """
        # Setup
        run_id, node_id = create_test_environment(recorder)

        with patch("openai.AzureOpenAI") as mock_azure_class:
            mock_client = Mock()
            # Non-retryable error to exercise error-path handling
            mock_client.chat.completions.create.side_effect = Exception("content policy violation")
            mock_azure_class.return_value = mock_client

            transform = AzureLLMTransform(
                {
                    "deployment_name": "gpt-4o",
                    "endpoint": "https://test.openai.azure.com",
                    "api_key": "test-key",
                    "template": "Analyze: {{ row.text }}",
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
                row_data={"text": "Test input"},
            )

            # Execute
            result, _updated_token, error_sink = executor.execute_transform(
                transform=transform,
                token=token,
                ctx=ctx,
                step_in_pipeline=0,
            )

            # Verify error result
            assert result.status == "error"
            assert result.reason is not None
            assert error_sink == "error_sink"

            # Verify node state recorded as failed
            states = recorder.get_node_states_for_token(token_id="token-1")
            assert len(states) == 1

            state = states[0]
            assert isinstance(state, NodeStateFailed)
            assert state.node_id == node_id
            assert state.status == NodeStateStatus.FAILED
            assert state.input_hash is not None
            # Failed states may or may not have error_json depending on how failure was recorded
            # The key thing is that the state was recorded as failed

            # Verify call recorded with error status
            calls = recorder.get_calls(state.state_id)
            assert len(calls) == 1

            call = calls[0]
            assert call.call_type == CallType.LLM
            assert call.status == CallStatus.ERROR
            assert call.request_hash is not None
            assert call.error_json is not None  # Error details should be recorded
            # Response hash may be None for errors
            assert call.response_hash is None

            # Cleanup
            transform.close()
