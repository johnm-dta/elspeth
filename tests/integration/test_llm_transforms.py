# tests/integration/test_llm_transforms.py
"""Integration tests for LLM transform plugins.

These tests verify the complete flow from template rendering through
API calls to response parsing, using mocked HTTP/SDK responses and
real LandscapeRecorder with in-memory databases.

Test scenarios covered:
1. Template rendering -> API call -> response parsing (BaseLLMTransform)
2. Audit trail records template_hash, variables_hash in output
3. Batch aggregation -> fan-out works correctly (AzureBatchLLMTransform)
4. Error handling for API failures (recorded in audit trail)
5. OpenRouterLLMTransform HTTP client integration
6. Rate limit errors are retryable
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, Mock

import httpx
import pytest

from elspeth.contracts import CallStatus, CallType, TransformResult
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.recorder import LandscapeRecorder
from elspeth.plugins.clients.llm import (
    AuditedLLMClient,
    LLMClientError,
)
from elspeth.plugins.context import PluginContext
from elspeth.plugins.llm.azure_batch import AzureBatchLLMTransform
from elspeth.plugins.llm.base import BaseLLMTransform
from elspeth.plugins.llm.batch_errors import BatchPendingError
from elspeth.plugins.llm.openrouter import OpenRouterLLMTransform

# Dynamic schema for tests
DYNAMIC_SCHEMA = {"fields": "dynamic"}


# Concrete subclass of BaseLLMTransform for testing
# Named without 'Test' prefix to avoid pytest collection warning
class ConcreteLLMTransform(BaseLLMTransform):
    """Concrete LLM transform for testing.

    This class implements _get_llm_client() by reading the client from
    ctx.llm_client, which is set up by the tests. This allows testing
    the base class behavior with a mocked client.
    """

    name = "concrete_llm"

    def _get_llm_client(self, ctx: PluginContext) -> AuditedLLMClient:
        """Return the client from context (set up by test fixtures)."""
        # Tests set ctx.llm_client with a mocked AuditedLLMClient
        return ctx.llm_client  # type: ignore[return-value]


class TestLLMTransformIntegration:
    """Integration tests for LLM transforms with real audit trail."""

    @pytest.fixture
    def recorder(self) -> LandscapeRecorder:
        """Create recorder with in-memory DB."""
        db = LandscapeDB.in_memory()
        return LandscapeRecorder(db)

    @pytest.fixture
    def setup_state(self, recorder: LandscapeRecorder) -> tuple[str, str, str, str, str]:
        """Create run, node, row, token, and state for testing.

        Returns:
            Tuple of (run_id, node_id, row_id, token_id, state_id)
        """
        schema = SchemaConfig.from_dict({"fields": "dynamic"})
        run = recorder.begin_run(config={}, canonical_version="v1")
        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="llm_transform",
            node_type="transform",
            plugin_version="1.0",
            config={},
            schema_config=schema,
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=node.node_id,
            row_index=0,
            data={"input": "test"},
        )
        token = recorder.create_token(row_id=row.row_id)
        state = recorder.begin_node_state(
            token_id=token.token_id,
            node_id=node.node_id,
            step_index=0,
            input_data={"input": "test"},
        )
        return run.run_id, node.node_id, row.row_id, token.token_id, state.state_id

    def test_template_rendering_api_call_response_parsing(
        self, recorder: LandscapeRecorder, setup_state: tuple[str, str, str, str, str]
    ) -> None:
        """Verify template renders, API is called, response is parsed."""
        run_id, _node_id, _row_id, _token_id, state_id = setup_state

        # Create mock OpenAI-compatible client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello, Alice!"
        mock_response.model = "gpt-4"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.model_dump.return_value = {}
        mock_client.chat.completions.create.return_value = mock_response

        # Create audited client that records to audit trail
        audited_client = AuditedLLMClient(
            recorder=recorder,
            state_id=state_id,
            underlying_client=mock_client,
            provider="openai",
        )

        # Create context with client
        ctx = PluginContext(run_id=run_id, config={})
        ctx.llm_client = audited_client

        # Create and run transform
        transform = ConcreteLLMTransform(
            {
                "model": "gpt-4",
                "template": "Say hello to {{ row.name }}!",
                "schema": DYNAMIC_SCHEMA,
            }
        )

        result = transform.process({"name": "Alice"}, ctx)

        # Verify result
        assert result.status == "success"
        assert result.row is not None
        assert result.row["llm_response"] == "Hello, Alice!"
        assert result.row["llm_response_usage"]["prompt_tokens"] == 10
        assert result.row["llm_response_usage"]["completion_tokens"] == 5

        # Verify API was called with rendered template
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Say hello to Alice!"

        # Verify audit trail - call was recorded
        calls = recorder.get_calls(state_id)
        assert len(calls) == 1
        assert calls[0].call_type == CallType.LLM
        assert calls[0].status == CallStatus.SUCCESS
        assert calls[0].latency_ms is not None
        assert calls[0].latency_ms > 0

    def test_audit_trail_records_template_hashes(self, recorder: LandscapeRecorder, setup_state: tuple[str, str, str, str, str]) -> None:
        """Verify template_hash and variables_hash are in output."""
        run_id, _node_id, _row_id, _token_id, state_id = setup_state

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.model = "gpt-4"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 3
        mock_response.model_dump.return_value = {}
        mock_client.chat.completions.create.return_value = mock_response

        audited_client = AuditedLLMClient(
            recorder=recorder,
            state_id=state_id,
            underlying_client=mock_client,
        )

        ctx = PluginContext(run_id=run_id, config={})
        ctx.llm_client = audited_client

        transform = ConcreteLLMTransform(
            {
                "model": "gpt-4",
                "template": "Analyze: {{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
            }
        )

        result = transform.process({"text": "Hello world"}, ctx)

        assert result.status == "success"
        assert result.row is not None

        # Verify template hash is present
        assert "llm_response_template_hash" in result.row
        assert isinstance(result.row["llm_response_template_hash"], str)
        assert len(result.row["llm_response_template_hash"]) == 64  # SHA-256 hex

        # Verify variables hash is present
        assert "llm_response_variables_hash" in result.row
        assert isinstance(result.row["llm_response_variables_hash"], str)
        assert len(result.row["llm_response_variables_hash"]) == 64  # SHA-256 hex

    def test_api_error_recorded_in_audit_trail(self, recorder: LandscapeRecorder, setup_state: tuple[str, str, str, str, str]) -> None:
        """Verify API errors are recorded in audit trail."""
        run_id, _node_id, _row_id, _token_id, state_id = setup_state

        # Mock client that raises an error
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API server error")

        audited_client = AuditedLLMClient(
            recorder=recorder,
            state_id=state_id,
            underlying_client=mock_client,
        )

        ctx = PluginContext(run_id=run_id, config={})
        ctx.llm_client = audited_client

        transform = ConcreteLLMTransform(
            {
                "model": "gpt-4",
                "template": "{{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
            }
        )

        result = transform.process({"text": "test"}, ctx)

        # Transform should return error result
        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "llm_call_failed"

        # Audit trail should have the error recorded
        calls = recorder.get_calls(state_id)
        assert len(calls) == 1
        assert calls[0].call_type == CallType.LLM
        assert calls[0].status == CallStatus.ERROR
        assert calls[0].error_json is not None
        assert "API server error" in calls[0].error_json

    def test_rate_limit_error_is_retryable(self, recorder: LandscapeRecorder, setup_state: tuple[str, str, str, str, str]) -> None:
        """Verify rate limit errors are marked retryable."""
        run_id, _node_id, _row_id, _token_id, state_id = setup_state

        # Mock client that raises rate limit error
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("Error 429: Rate limit exceeded")

        audited_client = AuditedLLMClient(
            recorder=recorder,
            state_id=state_id,
            underlying_client=mock_client,
        )

        ctx = PluginContext(run_id=run_id, config={})
        ctx.llm_client = audited_client

        transform = ConcreteLLMTransform(
            {
                "model": "gpt-4",
                "template": "{{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
            }
        )

        result = transform.process({"text": "test"}, ctx)

        # Transform should return retryable error
        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "rate_limited"
        assert result.retryable is True

        # Audit trail should record as error
        calls = recorder.get_calls(state_id)
        assert len(calls) == 1
        assert calls[0].status == CallStatus.ERROR

    def test_system_prompt_included_when_configured(self, recorder: LandscapeRecorder, setup_state: tuple[str, str, str, str, str]) -> None:
        """Verify system prompt is included in API call."""
        run_id, _node_id, _row_id, _token_id, state_id = setup_state

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.model = "gpt-4"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.model_dump.return_value = {}
        mock_client.chat.completions.create.return_value = mock_response

        audited_client = AuditedLLMClient(
            recorder=recorder,
            state_id=state_id,
            underlying_client=mock_client,
        )

        ctx = PluginContext(run_id=run_id, config={})
        ctx.llm_client = audited_client

        transform = ConcreteLLMTransform(
            {
                "model": "gpt-4",
                "template": "{{ row.text }}",
                "system_prompt": "You are a helpful assistant.",
                "schema": DYNAMIC_SCHEMA,
            }
        )

        transform.process({"text": "Hello"}, ctx)

        # Verify messages include system prompt
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a helpful assistant."
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Hello"


class TestOpenRouterLLMTransformIntegration:
    """Integration tests for OpenRouter LLM transform with HTTP client."""

    @pytest.fixture
    def recorder(self) -> LandscapeRecorder:
        """Create recorder with in-memory DB."""
        db = LandscapeDB.in_memory()
        return LandscapeRecorder(db)

    @pytest.fixture
    def setup_state(self, recorder: LandscapeRecorder) -> tuple[str, str]:
        """Create run and state for testing.

        Returns:
            Tuple of (run_id, state_id)
        """
        schema = SchemaConfig.from_dict({"fields": "dynamic"})
        run = recorder.begin_run(config={}, canonical_version="v1")
        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="openrouter_llm",
            node_type="transform",
            plugin_version="1.0",
            config={},
            schema_config=schema,
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=node.node_id,
            row_index=0,
            data={"input": "test"},
        )
        token = recorder.create_token(row_id=row.row_id)
        state = recorder.begin_node_state(
            token_id=token.token_id,
            node_id=node.node_id,
            step_index=0,
            input_data={"input": "test"},
        )
        return run.run_id, state.state_id

    def test_http_client_call_and_response_parsing(self, recorder: LandscapeRecorder, setup_state: tuple[str, str]) -> None:
        """Verify OpenRouter uses HTTP client and parses response."""
        from unittest.mock import patch

        run_id, state_id = setup_state

        # Create transform
        transform = OpenRouterLLMTransform(
            {
                "model": "anthropic/claude-3-opus",
                "template": "Analyze: {{ row.text }}",
                "api_key": "test-api-key",
                "schema": DYNAMIC_SCHEMA,
            }
        )

        # Mock HTTP response
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Analysis complete"}}],
            "usage": {"prompt_tokens": 15, "completion_tokens": 10},
            "model": "anthropic/claude-3-opus",
        }
        mock_response.raise_for_status = MagicMock()
        mock_response.content = b""
        mock_response.text = ""

        ctx = PluginContext(
            run_id=run_id,
            config={},
            landscape=recorder,
            state_id=state_id,
        )

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.post.return_value = mock_response
            mock_client_class.return_value.__enter__ = MagicMock(return_value=mock_client)
            mock_client_class.return_value.__exit__ = MagicMock(return_value=None)

            result = transform.process({"text": "Hello world"}, ctx)

        # Verify result
        assert result.status == "success"
        assert result.row is not None
        assert result.row["llm_response"] == "Analysis complete"
        assert result.row["llm_response_usage"]["prompt_tokens"] == 15
        assert result.row["llm_response_model"] == "anthropic/claude-3-opus"

    def test_http_error_returns_transform_error(self, recorder: LandscapeRecorder, setup_state: tuple[str, str]) -> None:
        """Verify HTTP errors are handled gracefully."""
        from unittest.mock import patch

        run_id, state_id = setup_state

        transform = OpenRouterLLMTransform(
            {
                "model": "anthropic/claude-3-opus",
                "template": "{{ row.text }}",
                "api_key": "test-api-key",
                "schema": DYNAMIC_SCHEMA,
            }
        )

        ctx = PluginContext(
            run_id=run_id,
            config={},
            landscape=recorder,
            state_id=state_id,
        )

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.post.side_effect = httpx.HTTPStatusError(
                "Server Error",
                request=MagicMock(),
                response=MagicMock(status_code=500),
            )
            mock_client_class.return_value.__enter__ = MagicMock(return_value=mock_client)
            mock_client_class.return_value.__exit__ = MagicMock(return_value=None)

            result = transform.process({"text": "test"}, ctx)

        # Verify error result
        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "api_call_failed"
        assert result.retryable is False  # 500 is not rate limit

    def test_rate_limit_http_error_is_retryable(self, recorder: LandscapeRecorder, setup_state: tuple[str, str]) -> None:
        """Verify 429 HTTP errors are marked retryable."""
        from unittest.mock import patch

        run_id, state_id = setup_state

        transform = OpenRouterLLMTransform(
            {
                "model": "anthropic/claude-3-opus",
                "template": "{{ row.text }}",
                "api_key": "test-api-key",
                "schema": DYNAMIC_SCHEMA,
            }
        )

        ctx = PluginContext(
            run_id=run_id,
            config={},
            landscape=recorder,
            state_id=state_id,
        )

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.post.side_effect = httpx.HTTPStatusError(
                "Rate Limit",
                request=MagicMock(),
                response=MagicMock(status_code=429),
            )
            mock_client_class.return_value.__enter__ = MagicMock(return_value=mock_client)
            mock_client_class.return_value.__exit__ = MagicMock(return_value=None)

            result = transform.process({"text": "test"}, ctx)

        # Verify retryable error
        assert result.status == "error"
        assert result.retryable is True


class TestAzureBatchLLMTransformIntegration:
    """Integration tests for Azure Batch LLM transform."""

    @pytest.fixture
    def ctx_with_checkpoint(self) -> PluginContext:
        """Create plugin context for testing.

        PluginContext now has native checkpoint support via
        get_checkpoint/update_checkpoint/clear_checkpoint methods.
        """
        return PluginContext(run_id="test-run", config={})

    @pytest.fixture
    def transform(self) -> AzureBatchLLMTransform:
        """Create a basic Azure batch transform."""
        return AzureBatchLLMTransform(
            {
                "deployment_name": "gpt-4o-batch",
                "endpoint": "https://test.openai.azure.com",
                "api_key": "test-key",
                "template": "Process: {{ row.text }}",
                "schema": DYNAMIC_SCHEMA,
            }
        )

    def test_batch_submit_and_checkpoint_flow(self, ctx_with_checkpoint: PluginContext, transform: AzureBatchLLMTransform) -> None:
        """Verify batch submission creates checkpoint with batch_id."""
        # Mock Azure client
        mock_client = Mock()
        mock_file = Mock()
        mock_file.id = "file-abc123"
        mock_client.files.create.return_value = mock_file

        mock_batch = Mock()
        mock_batch.id = "batch-xyz789"
        mock_client.batches.create.return_value = mock_batch

        transform._client = mock_client

        rows = [
            {"text": "Item 1"},
            {"text": "Item 2"},
            {"text": "Item 3"},
        ]

        # Should raise BatchPendingError after submission
        with pytest.raises(BatchPendingError) as exc_info:
            transform.process(rows, ctx_with_checkpoint)

        error = exc_info.value
        assert error.batch_id == "batch-xyz789"
        assert error.status == "submitted"

        # Verify checkpoint was saved
        checkpoint = ctx_with_checkpoint._checkpoint  # type: ignore[attr-defined]
        assert checkpoint["batch_id"] == "batch-xyz789"
        assert checkpoint["input_file_id"] == "file-abc123"
        assert checkpoint["row_count"] == 3
        assert "row_mapping" in checkpoint
        assert len(checkpoint["row_mapping"]) == 3
        assert "submitted_at" in checkpoint

    def test_batch_resume_and_completion_flow(self, transform: AzureBatchLLMTransform) -> None:
        """Verify batch resume downloads results and maps to correct rows."""
        from datetime import UTC, datetime

        # Set up context with existing checkpoint for resume test
        ctx = PluginContext(run_id="test-run", config={})
        recent_timestamp = datetime.now(UTC).isoformat()
        ctx._checkpoint.update(
            {
                "batch_id": "batch-xyz789",
                "input_file_id": "file-abc123",
                "row_mapping": {
                    "row-0-aaa": 0,
                    "row-1-bbb": 1,
                    "row-2-ccc": 2,
                },
                "template_errors": [],
                "submitted_at": recent_timestamp,
                "row_count": 3,
            }
        )

        # Mock completed batch
        mock_client = Mock()
        mock_batch = Mock()
        mock_batch.id = "batch-xyz789"
        mock_batch.status = "completed"
        mock_batch.output_file_id = "output-file-456"
        mock_client.batches.retrieve.return_value = mock_batch

        # Mock output file with results (returned in different order)
        output_lines = [
            json.dumps(
                {
                    "custom_id": "row-2-ccc",
                    "response": {
                        "body": {
                            "choices": [{"message": {"content": "Result C"}}],
                            "usage": {"prompt_tokens": 3, "completion_tokens": 3},
                        }
                    },
                }
            ),
            json.dumps(
                {
                    "custom_id": "row-0-aaa",
                    "response": {
                        "body": {
                            "choices": [{"message": {"content": "Result A"}}],
                            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                        }
                    },
                }
            ),
            json.dumps(
                {
                    "custom_id": "row-1-bbb",
                    "response": {
                        "body": {
                            "choices": [{"message": {"content": "Result B"}}],
                            "usage": {"prompt_tokens": 2, "completion_tokens": 2},
                        }
                    },
                }
            ),
        ]
        output_content = Mock()
        output_content.text = "\n".join(output_lines)
        mock_client.files.content.return_value = output_content

        transform._client = mock_client

        rows = [
            {"text": "Item A"},
            {"text": "Item B"},
            {"text": "Item C"},
        ]

        result = transform.process(rows, ctx)

        # Verify result
        assert result.status == "success"
        assert result.rows is not None
        assert len(result.rows) == 3

        # Results should be in original input order
        assert result.rows[0]["llm_response"] == "Result A"
        assert result.rows[0]["text"] == "Item A"
        assert result.rows[1]["llm_response"] == "Result B"
        assert result.rows[1]["text"] == "Item B"
        assert result.rows[2]["llm_response"] == "Result C"
        assert result.rows[2]["text"] == "Item C"

        # Checkpoint should be cleared
        assert ctx._checkpoint == {}  # type: ignore[attr-defined]

    def test_batch_partial_template_failures(self, ctx_with_checkpoint: PluginContext, transform: AzureBatchLLMTransform) -> None:
        """Verify partial template failures are tracked and results assembled."""

        # Mock Azure client for submission
        mock_client = Mock()
        mock_file = Mock()
        mock_file.id = "file-abc123"
        mock_client.files.create.return_value = mock_file

        mock_batch = Mock()
        mock_batch.id = "batch-xyz789"
        mock_client.batches.create.return_value = mock_batch

        transform._client = mock_client

        # Mix of valid and invalid rows (missing required template field)
        rows = [
            {"text": "valid1"},
            {"other_field": "missing_text"},  # Will fail template
            {"text": "valid2"},
        ]

        with pytest.raises(BatchPendingError):
            transform.process(rows, ctx_with_checkpoint)

        # Checkpoint should track template errors
        checkpoint = ctx_with_checkpoint._checkpoint  # type: ignore[attr-defined]
        assert "template_errors" in checkpoint
        assert len(checkpoint["template_errors"]) == 1
        assert checkpoint["template_errors"][0][0] == 1  # Index of failed row

        # row_mapping should only have valid rows
        assert len(checkpoint["row_mapping"]) == 2

    def test_batch_api_error_per_row_handled(self, transform: AzureBatchLLMTransform) -> None:
        """Verify per-row API errors are included in results."""
        from datetime import UTC, datetime

        ctx = PluginContext(run_id="test-run", config={})
        # Pre-populate checkpoint for resume test
        recent_timestamp = datetime.now(UTC).isoformat()
        ctx._checkpoint.update(
            {
                "batch_id": "batch-xyz789",
                "input_file_id": "file-abc123",
                "row_mapping": {
                    "row-0-aaa": 0,
                    "row-1-bbb": 1,
                },
                "template_errors": [],
                "submitted_at": recent_timestamp,
                "row_count": 2,
            }
        )

        mock_client = Mock()
        mock_batch = Mock()
        mock_batch.id = "batch-xyz789"
        mock_batch.status = "completed"
        mock_batch.output_file_id = "output-file-456"
        mock_client.batches.retrieve.return_value = mock_batch

        # One success, one error
        output_lines = [
            json.dumps(
                {
                    "custom_id": "row-0-aaa",
                    "response": {
                        "body": {
                            "choices": [{"message": {"content": "Success"}}],
                            "usage": {},
                        }
                    },
                }
            ),
            json.dumps(
                {
                    "custom_id": "row-1-bbb",
                    "error": {"code": "content_filter", "message": "Content blocked"},
                }
            ),
        ]
        output_content = Mock()
        output_content.text = "\n".join(output_lines)
        mock_client.files.content.return_value = output_content

        transform._client = mock_client

        rows = [{"text": "good"}, {"text": "blocked"}]

        result = transform.process(rows, ctx)

        assert result.status == "success"
        assert result.rows is not None
        assert len(result.rows) == 2

        # First row succeeded
        assert result.rows[0]["llm_response"] == "Success"

        # Second row has error
        assert result.rows[1]["llm_response"] is None
        assert "llm_response_error" in result.rows[1]
        assert result.rows[1]["llm_response_error"]["reason"] == "api_error"


class TestAuditedLLMClientIntegration:
    """Integration tests for AuditedLLMClient with real recorder."""

    @pytest.fixture
    def recorder(self) -> LandscapeRecorder:
        """Create recorder with in-memory DB."""
        db = LandscapeDB.in_memory()
        return LandscapeRecorder(db)

    @pytest.fixture
    def state_id(self, recorder: LandscapeRecorder) -> str:
        """Create a node state for testing."""
        schema = SchemaConfig.from_dict({"fields": "dynamic"})
        run = recorder.begin_run(config={}, canonical_version="v1")
        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="test",
            node_type="transform",
            plugin_version="1.0",
            config={},
            schema_config=schema,
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=node.node_id,
            row_index=0,
            data={},
        )
        token = recorder.create_token(row_id=row.row_id)
        state = recorder.begin_node_state(
            token_id=token.token_id,
            node_id=node.node_id,
            step_index=0,
            input_data={},
        )
        return state.state_id

    def test_successful_call_recorded_with_all_fields(self, recorder: LandscapeRecorder, state_id: str) -> None:
        """Verify successful call records request, response, and latency."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.model = "gpt-4"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.model_dump.return_value = {"id": "resp-123"}
        mock_client.chat.completions.create.return_value = mock_response

        audited_client = AuditedLLMClient(
            recorder=recorder,
            state_id=state_id,
            underlying_client=mock_client,
        )

        response = audited_client.chat_completion(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.7,
        )

        # Verify response
        assert response.content == "Test response"
        assert response.model == "gpt-4"
        assert response.usage["prompt_tokens"] == 10

        # Verify audit trail
        calls = recorder.get_calls(state_id)
        assert len(calls) == 1
        call = calls[0]
        assert call.call_type == CallType.LLM
        assert call.status == CallStatus.SUCCESS
        assert call.request_hash is not None
        assert call.response_hash is not None
        assert call.latency_ms is not None
        assert call.latency_ms > 0
        assert call.error_json is None

    def test_multiple_calls_indexed_correctly(self, recorder: LandscapeRecorder, state_id: str) -> None:
        """Verify multiple calls have sequential indices."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.model = "gpt-4"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 3
        mock_response.model_dump.return_value = {}
        mock_client.chat.completions.create.return_value = mock_response

        audited_client = AuditedLLMClient(
            recorder=recorder,
            state_id=state_id,
            underlying_client=mock_client,
        )

        # Make 3 calls
        audited_client.chat_completion(model="gpt-4", messages=[{"role": "user", "content": "First"}])
        audited_client.chat_completion(model="gpt-4", messages=[{"role": "user", "content": "Second"}])
        audited_client.chat_completion(model="gpt-4", messages=[{"role": "user", "content": "Third"}])

        # Verify indices
        calls = recorder.get_calls(state_id)
        assert len(calls) == 3
        assert calls[0].call_index == 0
        assert calls[1].call_index == 1
        assert calls[2].call_index == 2

    def test_error_call_recorded_with_error_details(self, recorder: LandscapeRecorder, state_id: str) -> None:
        """Verify failed calls record error details."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API down")

        audited_client = AuditedLLMClient(
            recorder=recorder,
            state_id=state_id,
            underlying_client=mock_client,
        )

        with pytest.raises(LLMClientError) as exc_info:
            audited_client.chat_completion(model="gpt-4", messages=[{"role": "user", "content": "Test"}])

        assert "API down" in str(exc_info.value)

        # Verify error recorded
        calls = recorder.get_calls(state_id)
        assert len(calls) == 1
        call = calls[0]
        assert call.status == CallStatus.ERROR
        assert call.error_json is not None
        assert "API down" in call.error_json
        assert call.response_hash is None  # No response on error


class TestOpenRouterPooledExecution:
    """Integration tests for OpenRouter with pooled execution."""

    @pytest.fixture
    def recorder(self) -> LandscapeRecorder:
        """Create recorder with in-memory DB."""
        db = LandscapeDB.in_memory()
        return LandscapeRecorder(db)

    def test_pool_size_1_uses_sequential_processing(self, recorder: LandscapeRecorder) -> None:
        """pool_size=1 should use existing sequential logic."""
        from unittest.mock import patch

        # Setup state
        schema = SchemaConfig.from_dict({"fields": "dynamic"})
        run = recorder.begin_run(config={}, canonical_version="v1")
        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="openrouter_llm",
            node_type="transform",
            plugin_version="1.0",
            config={},
            schema_config=schema,
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=node.node_id,
            row_index=0,
            data={"text": "test"},
        )
        token = recorder.create_token(row_id=row.row_id)
        state = recorder.begin_node_state(
            token_id=token.token_id,
            node_id=node.node_id,
            step_index=0,
            input_data={"text": "test"},
        )

        transform = OpenRouterLLMTransform(
            {
                "model": "anthropic/claude-3-opus",
                "template": "{{ row.text }}",
                "api_key": "test-key",
                "schema": {"fields": "dynamic"},
                "pool_size": 1,  # Sequential
            }
        )

        # Verify no executor created
        assert transform._executor is None

        ctx = PluginContext(
            run_id=run.run_id,
            config={},
            landscape=recorder,
            state_id=state.state_id,
        )

        # Mock HTTP and verify single-row processing still works
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "result"}}],
            "usage": {},
        }
        mock_response.raise_for_status = MagicMock()
        mock_response.content = b""
        mock_response.text = ""

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.post.return_value = mock_response
            mock_client_class.return_value.__enter__ = MagicMock(return_value=mock_client)
            mock_client_class.return_value.__exit__ = MagicMock(return_value=None)

            result = transform.process({"text": "hello"}, ctx)

        assert result.status == "success"

    def test_pool_size_greater_than_1_creates_executor(self) -> None:
        """pool_size > 1 should create pooled executor."""
        transform = OpenRouterLLMTransform(
            {
                "model": "anthropic/claude-3-opus",
                "template": "{{ row.text }}",
                "api_key": "test-key",
                "schema": {"fields": "dynamic"},
                "pool_size": 5,
            }
        )

        assert transform._executor is not None
        assert transform._executor.pool_size == 5

        transform.close()


class TestPooledExecutionIntegration:
    """Full integration tests for pooled LLM execution.

    Note: Tests that involve capacity error retries use a mock process function
    instead of _process_single_with_state because retries with the same state_id
    create a call_index collision in the audit trail (each retry creates a new
    AuditedHTTPClient instance with call_index starting at 0).

    Tests that don't involve retries (mixed results, single success) can use the
    full _process_single_with_state path with file-based SQLite for thread safety.
    """

    def test_batch_with_simulated_capacity_errors(self) -> None:
        """Verify pooled execution handles capacity errors correctly.

        Uses mock process function because capacity retries with the same
        state_id would cause call_index collisions in the audit trail.
        """
        import random
        from threading import Lock

        from elspeth.plugins.llm.capacity_errors import CapacityError
        from elspeth.plugins.llm.pooled_executor import RowContext

        # Seed random for reproducibility
        random.seed(42)

        # Create transform with pooling
        transform = OpenRouterLLMTransform(
            {
                "model": "test-model",
                "template": "{{ row.text }}",
                "api_key": "test-key",
                "schema": {"fields": "dynamic"},
                "pool_size": 3,
                "max_dispatch_delay_ms": 100,
                "max_capacity_retry_seconds": 10,
            }
        )

        # Track calls per row
        call_counts: dict[int, int] = {}
        lock = Lock()

        def mock_process(row: dict[str, Any], state_id: str) -> TransformResult:
            """Mock process function simulating 50% capacity error rate on first call."""
            idx = int(row["text"].split("_")[1])
            with lock:
                call_counts[idx] = call_counts.get(idx, 0) + 1
                current_count = call_counts[idx]

            # First call has 50% chance of capacity error
            if current_count == 1 and random.random() < 0.5:
                raise CapacityError(429, "Rate limited")

            # Success
            return TransformResult.success(
                {
                    **row,
                    "llm_response": "done",
                    "llm_response_usage": {},
                }
            )

        # Create row contexts
        row_contexts = [RowContext(row={"text": f"row_{i}"}, state_id=f"state_{i}", row_index=i) for i in range(5)]

        # Execute batch through pooled executor
        assert transform._executor is not None

        results = transform._executor.execute_batch(
            contexts=row_contexts,
            process_fn=mock_process,
        )

        # All should succeed (capacity errors were retried)
        assert all(r.status == "success" for r in results), f"Failed: {[r for r in results if r.status != 'success']}"
        assert len(results) == 5

        # Verify results are in correct order (reorder buffer working)
        for i, result in enumerate(results):
            assert result.row is not None
            assert result.row["text"] == f"row_{i}"
            assert result.row["llm_response"] == "done"

        # Get stats to verify capacity errors were handled
        stats = transform._executor.get_stats()
        # With seed 42 and 50% chance, some rows should have hit capacity errors
        assert stats["pool_stats"]["successes"] >= 5  # At least all rows eventually succeeded

        transform.close()

    def test_batch_capacity_retry_timeout(self) -> None:
        """Verify capacity errors that exceed timeout return error result.

        Uses mock process function because capacity retries with the same
        state_id would cause call_index collisions in the audit trail.
        """
        from elspeth.plugins.llm.capacity_errors import CapacityError
        from elspeth.plugins.llm.pooled_executor import RowContext

        # Create transform with very short timeout for testing
        transform = OpenRouterLLMTransform(
            {
                "model": "test-model",
                "template": "{{ row.text }}",
                "api_key": "test-key",
                "schema": {"fields": "dynamic"},
                "pool_size": 2,
                "max_dispatch_delay_ms": 10,
                "max_capacity_retry_seconds": 1,  # Very short timeout
            }
        )

        def mock_process_always_fails(row: dict[str, Any], state_id: str) -> TransformResult:
            """Always raise capacity error to test timeout."""
            raise CapacityError(429, "Rate limited")

        row_context = RowContext(row={"text": "test_timeout"}, state_id="state_0", row_index=0)

        assert transform._executor is not None

        results = transform._executor.execute_batch(
            contexts=[row_context],
            process_fn=mock_process_always_fails,
        )

        # Should get capacity_retry_timeout error
        assert len(results) == 1
        result = results[0]
        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "capacity_retry_timeout"
        assert result.reason["status_code"] == 429
        assert result.retryable is False

        transform.close()

    def test_batch_mixed_results(self, tmp_path: Any) -> None:
        """Verify batch handles mix of success and non-capacity errors."""
        from unittest.mock import patch

        from elspeth.plugins.llm.pooled_executor import RowContext

        # Create transform with pooling
        transform = OpenRouterLLMTransform(
            {
                "model": "test-model",
                "template": "{{ row.text }}",
                "api_key": "test-key",
                "schema": {"fields": "dynamic"},
                "pool_size": 3,
                "max_dispatch_delay_ms": 10,
            }
        )

        def mock_post_mixed(*args: Any, **kwargs: Any) -> MagicMock:
            """Return success for even rows, 500 error for odd rows."""
            body = kwargs.get("json", {})
            messages = body.get("messages", [])

            if messages:
                content = messages[-1].get("content", "")
                idx = int(content.split("_")[1]) if "_" in content else 0

                if idx % 2 == 1:
                    # Odd rows get 500 error (not capacity error)
                    response = MagicMock(spec=httpx.Response)
                    response.status_code = 500
                    raise httpx.HTTPStatusError(
                        "Server Error",
                        request=MagicMock(),
                        response=response,
                    )

            # Success response
            response = MagicMock(spec=httpx.Response)
            response.status_code = 200
            response.headers = {"content-type": "application/json"}
            response.json.return_value = {
                "choices": [{"message": {"content": "success"}}],
                "usage": {},
            }
            response.raise_for_status = MagicMock()
            response.content = b""
            response.text = ""
            return response

        # Use file-based SQLite for thread safety
        db_path = tmp_path / "test_mixed.db"
        db = LandscapeDB.from_url(f"sqlite:///{db_path}")
        recorder = LandscapeRecorder(db)
        schema = SchemaConfig.from_dict({"fields": "dynamic"})
        run = recorder.begin_run(config={}, canonical_version="v1")
        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="openrouter_llm",
            node_type="transform",
            plugin_version="1.0",
            config={},
            schema_config=schema,
        )

        ctx = PluginContext(
            run_id=run.run_id,
            config={},
            landscape=recorder,
            state_id=None,
        )
        transform.on_start(ctx)

        # Create rows
        rows = [{"text": f"row_{i}"} for i in range(4)]
        row_contexts: list[RowContext] = []

        for i, row in enumerate(rows):
            row_rec = recorder.create_row(
                run_id=run.run_id,
                source_node_id=node.node_id,
                row_index=i,
                data=row,
            )
            token = recorder.create_token(row_id=row_rec.row_id)
            state = recorder.begin_node_state(
                token_id=token.token_id,
                node_id=node.node_id,
                step_index=0,
                input_data=row,
            )
            row_contexts.append(RowContext(row=row, state_id=state.state_id, row_index=i))

        assert transform._executor is not None

        with patch("httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.post = mock_post_mixed
            mock_client_class.return_value.__enter__ = MagicMock(return_value=mock_client)
            mock_client_class.return_value.__exit__ = MagicMock(return_value=None)

            results = transform._executor.execute_batch(
                contexts=row_contexts,
                process_fn=transform._process_single_with_state,
            )

        # Verify results in order
        assert len(results) == 4

        # Even rows (0, 2) should succeed
        assert results[0].status == "success"
        assert results[0].row is not None
        assert results[0].row["text"] == "row_0"

        assert results[2].status == "success"
        assert results[2].row is not None
        assert results[2].row["text"] == "row_2"

        # Odd rows (1, 3) should fail with non-capacity error
        assert results[1].status == "error"
        assert results[1].reason is not None
        assert results[1].reason["reason"] == "api_call_failed"
        assert results[1].retryable is False

        assert results[3].status == "error"
        assert results[3].reason is not None
        assert results[3].reason["reason"] == "api_call_failed"
        assert results[3].retryable is False

        transform.close()
