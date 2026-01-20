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

from elspeth.contracts import CallStatus, CallType
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
    """Concrete LLM transform for testing."""

    name = "concrete_llm"


class TestLLMTransformIntegration:
    """Integration tests for LLM transforms with real audit trail."""

    @pytest.fixture
    def recorder(self) -> LandscapeRecorder:
        """Create recorder with in-memory DB."""
        db = LandscapeDB.in_memory()
        return LandscapeRecorder(db)

    @pytest.fixture
    def setup_state(
        self, recorder: LandscapeRecorder
    ) -> tuple[str, str, str, str, str]:
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
        run_id, node_id, row_id, token_id, state_id = setup_state

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
                "template": "Say hello to {{ name }}!",
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

    def test_audit_trail_records_template_hashes(
        self, recorder: LandscapeRecorder, setup_state: tuple[str, str, str, str, str]
    ) -> None:
        """Verify template_hash and variables_hash are in output."""
        run_id, node_id, row_id, token_id, state_id = setup_state

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
                "template": "Analyze: {{ text }}",
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

    def test_api_error_recorded_in_audit_trail(
        self, recorder: LandscapeRecorder, setup_state: tuple[str, str, str, str, str]
    ) -> None:
        """Verify API errors are recorded in audit trail."""
        run_id, node_id, row_id, token_id, state_id = setup_state

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
                "template": "{{ text }}",
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

    def test_rate_limit_error_is_retryable(
        self, recorder: LandscapeRecorder, setup_state: tuple[str, str, str, str, str]
    ) -> None:
        """Verify rate limit errors are marked retryable."""
        run_id, node_id, row_id, token_id, state_id = setup_state

        # Mock client that raises rate limit error
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception(
            "Error 429: Rate limit exceeded"
        )

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
                "template": "{{ text }}",
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

    def test_system_prompt_included_when_configured(
        self, recorder: LandscapeRecorder, setup_state: tuple[str, str, str, str, str]
    ) -> None:
        """Verify system prompt is included in API call."""
        run_id, node_id, row_id, token_id, state_id = setup_state

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
                "template": "{{ text }}",
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

    def test_http_client_call_and_response_parsing(
        self, recorder: LandscapeRecorder
    ) -> None:
        """Verify OpenRouter uses HTTP client and parses response."""
        schema = SchemaConfig.from_dict({"fields": "dynamic"})
        run = recorder.begin_run(config={}, canonical_version="v1")
        # Register node - needed to have a valid run context
        _ = recorder.register_node(
            run_id=run.run_id,
            plugin_name="openrouter_llm",
            node_type="transform",
            plugin_version="1.0",
            config={},
            schema_config=schema,
        )

        # Create transform
        transform = OpenRouterLLMTransform(
            {
                "model": "anthropic/claude-3-opus",
                "template": "Analyze: {{ text }}",
                "api_key": "test-api-key",
                "schema": DYNAMIC_SCHEMA,
            }
        )

        # Mock HTTP client
        mock_http_client = MagicMock()
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Analysis complete"}}],
            "usage": {"prompt_tokens": 15, "completion_tokens": 10},
            "model": "anthropic/claude-3-opus",
        }
        mock_response.raise_for_status = MagicMock()
        mock_http_client.post.return_value = mock_response

        ctx = PluginContext(run_id=run.run_id, config={})
        ctx.http_client = mock_http_client

        result = transform.process({"text": "Hello world"}, ctx)

        # Verify result
        assert result.status == "success"
        assert result.row is not None
        assert result.row["llm_response"] == "Analysis complete"
        assert result.row["llm_response_usage"]["prompt_tokens"] == 15
        assert result.row["llm_response_model"] == "anthropic/claude-3-opus"

        # Verify HTTP call was made
        mock_http_client.post.assert_called_once()
        call_args = mock_http_client.post.call_args

        # Check URL
        assert "openrouter.ai" in call_args.args[0]
        assert "chat/completions" in call_args.args[0]

        # Check headers include auth
        assert "Authorization" in call_args.kwargs["headers"]
        assert "Bearer test-api-key" in call_args.kwargs["headers"]["Authorization"]

        # Check request body
        body = call_args.kwargs["json"]
        assert body["model"] == "anthropic/claude-3-opus"
        assert len(body["messages"]) == 1
        assert body["messages"][0]["content"] == "Analyze: Hello world"

    def test_http_error_returns_transform_error(
        self, recorder: LandscapeRecorder
    ) -> None:
        """Verify HTTP errors are handled gracefully."""
        run = recorder.begin_run(config={}, canonical_version="v1")

        transform = OpenRouterLLMTransform(
            {
                "model": "anthropic/claude-3-opus",
                "template": "{{ text }}",
                "api_key": "test-api-key",
                "schema": DYNAMIC_SCHEMA,
            }
        )

        # Mock HTTP client that raises status error
        mock_http_client = MagicMock()
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server Error",
            request=MagicMock(),
            response=mock_response,
        )
        mock_http_client.post.return_value = mock_response

        ctx = PluginContext(run_id=run.run_id, config={})
        ctx.http_client = mock_http_client

        result = transform.process({"text": "test"}, ctx)

        # Verify error result
        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "api_call_failed"
        assert result.retryable is False  # 500 is not rate limit

    def test_rate_limit_http_error_is_retryable(
        self, recorder: LandscapeRecorder
    ) -> None:
        """Verify 429 HTTP errors are marked retryable."""
        run = recorder.begin_run(config={}, canonical_version="v1")

        transform = OpenRouterLLMTransform(
            {
                "model": "anthropic/claude-3-opus",
                "template": "{{ text }}",
                "api_key": "test-api-key",
                "schema": DYNAMIC_SCHEMA,
            }
        )

        # Mock HTTP client that returns 429
        mock_http_client = MagicMock()
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 429
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Rate Limit",
            request=MagicMock(),
            response=mock_response,
        )
        mock_http_client.post.return_value = mock_response

        ctx = PluginContext(run_id=run.run_id, config={})
        ctx.http_client = mock_http_client

        result = transform.process({"text": "test"}, ctx)

        # Verify retryable error
        assert result.status == "error"
        assert result.retryable is True


class TestAzureBatchLLMTransformIntegration:
    """Integration tests for Azure Batch LLM transform."""

    @pytest.fixture
    def ctx_with_checkpoint(self) -> PluginContext:
        """Create plugin context with checkpoint support."""

        ctx = PluginContext(run_id="test-run", config={})
        ctx._checkpoint: dict[str, Any] = {}
        ctx.get_checkpoint = lambda: ctx._checkpoint or None  # type: ignore[method-assign]
        ctx.update_checkpoint = lambda d: ctx._checkpoint.update(d)  # type: ignore[method-assign]
        ctx.clear_checkpoint = lambda: ctx._checkpoint.clear()  # type: ignore[method-assign]
        return ctx

    @pytest.fixture
    def transform(self) -> AzureBatchLLMTransform:
        """Create a basic Azure batch transform."""
        return AzureBatchLLMTransform(
            {
                "deployment_name": "gpt-4o-batch",
                "endpoint": "https://test.openai.azure.com",
                "api_key": "test-key",
                "template": "Process: {{ text }}",
                "schema": DYNAMIC_SCHEMA,
            }
        )

    def test_batch_submit_and_checkpoint_flow(
        self, ctx_with_checkpoint: PluginContext, transform: AzureBatchLLMTransform
    ) -> None:
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
        checkpoint = ctx_with_checkpoint._checkpoint
        assert checkpoint["batch_id"] == "batch-xyz789"
        assert checkpoint["input_file_id"] == "file-abc123"
        assert checkpoint["row_count"] == 3
        assert "row_mapping" in checkpoint
        assert len(checkpoint["row_mapping"]) == 3
        assert "submitted_at" in checkpoint

    def test_batch_resume_and_completion_flow(
        self, transform: AzureBatchLLMTransform
    ) -> None:
        """Verify batch resume downloads results and maps to correct rows."""
        from datetime import UTC, datetime

        # Set up context with existing checkpoint
        ctx = PluginContext(run_id="test-run", config={})
        recent_timestamp = datetime.now(UTC).isoformat()
        ctx._checkpoint: dict[str, Any] = {
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
        ctx.get_checkpoint = lambda: ctx._checkpoint if ctx._checkpoint else None  # type: ignore[method-assign]
        ctx.update_checkpoint = lambda d: ctx._checkpoint.update(d)  # type: ignore[method-assign]
        ctx.clear_checkpoint = lambda: ctx._checkpoint.clear()  # type: ignore[method-assign]

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
        assert ctx._checkpoint == {}

    def test_batch_partial_template_failures(
        self, ctx_with_checkpoint: PluginContext, transform: AzureBatchLLMTransform
    ) -> None:
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
        checkpoint = ctx_with_checkpoint._checkpoint
        assert "template_errors" in checkpoint
        assert len(checkpoint["template_errors"]) == 1
        assert checkpoint["template_errors"][0][0] == 1  # Index of failed row

        # row_mapping should only have valid rows
        assert len(checkpoint["row_mapping"]) == 2

    def test_batch_api_error_per_row_handled(
        self, transform: AzureBatchLLMTransform
    ) -> None:
        """Verify per-row API errors are included in results."""
        from datetime import UTC, datetime

        ctx = PluginContext(run_id="test-run", config={})
        recent_timestamp = datetime.now(UTC).isoformat()
        ctx._checkpoint: dict[str, Any] = {
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
        ctx.get_checkpoint = lambda: ctx._checkpoint if ctx._checkpoint else None  # type: ignore[method-assign]
        ctx.update_checkpoint = lambda d: ctx._checkpoint.update(d)  # type: ignore[method-assign]
        ctx.clear_checkpoint = lambda: ctx._checkpoint.clear()  # type: ignore[method-assign]

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

    def test_successful_call_recorded_with_all_fields(
        self, recorder: LandscapeRecorder, state_id: str
    ) -> None:
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

    def test_multiple_calls_indexed_correctly(
        self, recorder: LandscapeRecorder, state_id: str
    ) -> None:
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
        audited_client.chat_completion(
            model="gpt-4", messages=[{"role": "user", "content": "First"}]
        )
        audited_client.chat_completion(
            model="gpt-4", messages=[{"role": "user", "content": "Second"}]
        )
        audited_client.chat_completion(
            model="gpt-4", messages=[{"role": "user", "content": "Third"}]
        )

        # Verify indices
        calls = recorder.get_calls(state_id)
        assert len(calls) == 3
        assert calls[0].call_index == 0
        assert calls[1].call_index == 1
        assert calls[2].call_index == 2

    def test_error_call_recorded_with_error_details(
        self, recorder: LandscapeRecorder, state_id: str
    ) -> None:
        """Verify failed calls record error details."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API down")

        audited_client = AuditedLLMClient(
            recorder=recorder,
            state_id=state_id,
            underlying_client=mock_client,
        )

        with pytest.raises(LLMClientError) as exc_info:
            audited_client.chat_completion(
                model="gpt-4", messages=[{"role": "user", "content": "Test"}]
            )

        assert "API down" in str(exc_info.value)

        # Verify error recorded
        calls = recorder.get_calls(state_id)
        assert len(calls) == 1
        call = calls[0]
        assert call.status == CallStatus.ERROR
        assert call.error_json is not None
        assert "API down" in call.error_json
        assert call.response_hash is None  # No response on error
