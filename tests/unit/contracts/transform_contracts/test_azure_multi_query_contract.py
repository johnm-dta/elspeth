# tests/unit/contracts/transform_contracts/test_azure_multi_query_contract.py
"""Contract tests for multi-query LLM transform (Azure provider).

Note: Row-based contract tests (TransformContractPropertyTestBase) were removed because
LLMTransform uses BatchTransformMixin and doesn't support process(). The
attribute tests (name, schema, determinism) are now included in BatchTransformContractTestBase.

Migrated from AzureMultiQueryLLMTransform to unified LLMTransform with provider="azure"
and queries dict format (T10 Phase B consolidation).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage

from elspeth.plugins.infrastructure.batching.mixin import BatchTransformMixin
from elspeth.plugins.transforms.llm.transform import LLMTransform

from ._azure_batch_helpers import submit_and_collect_single_result
from .test_batch_transform_protocol import BatchTransformContractTestBase, CollectingOutputPort


def _make_chat_completion(content: str = '{"score": 85, "rationale": "test"}') -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-test",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(role="assistant", content=content),
            )
        ],
        created=0,
        model="gpt-4o",
        object="chat.completion",
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


class _FakeCompletions:
    def __init__(self) -> None:
        self.content = '{"score": 85, "rationale": "test"}'
        self.error: Exception | None = None

    def create(self, **_: Any) -> ChatCompletion:
        if self.error is not None:
            raise self.error
        return _make_chat_completion(self.content)


class _FakeChat:
    def __init__(self) -> None:
        self.completions = _FakeCompletions()


class _FakeAzureOpenAI:
    def __init__(self) -> None:
        self.chat = _FakeChat()

    def close(self) -> None:
        pass


class TestMultiQueryLLMSpecific:
    def test_query_expansion_produces_expected_output_fields(self) -> None:
        """6 configured queries produce 6 sets of prefixed output fields."""
        transform = LLMTransform(
            {
                "provider": "azure",
                "deployment_name": "gpt-4o",
                "endpoint": "https://test.openai.azure.com",
                "api_key": "test-key",
                "prompt_template": "{{ row.text_content }}",
                "schema": {"mode": "observed"},
                "required_input_fields": [],
                "queries": {
                    "cs1_crit1": {
                        "input_fields": {"text_content": "a"},
                        "output_fields": [{"suffix": "score", "type": "integer"}],
                    },
                    "cs1_crit2": {
                        "input_fields": {"text_content": "a"},
                        "output_fields": [{"suffix": "rating", "type": "integer"}],
                    },
                    "cs1_crit3": {
                        "input_fields": {"text_content": "a"},
                        "output_fields": [{"suffix": "grade", "type": "integer"}],
                    },
                    "cs2_crit1": {
                        "input_fields": {"text_content": "b"},
                        "output_fields": [{"suffix": "eval_score", "type": "integer"}],
                    },
                    "cs2_crit2": {
                        "input_fields": {"text_content": "b"},
                        "output_fields": [{"suffix": "eval_rating", "type": "integer"}],
                    },
                    "cs2_crit3": {
                        "input_fields": {"text_content": "b"},
                        "output_fields": [{"suffix": "eval_grade", "type": "integer"}],
                    },
                },
            }
        )
        transform.on_error = "quarantine_sink"

        # Observable: declared_output_fields contains prefixed fields for each query
        declared = transform.declared_output_fields
        expected_fields = {
            "cs1_crit1_score",
            "cs1_crit2_rating",
            "cs1_crit3_grade",
            "cs2_crit1_eval_score",
            "cs2_crit2_eval_rating",
            "cs2_crit3_eval_grade",
        }
        assert expected_fields.issubset(declared), f"Missing expected output fields: {expected_fields - declared}"

    def test_creates_tokens_false(self) -> None:
        transform = LLMTransform(
            {
                "provider": "azure",
                "deployment_name": "gpt-4o",
                "endpoint": "https://test.openai.azure.com",
                "api_key": "test-key",
                "prompt_template": "{{ row.text_content }}",
                "schema": {"mode": "observed"},
                "required_input_fields": [],
                "queries": {
                    "cs1_crit1": {
                        "input_fields": {"text_content": "a"},
                        "output_fields": [{"suffix": "score", "type": "integer"}],
                    },
                },
            }
        )
        transform.on_error = "quarantine_sink"

        assert transform.creates_tokens is False


class TestMultiQueryBatchContract(BatchTransformContractTestBase):
    """Verify multi-query LLM transform honors BatchTransformMixin contract."""

    @pytest.fixture(autouse=True)
    def mock_azure_openai(self):
        with patch("openai.AzureOpenAI", autospec=True) as mock_azure_class:
            mock_client = _FakeAzureOpenAI()
            mock_azure_class.return_value = mock_client
            yield mock_client

    @pytest.fixture
    def batch_transform(self) -> BatchTransformMixin:
        """Provide unconfigured transform (no connect_output yet)."""
        t = LLMTransform(
            {
                "provider": "azure",
                "deployment_name": "gpt-4o",
                "endpoint": "https://test.openai.azure.com",
                "api_key": "test-key",
                "prompt_template": "{{ row.text_content }} {{ row.criterion_name }}",
                "schema": {"mode": "observed"},
                "required_input_fields": [],
                "queries": {
                    "cs1_test_criterion": {
                        "input_fields": {"text_content": "cs1_a", "criterion_name": "cs1_b"},
                        "output_fields": [
                            {"suffix": "score", "type": "integer"},
                            {"suffix": "rationale", "type": "string"},
                        ],
                    },
                },
            }
        )
        t.on_error = "quarantine_sink"
        return t

    @pytest.fixture
    def valid_input(self) -> dict[str, Any]:
        return {"cs1_a": "value_a", "cs1_b": "value_b"}

    def test_malformed_json_response_emits_non_retryable_query_error(
        self,
        started_transform: BatchTransformMixin,
        valid_input: dict[str, Any],
        mock_ctx_factory: Any,
        output_port: CollectingOutputPort,
        mock_azure_openai: _FakeAzureOpenAI,
    ) -> None:
        """Contract: malformed LLM JSON fails closed through the batch output path."""
        mock_azure_openai.chat.completions.content = "not valid JSON"

        result = submit_and_collect_single_result(
            started_transform,
            valid_input,
            mock_ctx_factory(),
            output_port,
        )

        assert result.status == "error"
        assert result.retryable is False
        assert result.row is None
        assert result.reason is not None
        assert result.reason["reason"] == "json_parse_failed"
        assert result.reason["query_name"] == "cs1_test_criterion"
        assert result.reason["query_index"] == 0
        assert result.reason["raw_response_preview"] == "not valid JSON"
        assert result.reason["discarded_successful_queries"] == 0

    def test_provider_auth_failure_emits_non_retryable_query_error(
        self,
        started_transform: BatchTransformMixin,
        valid_input: dict[str, Any],
        mock_ctx_factory: Any,
        output_port: CollectingOutputPort,
        mock_azure_openai: _FakeAzureOpenAI,
    ) -> None:
        """Contract: provider auth failures fail closed through on_error routing."""
        mock_azure_openai.chat.completions.error = RuntimeError("authentication failed")

        result = submit_and_collect_single_result(
            started_transform,
            valid_input,
            mock_ctx_factory(),
            output_port,
        )

        assert result.status == "error"
        assert result.retryable is False
        assert result.reason is not None
        assert result.reason["reason"] == "multi_query_failed"
        assert result.reason["failed_query_name"] == "cs1_test_criterion"
        assert result.reason["failed_query_index"] == 0
        # Persisted error text is the audit-safe constant — raw provider text
        # must not reach transform_errors.error_details_json (elspeth-5d17bcff15).
        assert result.reason["error"] == "LLM provider request failed"
        assert "authentication failed" not in result.reason["error"]
        assert result.reason["discarded_successful_queries"] == 0

    def test_token_usage_is_recorded_on_success(
        self,
        started_transform: BatchTransformMixin,
        valid_input: dict[str, Any],
        mock_ctx_factory: Any,
        output_port: CollectingOutputPort,
    ) -> None:
        """Contract: successful multi-query rows carry provider usage metadata."""
        result = submit_and_collect_single_result(
            started_transform,
            valid_input,
            mock_ctx_factory(),
            output_port,
        )

        assert result.status == "success"
        assert result.row is not None
        row = result.row.to_dict()
        assert row["cs1_test_criterion_llm_response_usage"] == {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }
        assert row["cs1_test_criterion_llm_response_model"] == "gpt-4o"
