# tests/contracts/transform_contracts/test_azure_multi_query_contract.py
"""Contract tests for Azure Multi-Query LLM transform.

Note: Row-based contract tests (TransformContractPropertyTestBase) were removed because
AzureMultiQueryLLMTransform uses BatchTransformMixin and doesn't support process(). The
attribute tests (name, schema, determinism) are now included in BatchTransformContractTestBase.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock, patch

import pytest

from elspeth.plugins.batching.mixin import BatchTransformMixin
from elspeth.plugins.context import PluginContext
from elspeth.plugins.llm.azure_multi_query import AzureMultiQueryLLMTransform

from .test_batch_transform_protocol import BatchTransformContractTestBase


def _make_mock_response(content: str = '{"score": 85, "rationale": "test"}') -> Mock:
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content=content))]
    mock_response.model = "gpt-4o"
    mock_response.usage = Mock(prompt_tokens=10, completion_tokens=5)
    mock_response.model_dump = Mock(return_value={})
    return mock_response


def _make_mock_context() -> Mock:
    ctx = Mock(spec=PluginContext)
    ctx.run_id = "test-run-001"
    ctx.state_id = "state-001"
    ctx.landscape = Mock()
    ctx.landscape.record_call = Mock()
    ctx.landscape.allocate_call_index = Mock(return_value=0)
    return ctx


@pytest.fixture(autouse=True)
def mock_azure_openai():
    with patch("openai.AzureOpenAI") as mock_azure_class:
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = _make_mock_response()
        mock_azure_class.return_value = mock_client
        yield mock_client


class TestAzureMultiQueryLLMSpecific:
    def test_query_expansion_matches_cross_product(self) -> None:
        transform = AzureMultiQueryLLMTransform(
            {
                "deployment_name": "gpt-4o",
                "endpoint": "https://test.openai.azure.com",
                "api_key": "test-key",
                "template": "{{ row.input_1 }}",
                "case_studies": [
                    {"name": "cs1", "input_fields": ["a"]},
                    {"name": "cs2", "input_fields": ["b"]},
                ],
                "criteria": [
                    {"name": "crit1"},
                    {"name": "crit2"},
                    {"name": "crit3"},
                ],
                "response_format": "standard",
                "output_mapping": {"score": {"suffix": "score", "type": "integer"}},
                "schema": {"fields": "dynamic"},
                "on_error": "quarantine_sink",
                "required_input_fields": [],
            }
        )

        assert len(transform._query_specs) == 6

        prefixes = {s.output_prefix for s in transform._query_specs}
        assert prefixes == {
            "cs1_crit1",
            "cs1_crit2",
            "cs1_crit3",
            "cs2_crit1",
            "cs2_crit2",
            "cs2_crit3",
        }

    def test_creates_tokens_false(self) -> None:
        transform = AzureMultiQueryLLMTransform(
            {
                "deployment_name": "gpt-4o",
                "endpoint": "https://test.openai.azure.com",
                "api_key": "test-key",
                "template": "{{ row.input_1 }}",
                "case_studies": [{"name": "cs1", "input_fields": ["a"]}],
                "criteria": [{"name": "crit1"}],
                "response_format": "standard",
                "output_mapping": {"score": {"suffix": "score", "type": "integer"}},
                "schema": {"fields": "dynamic"},
                "on_error": "quarantine_sink",
                "required_input_fields": [],
            }
        )

        assert transform.creates_tokens is False


class TestAzureMultiQueryLLMAuditTrail:
    # NOTE: test_llm_call_recorded_in_audit was deleted because it tested the old
    # process() API. AzureMultiQueryLLMTransform now uses accept() via BatchTransformMixin.
    # Audit trail tests should use integration tests that exercise the full pipeline.

    def test_on_error_configuration_required(self) -> None:
        transform = AzureMultiQueryLLMTransform(
            {
                "deployment_name": "gpt-4o",
                "endpoint": "https://test.openai.azure.com",
                "api_key": "test-key",
                "template": "{{ row.input_1 }}",
                "case_studies": [{"name": "cs1", "input_fields": ["a"]}],
                "criteria": [{"name": "crit1"}],
                "response_format": "standard",
                "output_mapping": {"score": {"suffix": "score", "type": "integer"}},
                "schema": {"fields": "dynamic"},
                "on_error": "quarantine_sink",
                "required_input_fields": [],
            }
        )

        assert transform._on_error is not None


class TestAzureMultiQueryBatchContract(BatchTransformContractTestBase):
    """Verify Azure multi-query transform honors BatchTransformMixin contract."""

    @pytest.fixture
    def batch_transform(self) -> BatchTransformMixin:
        """Provide unconfigured transform (no connect_output yet)."""
        return AzureMultiQueryLLMTransform(
            {
                "deployment_name": "gpt-4o",
                "endpoint": "https://test.openai.azure.com",
                "api_key": "test-key",
                "template": "{{ row.input_1 }} {{ row.criterion.name }}",
                "case_studies": [
                    {"name": "cs1", "input_fields": ["cs1_a", "cs1_b"]},
                ],
                "criteria": [
                    {"name": "test_criterion", "code": "TEST"},
                ],
                "response_format": "standard",
                "output_mapping": {
                    "score": {"suffix": "score", "type": "integer"},
                    "rationale": {"suffix": "rationale", "type": "string"},
                },
                "schema": {"fields": "dynamic"},
                "on_error": "quarantine_sink",
                "required_input_fields": [],
            }
        )

    @pytest.fixture
    def valid_input(self) -> dict[str, Any]:
        return {"cs1_a": "value_a", "cs1_b": "value_b"}
