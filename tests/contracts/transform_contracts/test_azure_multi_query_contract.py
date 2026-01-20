# tests/contracts/transform_contracts/test_azure_multi_query_contract.py
"""Contract tests for AzureMultiQueryLLMTransform plugin.

Verifies AzureMultiQueryLLMTransform honors the TransformProtocol contract.
These tests mock the Azure OpenAI client since contract tests verify interface
compliance, not API integration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import Mock, patch

import pytest

from elspeth.plugins.context import PluginContext
from elspeth.plugins.llm.azure_multi_query import AzureMultiQueryLLMTransform

from .test_transform_protocol import TransformContractPropertyTestBase

if TYPE_CHECKING:
    from elspeth.plugins.protocols import TransformProtocol


def _make_mock_response(content: str = '{"score": 85, "rationale": "test"}') -> Mock:
    """Create a mock Azure OpenAI response."""
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content=content))]
    mock_response.model = "gpt-4o"
    mock_response.usage = Mock(prompt_tokens=10, completion_tokens=5)
    mock_response.model_dump = Mock(return_value={})
    return mock_response


def _make_mock_context() -> Mock:
    """Create mock PluginContext with landscape recorder and state_id."""
    ctx = Mock(spec=PluginContext)
    ctx.run_id = "test-run-001"
    ctx.state_id = "state-001"
    ctx.landscape = Mock()
    ctx.landscape.record_external_call = Mock()
    return ctx


@pytest.fixture(autouse=True)
def mock_azure_openai():
    """Auto-mock Azure OpenAI client for all contract tests."""
    with patch("openai.AzureOpenAI") as mock_azure_class:
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = _make_mock_response()
        mock_azure_class.return_value = mock_client
        yield mock_client


class TestAzureMultiQueryLLMContract(TransformContractPropertyTestBase):
    """Contract tests for AzureMultiQueryLLMTransform.

    Inherits all standard transform contract tests plus adds
    multi-query-specific validation.
    """

    @pytest.fixture
    def transform(self) -> "TransformProtocol":
        """Return a configured transform instance."""
        t = AzureMultiQueryLLMTransform({
            "deployment_name": "gpt-4o",
            "endpoint": "https://test.openai.azure.com",
            "api_key": "test-key",
            "template": "{{ input_1 }} {{ criterion.name }}",
            "case_studies": [
                {"name": "cs1", "input_fields": ["cs1_a", "cs1_b"]},
            ],
            "criteria": [
                {"name": "test_criterion", "code": "TEST"},
            ],
            "response_format": "json",
            "output_mapping": {"score": "score", "rationale": "rationale"},
            "schema": {"fields": "dynamic"},
        })
        # Pre-initialize with context for lifecycle tests
        mock_ctx = _make_mock_context()
        t.on_start(mock_ctx)
        return t

    @pytest.fixture
    def valid_input(self) -> dict[str, Any]:
        """Return input that should process successfully (with mocked LLM)."""
        return {"cs1_a": "value_a", "cs1_b": "value_b"}

    @pytest.fixture
    def ctx(self) -> Mock:
        """Override context to provide mocked landscape and state_id."""
        return _make_mock_context()


class TestAzureMultiQueryLLMSpecific:
    """Multi-query-specific contract tests."""

    def test_query_expansion_matches_cross_product(self) -> None:
        """Query specs match case_studies × criteria cross-product."""
        transform = AzureMultiQueryLLMTransform({
            "deployment_name": "gpt-4o",
            "endpoint": "https://test.openai.azure.com",
            "api_key": "test-key",
            "template": "{{ input_1 }}",
            "case_studies": [
                {"name": "cs1", "input_fields": ["a"]},
                {"name": "cs2", "input_fields": ["b"]},
            ],
            "criteria": [
                {"name": "crit1"},
                {"name": "crit2"},
                {"name": "crit3"},
            ],
            "response_format": "json",
            "output_mapping": {"score": "score"},
            "schema": {"fields": "dynamic"},
        })

        # 2 case studies × 3 criteria = 6 queries
        assert len(transform._query_specs) == 6

        # Verify all combinations present
        prefixes = {s.output_prefix for s in transform._query_specs}
        assert prefixes == {
            "cs1_crit1", "cs1_crit2", "cs1_crit3",
            "cs2_crit1", "cs2_crit2", "cs2_crit3",
        }

    def test_is_batch_aware_true(self) -> None:
        """Transform declares batch awareness for aggregation support."""
        transform = AzureMultiQueryLLMTransform({
            "deployment_name": "gpt-4o",
            "endpoint": "https://test.openai.azure.com",
            "api_key": "test-key",
            "template": "{{ input_1 }}",
            "case_studies": [{"name": "cs1", "input_fields": ["a"]}],
            "criteria": [{"name": "crit1"}],
            "response_format": "json",
            "output_mapping": {"score": "score"},
            "schema": {"fields": "dynamic"},
        })

        assert transform.is_batch_aware is True

    def test_creates_tokens_false(self) -> None:
        """Transform does not create new tokens (1-to-1 row mapping)."""
        transform = AzureMultiQueryLLMTransform({
            "deployment_name": "gpt-4o",
            "endpoint": "https://test.openai.azure.com",
            "api_key": "test-key",
            "template": "{{ input_1 }}",
            "case_studies": [{"name": "cs1", "input_fields": ["a"]}],
            "criteria": [{"name": "crit1"}],
            "response_format": "json",
            "output_mapping": {"score": "score"},
            "schema": {"fields": "dynamic"},
        })

        assert transform.creates_tokens is False
