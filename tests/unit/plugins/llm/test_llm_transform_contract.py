# tests/plugins/llm/test_llm_transform_contract.py
"""Tests for LLM transform contract integration."""

from typing import Any
from unittest.mock import MagicMock

import pytest

from elspeth.contracts.schema_contract import PipelineRow, SchemaContract
from elspeth.plugins.clients.llm import LLMResponse
from elspeth.plugins.llm.base import BaseLLMTransform
from elspeth.testing import make_field, make_row


class MockLLMTransform(BaseLLMTransform):
    """Test LLM transform that returns a mock LLM client.

    Overrides _get_llm_client to return a controllable mock client.
    The mock_response attribute controls what chat_completion() returns.
    """

    name = "mock_llm"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.mock_response_content = "mocked response"
        self._mock_client: MagicMock | None = None

    def _get_llm_client(self, ctx: Any) -> MagicMock:
        """Return a mock LLM client with chat_completion() configured."""
        if self._mock_client is None:
            self._mock_client = MagicMock()
            self._mock_client.chat_completion.return_value = LLMResponse(
                content=self.mock_response_content,
                model="mock-model",
                usage={"prompt_tokens": 10, "completion_tokens": 20},
                latency_ms=100.0,
            )
        return self._mock_client


class TestLLMTransformContract:
    """Test LLM transform with contract-aware templates."""

    @pytest.fixture
    def contract(self) -> SchemaContract:
        """Contract with original name mappings."""
        return SchemaContract(
            mode="FLEXIBLE",
            fields=(
                make_field("product_name", str, original_name="Product Name", required=True, source="declared"),
                make_field("description", str, original_name="DESCRIPTION", required=True, source="declared"),
            ),
            locked=True,
        )

    @pytest.fixture
    def data(self) -> dict[str, object]:
        """Sample row data."""
        return {
            "product_name": "Widget",
            "description": "A useful widget",
        }

    @pytest.fixture
    def mock_context(self) -> MagicMock:
        """Mock plugin context with required attributes."""
        ctx = MagicMock()
        ctx.run_id = "test-run"
        ctx.state_id = "test-state"
        ctx.landscape = MagicMock()
        # Explicitly set contract to None to test dict-without-contract cases
        # (MagicMock would return another MagicMock for unset attributes)
        ctx.contract = None
        return ctx

    def test_process_with_pipeline_row(
        self,
        data: dict[str, object],
        contract: SchemaContract,
        mock_context: MagicMock,
    ) -> None:
        """Process accepts PipelineRow and uses contract for template."""
        # Template uses ORIGINAL name "Product Name" which should resolve
        transform = MockLLMTransform(
            {
                "model": "test-model",
                "template": 'Analyze: {{ row["Product Name"] }}',
                "schema": {"mode": "observed"},
                "required_input_fields": ["product_name"],
            }
        )

        pipeline_row = make_row(data, contract=contract)
        result = transform.process(pipeline_row, mock_context)

        assert result.status == "success"
        # Verify the template actually rendered (contract resolved the name)
        assert transform._mock_client is not None
        call_args = transform._mock_client.chat_completion.call_args
        messages = call_args.kwargs.get("messages") or call_args[1].get("messages")
        user_message = next(m for m in messages if m["role"] == "user")
        assert "Analyze: Widget" in user_message["content"]

    def test_process_with_minimal_contract(
        self,
        data: dict[str, object],
        mock_context: MagicMock,
    ) -> None:
        """Process with minimal FLEXIBLE contract allows any fields."""
        # Create minimal contract (FLEXIBLE mode, no declared fields)
        minimal_contract = SchemaContract(
            fields=(),
            mode="FLEXIBLE",
            locked=True,
        )
        pipeline_row = make_row(data, contract=minimal_contract)

        # Template uses NORMALIZED name
        transform = MockLLMTransform(
            {
                "model": "test-model",
                "template": "Analyze: {{ row.product_name }}",
                "schema": {"mode": "observed"},
                "required_input_fields": ["product_name"],
            }
        )

        result = transform.process(pipeline_row, mock_context)

        assert result.status == "success"

    def test_result_has_contract_when_input_has_contract(
        self,
        data: dict[str, object],
        contract: SchemaContract,
        mock_context: MagicMock,
    ) -> None:
        """TransformResult includes contract when input is PipelineRow."""
        transform = MockLLMTransform(
            {
                "model": "test-model",
                "template": "{{ row.product_name }}",
                "schema": {"mode": "observed"},
                "required_input_fields": ["product_name"],
                "response_field": "llm_result",
            }
        )

        pipeline_row = make_row(data, contract=contract)
        result = transform.process(pipeline_row, mock_context)

        assert result.status == "success"
        # Result should have a contract (propagated with new fields, inside PipelineRow)
        assert isinstance(result.row, PipelineRow)
        # Contract should include the new llm_result field
        assert result.row.contract.get_field("llm_result") is not None

    def test_result_has_contract_even_with_minimal_input_contract(
        self,
        data: dict[str, object],
        mock_context: MagicMock,
    ) -> None:
        """TransformResult has contract even when input has minimal FLEXIBLE contract."""
        # Create minimal contract
        minimal_contract = SchemaContract(
            fields=(),
            mode="FLEXIBLE",
            locked=True,
        )
        pipeline_row = make_row(data, contract=minimal_contract)

        transform = MockLLMTransform(
            {
                "model": "test-model",
                "template": "{{ row.product_name }}",
                "schema": {"mode": "observed"},
                "required_input_fields": ["product_name"],
            }
        )

        result = transform.process(pipeline_row, mock_context)

        assert result.status == "success"
        assert isinstance(result.row, PipelineRow)  # Contract is propagated inside PipelineRow

    def test_template_error_with_original_name_minimal_contract(
        self,
        data: dict[str, object],
        mock_context: MagicMock,
    ) -> None:
        """Template using original name fails with minimal contract (no name mappings)."""
        # Create minimal FLEXIBLE contract (no field mappings for original names)
        minimal_contract = SchemaContract(
            fields=(),
            mode="FLEXIBLE",
            locked=True,
        )
        pipeline_row = make_row(data, contract=minimal_contract)

        # Template uses original name but contract has no mapping for it
        transform = MockLLMTransform(
            {
                "model": "test-model",
                "template": 'Analyze: {{ row["Product Name"] }}',
                "schema": {"mode": "observed"},
                "required_input_fields": [],  # Opt-out of field checking
            }
        )

        result = transform.process(pipeline_row, mock_context)

        # Should fail because "Product Name" is not in data and minimal contract has no mappings
        assert result.status == "error"
        assert result.reason is not None
        assert "template" in str(result.reason.get("reason", "")).lower()

    def test_contract_propagation_adds_new_fields(
        self,
        data: dict[str, object],
        contract: SchemaContract,
        mock_context: MagicMock,
    ) -> None:
        """Contract propagation infers types for new fields added by transform."""
        transform = MockLLMTransform(
            {
                "model": "test-model",
                "template": "{{ row.product_name }}",
                "schema": {"mode": "observed"},
                "required_input_fields": ["product_name"],
                "response_field": "analysis",
            }
        )

        pipeline_row = make_row(data, contract=contract)
        result = transform.process(pipeline_row, mock_context)

        assert result.status == "success"
        assert isinstance(result.row, PipelineRow)

        # New field should be in contract with inferred type
        analysis_field = result.row.contract.get_field("analysis")
        assert analysis_field is not None
        assert analysis_field.python_type is str  # Inferred from string response
        assert analysis_field.source == "inferred"

        # Original fields should still be present
        assert result.row.contract.get_field("product_name") is not None

    def test_fixed_contract_includes_usage_metadata_field(
        self,
        mock_context: MagicMock,
    ) -> None:
        """LLM output contract includes guaranteed _usage even in FIXED mode."""
        fixed_contract = SchemaContract(
            mode="FIXED",
            fields=(make_field("text", str, original_name="text", required=True, source="declared"),),
            locked=True,
        )
        pipeline_row = make_row({"text": "hello"}, contract=fixed_contract)

        transform = MockLLMTransform(
            {
                "model": "test-model",
                "template": "{{ row.text }}",
                "schema": {"mode": "fixed", "fields": ["text: str"]},
                "required_input_fields": ["text"],
            }
        )

        result = transform.process(pipeline_row, mock_context)

        assert result.status == "success"
        assert isinstance(result.row, PipelineRow)
        usage_field = result.row.contract.get_field("llm_response_usage")
        assert usage_field is not None
        assert usage_field.python_type is object
        assert result.row["llm_response_usage"] == {
            "prompt_tokens": 10,
            "completion_tokens": 20,
        }
