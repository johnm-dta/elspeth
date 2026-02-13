# tests/unit/contracts/transform_contracts/test_truncate_contract.py
"""Contract tests for Truncate transform.

Verifies Truncate honors the TransformProtocol contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from elspeth.plugins.transforms.truncate import Truncate
from elspeth.testing import make_field, make_pipeline_row
from tests.fixtures.factories import make_context

from .test_transform_protocol import (
    TransformContractPropertyTestBase,
    TransformErrorContractTestBase,
)

if TYPE_CHECKING:
    from elspeth.plugins.protocols import TransformProtocol


class TestTruncateContract(TransformContractPropertyTestBase):
    """Contract tests for Truncate plugin."""

    @pytest.fixture
    def transform(self) -> TransformProtocol:
        """Return a configured transform instance."""
        return Truncate(
            {
                "fields": {"title": 20, "description": 50},
                "schema": {"mode": "observed"},
            }
        )

    @pytest.fixture
    def valid_input(self) -> dict[str, Any]:
        """Return input that should process successfully."""
        return {"title": "Short title", "description": "Short description", "id": 1}


class TestTruncateWithSuffixContract(TransformContractPropertyTestBase):
    """Contract tests for Truncate plugin with suffix."""

    @pytest.fixture
    def transform(self) -> TransformProtocol:
        """Return a configured transform instance with suffix."""
        return Truncate(
            {
                "fields": {"title": 20},
                "suffix": "...",
                "schema": {"mode": "observed"},
            }
        )

    @pytest.fixture
    def valid_input(self) -> dict[str, Any]:
        """Return input that should process successfully."""
        return {"title": "Short", "id": 1}


class TestTruncateStrictContract(TransformErrorContractTestBase):
    """Contract tests for Truncate error handling in strict mode."""

    @pytest.fixture
    def transform(self) -> TransformProtocol:
        """Return a configured transform instance in strict mode."""
        return Truncate(
            {
                "fields": {"required_field": 100},
                "strict": True,
                "schema": {"mode": "observed"},
            }
        )

    @pytest.fixture
    def valid_input(self) -> dict[str, Any]:
        """Return input that should process successfully (has required field)."""
        return {"required_field": "value", "id": 1}

    @pytest.fixture
    def error_input(self) -> dict[str, Any]:
        """Return input that triggers error (missing required field in strict mode)."""
        return {"other_field": "value", "id": 2}

    def test_strict_missing_field_returns_error(
        self,
        transform: TransformProtocol,
        error_input: dict[str, Any],
        ctx: Any,
    ) -> None:
        """Contract: Strict mode MUST return error with missing_field reason."""
        ctx = make_context(run_id="test")
        pipeline_row = make_pipeline_row(error_input)
        result = transform.process(pipeline_row, ctx)
        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "missing_field"

    def test_strict_wrong_type_returns_error(
        self,
        transform: TransformProtocol,
        ctx: Any,
    ) -> None:
        """Contract: Strict mode MUST return error when field is not a string."""
        ctx = make_context(run_id="test")
        pipeline_row = make_pipeline_row({"required_field": 42, "id": 3})
        result = transform.process(pipeline_row, ctx)
        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "type_mismatch"
        assert result.reason["field"] == "required_field"
        assert result.reason["expected"] == "str"
        assert result.reason["actual"] == "int"

    def test_strict_none_value_returns_error(
        self,
        transform: TransformProtocol,
        ctx: Any,
    ) -> None:
        """Contract: Strict mode MUST return error when field value is None."""
        ctx = make_context(run_id="test")
        pipeline_row = make_pipeline_row({"required_field": None, "id": 4})
        result = transform.process(pipeline_row, ctx)
        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "type_mismatch"
        assert result.reason["actual"] == "NoneType"


class TestTruncateNonStringContract(TransformErrorContractTestBase):
    """Contract tests: configured non-string fields must return an explicit error."""

    @pytest.fixture
    def transform(self) -> TransformProtocol:
        """Return a configured transform in lenient (non-strict) mode."""
        return Truncate(
            {
                "fields": {"value": 10},
                "strict": False,
                "schema": {"mode": "observed"},
            }
        )

    @pytest.fixture
    def valid_input(self) -> dict[str, Any]:
        """Return input that should process successfully."""
        return {"value": "hello", "id": 1}

    @pytest.fixture
    def error_input(self) -> dict[str, Any]:
        """Return input with non-string field that must return an error."""
        return {"value": 42, "id": 1}

    def test_non_string_returns_type_mismatch_error(
        self,
        transform: TransformProtocol,
        error_input: dict[str, Any],
        ctx: Any,
    ) -> None:
        """Contract: Non-string configured field MUST return type_mismatch error."""
        ctx = make_context(run_id="test")
        pipeline_row = make_pipeline_row(error_input)
        result = transform.process(pipeline_row, ctx)
        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "type_mismatch"
        assert result.reason["field"] == "value"
        assert result.reason["expected"] == "str"
        assert result.reason["actual"] == "int"

    def test_configured_original_field_name_is_resolved(
        self,
        transform: TransformProtocol,
        ctx: Any,
    ) -> None:
        """Configured original names should resolve via PipelineRow contract."""
        from elspeth.contracts.schema_contract import PipelineRow, SchemaContract

        mapped_contract = SchemaContract(
            mode="OBSERVED",
            fields=(
                make_field("value", str, original_name="Value Text", required=False, source="inferred"),
                make_field("id", int, original_name="ID", required=False, source="inferred"),
            ),
            locked=True,
        )
        mapped_row = PipelineRow({"value": "abcdefghijk", "id": 1}, mapped_contract)

        # Use a transform configured with the original field name
        original_name_transform = Truncate(
            {
                "fields": {"Value Text": 5},
                "strict": False,
                "schema": {"mode": "observed"},
            }
        )

        result = original_name_transform.process(mapped_row, ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["value"] == "abcde"
