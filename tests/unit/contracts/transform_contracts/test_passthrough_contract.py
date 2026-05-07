# tests/unit/contracts/transform_contracts/test_passthrough_contract.py
"""Contract tests for PassThrough transform plugin.

Verifies PassThrough honors the TransformProtocol contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from elspeth.plugins.transforms.passthrough import PassThrough
from elspeth.testing import make_pipeline_row
from tests.fixtures.factories import make_context

from .test_transform_protocol import (
    TransformContractPropertyTestBase,
    TransformContractTestBase,
)

if TYPE_CHECKING:
    from elspeth.contracts import TransformProtocol


class TestPassThroughContract(TransformContractPropertyTestBase):
    """Contract tests for PassThrough transform."""

    @pytest.fixture
    def transform(self) -> TransformProtocol:
        """Create a PassThrough instance with dynamic schema."""
        return PassThrough(
            {
                "schema": {"mode": "observed"},
            }
        )

    @pytest.fixture
    def valid_input(self) -> dict[str, Any]:
        """Provide a valid input row."""
        return {"id": 1, "name": "test", "value": 42.5}

    # Additional PassThrough-specific contract tests

    def test_passthrough_preserves_all_fields(self, transform: TransformProtocol) -> None:
        """PassThrough MUST preserve all input fields in output."""
        ctx = make_context(run_id="test")
        input_row = {"a": 1, "b": "two", "c": [1, 2, 3], "d": {"nested": True}}
        pipeline_row = make_pipeline_row(input_row)

        result = transform.process(pipeline_row, ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row.to_dict() == input_row

    def test_passthrough_does_not_mutate_input(self, transform: TransformProtocol) -> None:
        """PassThrough MUST NOT mutate the input row."""
        from copy import deepcopy

        ctx = make_context(run_id="test")
        input_row = {"id": 1, "nested": {"value": [1, 2, 3]}}
        input_copy = deepcopy(input_row)
        pipeline_row = make_pipeline_row(input_row)

        transform.process(pipeline_row, ctx)

        assert input_row == input_copy, "PassThrough mutated input row"

    def test_passthrough_output_is_independent_copy(self, transform: TransformProtocol) -> None:
        """PassThrough output MUST be independent of input (deep copy)."""
        ctx = make_context(run_id="test")
        input_row = {"id": 1, "nested": {"value": [1, 2, 3]}}
        pipeline_row = make_pipeline_row(input_row)

        result = transform.process(pipeline_row, ctx)
        assert result.row is not None
        row_dict = result.row.to_dict()
        nested = row_dict["nested"]
        assert isinstance(nested, dict)

        # Mutate input after processing
        input_row["nested"]["value"].append(4)  # type: ignore[index]

        # Output should be unaffected
        assert nested["value"] == [1, 2, 3]


class TestPassThroughStrictSchemaContract(TransformContractTestBase):
    """Contract tests for PassThrough with strict schema validation."""

    @pytest.fixture
    def transform(self) -> TransformProtocol:
        """Create a PassThrough instance with strict schema."""
        return PassThrough(
            {
                "schema": {
                    "mode": "fixed",
                    "fields": ["id: int", "name: str"],
                },
            }
        )

    @pytest.fixture
    def valid_input(self) -> dict[str, Any]:
        """Provide a valid input row matching the strict schema."""
        return {"id": 1, "name": "test"}

    @pytest.mark.parametrize(
        ("invalid_input", "message"),
        [
            ({"id": "not-an-int", "name": "test"}, "id"),
            ({"id": 1}, "name"),
        ],
    )
    def test_strict_schema_rejects_invalid_rows_at_input_boundary(
        self,
        transform: TransformProtocol,
        invalid_input: dict[str, Any],
        message: str,
    ) -> None:
        with pytest.raises(ValidationError, match=message):
            transform.input_schema.model_validate(invalid_input)


class TestPassThroughPropertyBased:
    """Property-based tests for PassThrough transform."""

    # RFC 8785 safe integer bounds for JSON compatibility
    _MAX_SAFE_INT = 2**53 - 1
    _MIN_SAFE_INT = -(2**53 - 1)

    @pytest.fixture
    def transform(self) -> PassThrough:
        """Create a PassThrough instance with dynamic schema."""
        return PassThrough(
            {
                "schema": {"mode": "observed"},
            }
        )

    @pytest.fixture
    def ctx(self) -> Any:
        """Create a PluginContext."""
        return make_context(run_id="test")

    @given(
        data=st.dictionaries(
            keys=st.text(min_size=1, max_size=10).filter(lambda s: s.isidentifier()),
            values=st.integers(min_value=-(2**53 - 1), max_value=2**53 - 1),
            min_size=1,
            max_size=5,
        )
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_passthrough_is_deterministic(self, transform: PassThrough, ctx: Any, data: dict[str, Any]) -> None:
        """Property: PassThrough produces same output for same input."""
        pipeline_row = make_pipeline_row(data)
        result1 = transform.process(pipeline_row, ctx)
        result2 = transform.process(pipeline_row, ctx)

        assert result1.status == "success"
        assert result2.status == "success"
        assert result1.row is not None
        assert result2.row is not None
        assert result1.row.to_dict() == result2.row.to_dict()
