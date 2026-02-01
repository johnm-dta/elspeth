# tests/contracts/transform_contracts/test_passthrough_contract.py
"""Contract tests for PassThrough transform plugin.

Verifies PassThrough honors the TransformProtocol contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from elspeth.plugins.transforms.passthrough import PassThrough

from .test_transform_protocol import (
    TransformContractPropertyTestBase,
    TransformContractTestBase,
)

if TYPE_CHECKING:
    from elspeth.plugins.protocols import TransformProtocol


class TestPassThroughContract(TransformContractPropertyTestBase):
    """Contract tests for PassThrough transform."""

    @pytest.fixture
    def transform(self) -> TransformProtocol:
        """Create a PassThrough instance with dynamic schema."""
        return PassThrough(
            {
                "schema": {"fields": "dynamic"},
            }
        )

    @pytest.fixture
    def valid_input(self) -> dict[str, Any]:
        """Provide a valid input row."""
        return {"id": 1, "name": "test", "value": 42.5}

    # Additional PassThrough-specific contract tests

    def test_passthrough_preserves_all_fields(self, transform: TransformProtocol) -> None:
        """PassThrough MUST preserve all input fields in output."""
        from elspeth.plugins.context import PluginContext

        ctx = PluginContext(run_id="test", config={})
        input_row = {"a": 1, "b": "two", "c": [1, 2, 3], "d": {"nested": True}}

        result = transform.process(input_row, ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row == input_row

    def test_passthrough_does_not_mutate_input(self, transform: TransformProtocol) -> None:
        """PassThrough MUST NOT mutate the input row."""
        from copy import deepcopy

        from elspeth.plugins.context import PluginContext

        ctx = PluginContext(run_id="test", config={})
        input_row = {"id": 1, "nested": {"value": [1, 2, 3]}}
        input_copy = deepcopy(input_row)

        transform.process(input_row, ctx)

        assert input_row == input_copy, "PassThrough mutated input row"

    def test_passthrough_output_is_independent_copy(self, transform: TransformProtocol) -> None:
        """PassThrough output MUST be independent of input (deep copy)."""
        from elspeth.plugins.context import PluginContext

        ctx = PluginContext(run_id="test", config={})
        input_row = {"id": 1, "nested": {"value": [1, 2, 3]}}

        result = transform.process(input_row, ctx)
        assert result.row is not None
        row = result.row
        assert isinstance(row, dict)
        # After isinstance check, mypy should know row is dict, but Protocol returns object
        # so we need to help it
        nested = row["nested"]  # type: ignore[index]
        assert isinstance(nested, dict)

        # Mutate input after processing
        input_row["nested"]["value"].append(4)  # type: ignore[index]

        # Output should be unaffected
        assert nested["value"] == [1, 2, 3]


class TestPassThroughStrictSchemaContract(TransformContractTestBase):
    """Contract tests for PassThrough with strict schema validation."""

    @pytest.fixture
    def transform(self) -> TransformProtocol:
        """Create a PassThrough instance with strict schema and input validation."""
        return PassThrough(
            {
                "schema": {
                    "mode": "strict",
                    "fields": ["id: int", "name: str"],
                },
                "validate_input": True,
            }
        )

    @pytest.fixture
    def valid_input(self) -> dict[str, Any]:
        """Provide a valid input row matching the strict schema."""
        return {"id": 1, "name": "test"}

    def test_strict_passthrough_rejects_wrong_type(self, transform: TransformProtocol) -> None:
        """Strict PassThrough MUST crash on wrong input type (upstream bug!)."""
        from pydantic import ValidationError

        from elspeth.plugins.context import PluginContext

        ctx = PluginContext(run_id="test", config={})
        wrong_type_input = {"id": "not_an_int", "name": "test"}

        # Per Three-Tier Trust Model: wrong types in pipeline data = crash
        with pytest.raises(ValidationError):
            transform.process(wrong_type_input, ctx)


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
                "schema": {"fields": "dynamic"},
            }
        )

    @pytest.fixture
    def ctx(self) -> Any:
        """Create a PluginContext."""
        from elspeth.plugins.context import PluginContext

        return PluginContext(run_id="test", config={})

    @given(
        data=st.dictionaries(
            keys=st.text(min_size=1, max_size=20).filter(lambda s: s.isidentifier()),
            values=(
                st.none()
                | st.booleans()
                | st.integers(min_value=-(2**53 - 1), max_value=2**53 - 1)
                | st.floats(allow_nan=False, allow_infinity=False)
                | st.text(max_size=50)
            ),
            min_size=1,
            max_size=10,
        )
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_passthrough_preserves_arbitrary_dicts(self, transform: PassThrough, ctx: Any, data: dict[str, Any]) -> None:
        """Property: PassThrough preserves any valid JSON-like dict."""
        result = transform.process(data, ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row == data

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
        result1 = transform.process(data, ctx)
        result2 = transform.process(data, ctx)

        assert result1.status == "success"
        assert result2.status == "success"
        assert result1.row is not None
        assert result2.row is not None
        assert result1.row == result2.row
