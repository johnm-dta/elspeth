"""Integration test: type_coerce + value_transform in a pipeline.

Tests the recommended pattern: normalize types first, then compute values.
"""

import pytest

from elspeth.testing import make_pipeline_row
from tests.fixtures.factories import make_context


class TestTypeCoerceValueTransformPipeline:
    """Test type_coerce followed by value_transform."""

    @pytest.fixture
    def ctx(self):
        return make_context()

    def test_typical_pipeline_pattern(self, ctx) -> None:
        """Normalize types, then compute derived values."""
        from elspeth.plugins.transforms.type_coerce import TypeCoerce
        from elspeth.plugins.transforms.value_transform import ValueTransform

        # Step 1: Normalize types
        type_coerce = TypeCoerce(
            {
                "schema": {"mode": "observed"},
                "conversions": [
                    {"field": "price", "to": "float"},
                    {"field": "quantity", "to": "int"},
                ],
            }
        )

        # Step 2: Compute derived values
        value_transform = ValueTransform(
            {
                "schema": {"mode": "observed"},
                "operations": [
                    {"target": "subtotal", "expression": "row['price'] * row['quantity']"},
                    {"target": "tax", "expression": "row['subtotal'] * 0.2"},
                    {"target": "total", "expression": "row['subtotal'] + row['tax']"},
                ],
            }
        )

        # Input with string types (typical from CSV/API)
        row = make_pipeline_row(
            {
                "price": " 12.50 ",
                "quantity": "3",
                "description": "Widget",
            }
        )

        # Apply type coercion
        result1 = type_coerce.process(row, ctx)
        assert result1.status == "success"
        assert result1.row is not None
        assert result1.row["price"] == 12.5
        assert result1.row["quantity"] == 3

        # Apply value transform
        result2 = value_transform.process(result1.row, ctx)
        assert result2.status == "success"
        assert result2.row is not None
        assert result2.row["subtotal"] == 37.5
        assert result2.row["tax"] == 7.5
        assert result2.row["total"] == 45.0
        # Original fields preserved
        assert result2.row["description"] == "Widget"

    def test_type_error_without_coercion(self, ctx) -> None:
        """Show what happens if you skip type_coerce with string data."""
        from elspeth.plugins.transforms.value_transform import ValueTransform

        value_transform = ValueTransform(
            {
                "schema": {"mode": "observed"},
                "operations": [
                    {"target": "total", "expression": "row['price'] * row['quantity']"},
                ],
            }
        )

        # String types will fail multiplication
        row = make_pipeline_row({"price": "12.50", "quantity": "3"})
        result = value_transform.process(row, ctx)
        assert result.status == "error"
        # This demonstrates why type_coerce should come first
