"""Tests for ValueTransform transform — behavioral unit tests."""

from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from elspeth.contracts.schema import SchemaConfig
from elspeth.testing import make_contract, make_pipeline_row, make_row
from tests.fixtures.factories import make_source_context
from tests.fixtures.pipeline import build_linear_pipeline

if TYPE_CHECKING:
    from elspeth.contracts.plugin_context import PluginContext

OBSERVED_SCHEMA_CONFIG = SchemaConfig.from_dict({"mode": "observed"})

DYNAMIC_SCHEMA = {"mode": "observed"}


class TestValueTransformBehavior:
    """Core value transformation mechanics."""

    @pytest.fixture
    def ctx(self) -> "PluginContext":
        return make_source_context()

    def test_arithmetic_expression(self, ctx: "PluginContext") -> None:
        from elspeth.plugins.transforms.value_transform import ValueTransform

        transform = ValueTransform(
            {
                "schema": DYNAMIC_SCHEMA,
                "operations": [{"target": "total", "expression": "row['price'] * row['quantity']"}],
            }
        )
        row = make_pipeline_row({"price": 10, "quantity": 2})
        result = transform.process(row, ctx)
        assert result.status == "success"
        assert result.row is not None
        assert result.row["total"] == 20

    def test_string_concatenation(self, ctx: "PluginContext") -> None:
        from elspeth.plugins.transforms.value_transform import ValueTransform

        transform = ValueTransform(
            {
                "schema": DYNAMIC_SCHEMA,
                "operations": [{"target": "line", "expression": "row['line'] + ' World'"}],
            }
        )
        row = make_pipeline_row({"line": "Hello"})
        result = transform.process(row, ctx)
        assert result.status == "success"
        assert result.row is not None
        assert result.row["line"] == "Hello World"

    def test_multiple_operations_sequential(self, ctx: "PluginContext") -> None:
        """Operations see results of prior operations."""
        from elspeth.plugins.transforms.value_transform import ValueTransform

        transform = ValueTransform(
            {
                "schema": DYNAMIC_SCHEMA,
                "operations": [
                    {"target": "subtotal", "expression": "row['price'] * row['quantity']"},
                    {"target": "tax", "expression": "row['subtotal'] * 0.2"},
                    {"target": "total", "expression": "row['subtotal'] + row['tax']"},
                ],
            }
        )
        row = make_pipeline_row({"price": 100, "quantity": 2})
        result = transform.process(row, ctx)
        assert result.status == "success"
        assert result.row is not None
        assert result.row["subtotal"] == 200
        assert result.row["tax"] == 40.0
        assert result.row["total"] == 240.0

    def test_self_reference_overwrite(self, ctx: "PluginContext") -> None:
        from elspeth.plugins.transforms.value_transform import ValueTransform

        transform = ValueTransform(
            {
                "schema": DYNAMIC_SCHEMA,
                "operations": [{"target": "price", "expression": "row['price'] * 1.1"}],
            }
        )
        row = make_pipeline_row({"price": 100})
        result = transform.process(row, ctx)
        assert result.status == "success"
        assert result.row is not None
        assert result.row["price"] == pytest.approx(110.0)

    def test_duplicate_targets_sequential_rewrite(self, ctx: "PluginContext") -> None:
        from elspeth.plugins.transforms.value_transform import ValueTransform

        transform = ValueTransform(
            {
                "schema": DYNAMIC_SCHEMA,
                "operations": [
                    {"target": "x", "expression": "row['x'] + 1"},
                    {"target": "x", "expression": "row['x'] * 2"},
                ],
            }
        )
        row = make_pipeline_row({"x": 5})
        result = transform.process(row, ctx)
        assert result.status == "success"
        assert result.row is not None
        # (5 + 1) * 2 = 12
        assert result.row["x"] == 12

    def test_creates_new_field(self, ctx: "PluginContext") -> None:
        from elspeth.plugins.transforms.value_transform import ValueTransform

        transform = ValueTransform(
            {
                "schema": DYNAMIC_SCHEMA,
                "operations": [{"target": "new_field", "expression": "row['x'] + 100"}],
            }
        )
        row = make_pipeline_row({"x": 5})
        result = transform.process(row, ctx)
        assert result.status == "success"
        assert result.row is not None
        assert result.row["new_field"] == 105
        assert result.row["x"] == 5  # Original preserved

    def test_ternary_expression(self, ctx: "PluginContext") -> None:
        from elspeth.plugins.transforms.value_transform import ValueTransform

        transform = ValueTransform(
            {
                "schema": DYNAMIC_SCHEMA,
                "operations": [
                    {"target": "discount", "expression": "row['price'] * 0.1 if row['price'] > 50 else 0"},
                ],
            }
        )
        row = make_pipeline_row({"price": 100})
        result = transform.process(row, ctx)
        assert result.status == "success"
        assert result.row is not None
        assert result.row["discount"] == 10.0

    def test_missing_field_errors(self, ctx: "PluginContext") -> None:
        from elspeth.plugins.transforms.value_transform import ValueTransform

        transform = ValueTransform(
            {
                "schema": DYNAMIC_SCHEMA,
                "operations": [{"target": "total", "expression": "row['missing'] * 2"}],
            }
        )
        row = make_pipeline_row({"other": 42})
        result = transform.process(row, ctx)
        assert result.status == "error"
        assert result.reason is not None
        assert result.reason.get("reason") == "invalid_input"
        assert "missing" in result.reason.get("message", "").lower()

    def test_type_error_in_expression(self, ctx: "PluginContext") -> None:
        from elspeth.plugins.transforms.value_transform import ValueTransform

        transform = ValueTransform(
            {
                "schema": DYNAMIC_SCHEMA,
                "operations": [{"target": "result", "expression": "row['text'] * row['num']"}],
            }
        )
        # Can't multiply string by string (would need int)
        row = make_pipeline_row({"text": "hello", "num": "world"})
        result = transform.process(row, ctx)
        assert result.status == "error"

    def test_atomic_failure_no_partial_mutation(self, ctx: "PluginContext") -> None:
        """If second operation fails, first operation should not be applied."""
        from elspeth.plugins.transforms.value_transform import ValueTransform

        transform = ValueTransform(
            {
                "schema": DYNAMIC_SCHEMA,
                "operations": [
                    {"target": "first", "expression": "row['x'] + 1"},  # Would succeed
                    {"target": "second", "expression": "row['missing'] * 2"},  # Will fail
                ],
            }
        )
        row = make_pipeline_row({"x": 5})
        result = transform.process(row, ctx)
        assert result.status == "error"
        # Original row should be unchanged (error path gets original row)

    def test_audit_trail(self, ctx: "PluginContext") -> None:
        from elspeth.plugins.transforms.value_transform import ValueTransform

        transform = ValueTransform(
            {
                "schema": DYNAMIC_SCHEMA,
                "operations": [
                    {"target": "total", "expression": "row['price'] * row['quantity']"},
                    {"target": "line", "expression": "row['line'] + ' modified'"},
                ],
            }
        )
        row = make_pipeline_row({"price": 10, "quantity": 2, "line": "Hello"})
        result = transform.process(row, ctx)
        assert result.status == "success"
        assert result.success_reason is not None
        assert result.success_reason["action"] == "transformed"
        assert "total" in result.success_reason["fields_added"]
        assert "line" in result.success_reason["fields_modified"]
        # Plugin-specific audit details in metadata
        assert result.success_reason["metadata"]["operations_applied"] == 2

    def test_row_get_with_none_handling(self, ctx: "PluginContext") -> None:
        """row.get() returning None in expression that handles it."""
        from elspeth.plugins.transforms.value_transform import ValueTransform

        transform = ValueTransform(
            {
                "schema": DYNAMIC_SCHEMA,
                "operations": [
                    {"target": "result", "expression": "row.get('optional') if row.get('optional') is not None else 0"},
                ],
            }
        )
        row = make_pipeline_row({"other": 42})  # 'optional' is missing
        result = transform.process(row, ctx)
        assert result.status == "success"
        assert result.row is not None
        assert result.row["result"] == 0

    def test_fixed_schema_sequential_operations_can_read_added_fields(self, ctx: "PluginContext") -> None:
        from elspeth.plugins.transforms.value_transform import ValueTransform

        transform = ValueTransform(
            {
                "schema": {"mode": "fixed", "fields": ["price: int", "quantity: int"]},
                "operations": [
                    {"target": "subtotal", "expression": "row['price'] * row['quantity']"},
                    {"target": "tax", "expression": "row['subtotal'] * 0.2"},
                ],
            }
        )
        row = make_row(
            {"price": 100, "quantity": 2},
            contract=make_contract(fields={"price": int, "quantity": int}, mode="FIXED"),
        )

        result = transform.process(row, ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["subtotal"] == 200
        assert result.row["tax"] == 40.0
        assert "subtotal" in result.row
        assert "tax" in result.row
        assert result.row.contract.find_field("subtotal") is not None
        assert result.row.contract.find_field("tax") is not None

    def test_output_schema_config_guarantees_configured_targets_for_dag_validation(self) -> None:
        from elspeth.plugins.transforms.value_transform import ValueTransform

        producer = ValueTransform(
            {
                "schema": {"mode": "fixed", "fields": ["price: int", "quantity: int"]},
                "operations": [{"target": "subtotal", "expression": "row['price'] * row['quantity']"}],
            }
        )
        consumer = ValueTransform(
            {
                "schema": {"mode": "fixed", "fields": ["subtotal: float"]},
                "required_input_fields": ["subtotal"],
                "operations": [{"target": "with_tax", "expression": "row['subtotal'] * 1.2"}],
            }
        )

        assert producer._output_schema_config is not None
        assert "subtotal" in producer._output_schema_config.get_effective_guaranteed_fields()
        build_linear_pipeline([{"price": 100, "quantity": 2}], transforms=[producer, consumer])

    def test_unexpected_evaluator_exceptions_propagate(self, ctx: "PluginContext", monkeypatch: pytest.MonkeyPatch) -> None:
        from elspeth.plugins.transforms.value_transform import ValueTransform

        transform = ValueTransform(
            {
                "schema": DYNAMIC_SCHEMA,
                "operations": [{"target": "out", "expression": "row['x'] + 1"}],
            }
        )
        parser = transform._operations[0].get_parser()

        def boom(row: object) -> object:
            raise RuntimeError("sentinel plugin bug")

        monkeypatch.setattr(parser, "evaluate", boom)

        with pytest.raises(RuntimeError, match="sentinel plugin bug"):
            transform.process(make_pipeline_row({"x": 1}), ctx)


class TestValueTransformConfig:
    """Pydantic config validation for ValueTransformConfig."""

    def test_valid_config(self) -> None:
        from elspeth.plugins.transforms.value_transform import ValueTransformConfig

        cfg = ValueTransformConfig(
            operations=[
                {"target": "total", "expression": "row['price'] * row['quantity']"},
            ],
            schema_config=OBSERVED_SCHEMA_CONFIG,
        )
        assert len(cfg.operations) == 1
        assert cfg.operations[0].target == "total"

    def test_rejects_empty_operations(self) -> None:
        from elspeth.plugins.transforms.value_transform import ValueTransformConfig

        with pytest.raises(ValidationError, match="at least one"):
            ValueTransformConfig(operations=[], schema_config=OBSERVED_SCHEMA_CONFIG)

    def test_rejects_empty_target(self) -> None:
        from elspeth.plugins.transforms.value_transform import ValueTransformConfig

        with pytest.raises(ValidationError, match="target"):
            ValueTransformConfig(
                operations=[{"target": "", "expression": "row['x']"}],
                schema_config=OBSERVED_SCHEMA_CONFIG,
            )

    def test_rejects_empty_expression(self) -> None:
        from elspeth.plugins.transforms.value_transform import ValueTransformConfig

        with pytest.raises(ValidationError, match="expression"):
            ValueTransformConfig(
                operations=[{"target": "x", "expression": ""}],
                schema_config=OBSERVED_SCHEMA_CONFIG,
            )

    def test_rejects_invalid_expression_syntax(self) -> None:
        from elspeth.plugins.transforms.value_transform import ValueTransformConfig

        with pytest.raises(ValidationError, match=r"syntax|parse"):
            ValueTransformConfig(
                operations=[{"target": "x", "expression": "row['x'"}],  # Missing ]
                schema_config=OBSERVED_SCHEMA_CONFIG,
            )

    def test_rejects_forbidden_expression_constructs(self) -> None:
        from elspeth.plugins.transforms.value_transform import ValueTransformConfig

        # Lambda is forbidden by ExpressionParser
        with pytest.raises(ValidationError, match=r"Lambda|forbidden"):
            ValueTransformConfig(
                operations=[{"target": "x", "expression": "lambda: 1"}],
                schema_config=OBSERVED_SCHEMA_CONFIG,
            )

    def test_rejects_non_finite_expression_literal(self) -> None:
        from elspeth.plugins.transforms.value_transform import ValueTransformConfig

        with pytest.raises(ValidationError, match="Non-finite float literal"):
            ValueTransformConfig(
                operations=[{"target": "huge", "expression": "1e309"}],
                schema_config=OBSERVED_SCHEMA_CONFIG,
            )

    def test_allows_duplicate_targets(self) -> None:
        from elspeth.plugins.transforms.value_transform import ValueTransformConfig

        cfg = ValueTransformConfig(
            operations=[
                {"target": "x", "expression": "row['x'] + 1"},
                {"target": "x", "expression": "row['x'] * 2"},  # Same target
            ],
            schema_config=OBSERVED_SCHEMA_CONFIG,
        )
        assert len(cfg.operations) == 2

    def test_from_dict_factory(self) -> None:
        from elspeth.plugins.transforms.value_transform import ValueTransformConfig

        cfg = ValueTransformConfig.from_dict(
            {
                "schema": {"mode": "observed"},
                "operations": [{"target": "x", "expression": "row['a'] + row['b']"}],
            }
        )
        assert cfg.operations[0].target == "x"
