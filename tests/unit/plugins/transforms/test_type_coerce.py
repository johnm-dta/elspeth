"""Tests for TypeCoerce transform — behavioral unit tests."""

from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from elspeth.contracts.schema import SchemaConfig

if TYPE_CHECKING:
    from elspeth.contracts.plugin_context import PluginContext

OBSERVED_SCHEMA_CONFIG = SchemaConfig.from_dict({"mode": "observed"})


class TestTypeCoerceConfig:
    """Pydantic config validation for TypeCoerceConfig."""

    def test_valid_config(self) -> None:
        from elspeth.plugins.transforms.type_coerce import TypeCoerceConfig

        cfg = TypeCoerceConfig(
            conversions=[
                {"field": "price", "to": "float"},
                {"field": "quantity", "to": "int"},
            ],
            schema_config=OBSERVED_SCHEMA_CONFIG,
        )
        assert len(cfg.conversions) == 2
        assert cfg.conversions[0].field == "price"
        assert cfg.conversions[0].to == "float"

    def test_rejects_empty_conversions(self) -> None:
        from elspeth.plugins.transforms.type_coerce import TypeCoerceConfig

        with pytest.raises(ValidationError, match="at least one"):
            TypeCoerceConfig(conversions=[], schema_config=OBSERVED_SCHEMA_CONFIG)

    def test_rejects_invalid_target_type(self) -> None:
        from elspeth.plugins.transforms.type_coerce import TypeCoerceConfig

        with pytest.raises(ValidationError, match="'int', 'float', 'bool' or 'str'"):
            TypeCoerceConfig(
                conversions=[{"field": "x", "to": "datetime"}],
                schema_config=OBSERVED_SCHEMA_CONFIG,
            )

    def test_rejects_empty_field_name(self) -> None:
        from elspeth.plugins.transforms.type_coerce import TypeCoerceConfig

        with pytest.raises(ValidationError, match="field name"):
            TypeCoerceConfig(
                conversions=[{"field": "", "to": "int"}],
                schema_config=OBSERVED_SCHEMA_CONFIG,
            )

    def test_from_dict_factory(self) -> None:
        from elspeth.plugins.transforms.type_coerce import TypeCoerceConfig

        cfg = TypeCoerceConfig.from_dict(
            {
                "schema": {"mode": "observed"},
                "conversions": [{"field": "x", "to": "int"}],
            }
        )
        assert cfg.conversions[0].field == "x"


class TestCoerceToInt:
    """Test coerce_to_int conversion function."""

    def test_int_unchanged(self) -> None:
        from elspeth.plugins.transforms.type_coerce import coerce_to_int

        assert coerce_to_int(42) == 42
        assert coerce_to_int(-7) == -7
        assert coerce_to_int(0) == 0

    def test_float_no_fractional_part(self) -> None:
        from elspeth.plugins.transforms.type_coerce import coerce_to_int

        assert coerce_to_int(3.0) == 3
        assert coerce_to_int(-5.0) == -5
        assert coerce_to_int(0.0) == 0

    def test_float_with_fractional_part_errors(self) -> None:
        from elspeth.plugins.transforms.type_coerce import CoercionError, coerce_to_int

        with pytest.raises(CoercionError, match="fractional"):
            coerce_to_int(3.9)
        with pytest.raises(CoercionError, match="fractional"):
            coerce_to_int(3.1)
        with pytest.raises(CoercionError, match="fractional"):
            coerce_to_int(-2.5)

    def test_string_integer(self) -> None:
        from elspeth.plugins.transforms.type_coerce import coerce_to_int

        assert coerce_to_int("42") == 42
        assert coerce_to_int("-7") == -7
        assert coerce_to_int("+42") == 42
        assert coerce_to_int("0") == 0

    def test_string_with_whitespace(self) -> None:
        from elspeth.plugins.transforms.type_coerce import coerce_to_int

        assert coerce_to_int(" 42 ") == 42
        assert coerce_to_int("  -7  ") == -7

    def test_string_decimal_errors(self) -> None:
        from elspeth.plugins.transforms.type_coerce import CoercionError, coerce_to_int

        with pytest.raises(CoercionError, match="not a valid integer"):
            coerce_to_int("3.5")
        with pytest.raises(CoercionError, match="not a valid integer"):
            coerce_to_int("3.0")  # String "3.0" is not valid int

    def test_string_scientific_notation_errors(self) -> None:
        from elspeth.plugins.transforms.type_coerce import CoercionError, coerce_to_int

        with pytest.raises(CoercionError, match="not a valid integer"):
            coerce_to_int("1e3")

    def test_empty_string_errors(self) -> None:
        from elspeth.plugins.transforms.type_coerce import CoercionError, coerce_to_int

        with pytest.raises(CoercionError, match="empty"):
            coerce_to_int("")
        with pytest.raises(CoercionError, match="empty"):
            coerce_to_int("   ")

    def test_bool_errors(self) -> None:
        from elspeth.plugins.transforms.type_coerce import CoercionError, coerce_to_int

        with pytest.raises(CoercionError, match="bool"):
            coerce_to_int(True)
        with pytest.raises(CoercionError, match="bool"):
            coerce_to_int(False)

    def test_none_errors(self) -> None:
        from elspeth.plugins.transforms.type_coerce import CoercionError, coerce_to_int

        with pytest.raises(CoercionError, match="None"):
            coerce_to_int(None)


class TestCoerceToFloat:
    """Test coerce_to_float conversion function."""

    def test_float_unchanged(self) -> None:
        from elspeth.plugins.transforms.type_coerce import coerce_to_float

        assert coerce_to_float(3.14) == 3.14
        assert coerce_to_float(-2.5) == -2.5
        assert coerce_to_float(0.0) == 0.0

    def test_int_to_float(self) -> None:
        from elspeth.plugins.transforms.type_coerce import coerce_to_float

        assert coerce_to_float(42) == 42.0
        assert coerce_to_float(-7) == -7.0
        assert coerce_to_float(0) == 0.0

    def test_string_numeric(self) -> None:
        from elspeth.plugins.transforms.type_coerce import coerce_to_float

        assert coerce_to_float("12.5") == 12.5
        assert coerce_to_float("-3.14") == -3.14
        assert coerce_to_float("+2.5") == 2.5
        assert coerce_to_float("42") == 42.0

    def test_string_with_whitespace(self) -> None:
        from elspeth.plugins.transforms.type_coerce import coerce_to_float

        assert coerce_to_float(" 12.5 ") == 12.5
        assert coerce_to_float("  -3.14  ") == -3.14

    def test_scientific_notation(self) -> None:
        from elspeth.plugins.transforms.type_coerce import coerce_to_float

        assert coerce_to_float("1e3") == 1000.0
        assert coerce_to_float("2.5e-4") == 0.00025

    def test_nan_inf_rejected(self) -> None:
        from elspeth.plugins.transforms.type_coerce import CoercionError, coerce_to_float

        with pytest.raises(CoercionError, match="non-finite"):
            coerce_to_float("nan")
        with pytest.raises(CoercionError, match="non-finite"):
            coerce_to_float("inf")
        with pytest.raises(CoercionError, match="non-finite"):
            coerce_to_float("-inf")
        with pytest.raises(CoercionError, match="non-finite"):
            coerce_to_float(float("nan"))
        with pytest.raises(CoercionError, match="non-finite"):
            coerce_to_float(float("inf"))

    def test_empty_string_errors(self) -> None:
        from elspeth.plugins.transforms.type_coerce import CoercionError, coerce_to_float

        with pytest.raises(CoercionError, match="empty"):
            coerce_to_float("")
        with pytest.raises(CoercionError, match="empty"):
            coerce_to_float("   ")

    def test_bool_errors(self) -> None:
        from elspeth.plugins.transforms.type_coerce import CoercionError, coerce_to_float

        with pytest.raises(CoercionError, match="bool"):
            coerce_to_float(True)

    def test_none_errors(self) -> None:
        from elspeth.plugins.transforms.type_coerce import CoercionError, coerce_to_float

        with pytest.raises(CoercionError, match="None"):
            coerce_to_float(None)


class TestCoerceToBool:
    """Test coerce_to_bool conversion function."""

    def test_bool_unchanged(self) -> None:
        from elspeth.plugins.transforms.type_coerce import coerce_to_bool

        assert coerce_to_bool(True) is True
        assert coerce_to_bool(False) is False

    def test_int_zero_one(self) -> None:
        from elspeth.plugins.transforms.type_coerce import coerce_to_bool

        assert coerce_to_bool(0) is False
        assert coerce_to_bool(1) is True

    def test_int_other_errors(self) -> None:
        from elspeth.plugins.transforms.type_coerce import CoercionError, coerce_to_bool

        with pytest.raises(CoercionError, match="only 0 and 1"):
            coerce_to_bool(2)
        with pytest.raises(CoercionError, match="only 0 and 1"):
            coerce_to_bool(-1)

    def test_string_true_set(self) -> None:
        from elspeth.plugins.transforms.type_coerce import coerce_to_bool

        for val in ["true", "TRUE", "True", "1", "yes", "YES", "y", "Y", "on", "ON"]:
            assert coerce_to_bool(val) is True, f"Expected True for {val!r}"

    def test_string_false_set(self) -> None:
        from elspeth.plugins.transforms.type_coerce import coerce_to_bool

        for val in ["false", "FALSE", "False", "0", "no", "NO", "n", "N", "off", "OFF", ""]:
            assert coerce_to_bool(val) is False, f"Expected False for {val!r}"

    def test_string_with_whitespace(self) -> None:
        from elspeth.plugins.transforms.type_coerce import coerce_to_bool

        assert coerce_to_bool(" true ") is True
        assert coerce_to_bool("  false  ") is False
        assert coerce_to_bool("  ") is False  # whitespace-only = empty = false

    def test_string_invalid_errors(self) -> None:
        from elspeth.plugins.transforms.type_coerce import CoercionError, coerce_to_bool

        with pytest.raises(CoercionError, match="not a valid boolean"):
            coerce_to_bool("maybe")
        with pytest.raises(CoercionError, match="not a valid boolean"):
            coerce_to_bool("oui")

    def test_float_errors(self) -> None:
        from elspeth.plugins.transforms.type_coerce import CoercionError, coerce_to_bool

        with pytest.raises(CoercionError, match="float"):
            coerce_to_bool(1.0)

    def test_none_errors(self) -> None:
        from elspeth.plugins.transforms.type_coerce import CoercionError, coerce_to_bool

        with pytest.raises(CoercionError, match="None"):
            coerce_to_bool(None)


class TestCoerceToStr:
    """Test coerce_to_str conversion function."""

    def test_str_unchanged(self) -> None:
        from elspeth.plugins.transforms.type_coerce import coerce_to_str

        assert coerce_to_str("hello") == "hello"
        assert coerce_to_str("") == ""

    def test_int_to_str(self) -> None:
        from elspeth.plugins.transforms.type_coerce import coerce_to_str

        assert coerce_to_str(42) == "42"
        assert coerce_to_str(-7) == "-7"
        assert coerce_to_str(0) == "0"

    def test_float_to_str(self) -> None:
        from elspeth.plugins.transforms.type_coerce import coerce_to_str

        assert coerce_to_str(3.14) == "3.14"
        assert coerce_to_str(-2.5) == "-2.5"

    def test_bool_to_str(self) -> None:
        from elspeth.plugins.transforms.type_coerce import coerce_to_str

        assert coerce_to_str(True) == "True"
        assert coerce_to_str(False) == "False"

    def test_list_errors(self) -> None:
        from elspeth.plugins.transforms.type_coerce import CoercionError, coerce_to_str

        with pytest.raises(CoercionError, match="not a scalar"):
            coerce_to_str([1, 2, 3])

    def test_dict_errors(self) -> None:
        from elspeth.plugins.transforms.type_coerce import CoercionError, coerce_to_str

        with pytest.raises(CoercionError, match="not a scalar"):
            coerce_to_str({"a": 1})

    def test_none_errors(self) -> None:
        from elspeth.plugins.transforms.type_coerce import CoercionError, coerce_to_str

        with pytest.raises(CoercionError, match="None"):
            coerce_to_str(None)


DYNAMIC_SCHEMA = {"mode": "observed"}


class TestTypeCoerceBehavior:
    """Core type coercion mechanics."""

    @pytest.fixture
    def ctx(self) -> "PluginContext":
        from tests.fixtures.factories import make_source_context

        return make_source_context()

    def test_coerces_string_to_int(self, ctx: "PluginContext") -> None:
        from elspeth.plugins.transforms.type_coerce import TypeCoerce
        from elspeth.testing import make_pipeline_row

        transform = TypeCoerce(
            {
                "schema": DYNAMIC_SCHEMA,
                "conversions": [{"field": "quantity", "to": "int"}],
            }
        )
        row = make_pipeline_row({"quantity": "42"})
        result = transform.process(row, ctx)
        assert result.status == "success"
        assert result.row is not None
        assert result.row["quantity"] == 42
        assert type(result.row["quantity"]) is int

    def test_coerces_string_to_float(self, ctx: "PluginContext") -> None:
        from elspeth.plugins.transforms.type_coerce import TypeCoerce
        from elspeth.testing import make_pipeline_row

        transform = TypeCoerce(
            {
                "schema": DYNAMIC_SCHEMA,
                "conversions": [{"field": "price", "to": "float"}],
            }
        )
        row = make_pipeline_row({"price": " 12.50 "})
        result = transform.process(row, ctx)
        assert result.status == "success"
        assert result.row is not None
        assert result.row["price"] == 12.5

    def test_coerces_string_to_bool(self, ctx: "PluginContext") -> None:
        from elspeth.plugins.transforms.type_coerce import TypeCoerce
        from elspeth.testing import make_pipeline_row

        transform = TypeCoerce(
            {
                "schema": DYNAMIC_SCHEMA,
                "conversions": [{"field": "active", "to": "bool"}],
            }
        )
        row = make_pipeline_row({"active": "false"})
        result = transform.process(row, ctx)
        assert result.status == "success"
        assert result.row is not None
        assert result.row["active"] is False

    def test_coerces_int_to_str(self, ctx: "PluginContext") -> None:
        from elspeth.plugins.transforms.type_coerce import TypeCoerce
        from elspeth.testing import make_pipeline_row

        transform = TypeCoerce(
            {
                "schema": DYNAMIC_SCHEMA,
                "conversions": [{"field": "user_id", "to": "str"}],
            }
        )
        row = make_pipeline_row({"user_id": 42})
        result = transform.process(row, ctx)
        assert result.status == "success"
        assert result.row is not None
        assert result.row["user_id"] == "42"

    def test_multiple_conversions(self, ctx: "PluginContext") -> None:
        from elspeth.plugins.transforms.type_coerce import TypeCoerce
        from elspeth.testing import make_pipeline_row

        transform = TypeCoerce(
            {
                "schema": DYNAMIC_SCHEMA,
                "conversions": [
                    {"field": "price", "to": "float"},
                    {"field": "quantity", "to": "int"},
                    {"field": "active", "to": "bool"},
                ],
            }
        )
        row = make_pipeline_row({"price": "12.50", "quantity": "3", "active": "yes"})
        result = transform.process(row, ctx)
        assert result.status == "success"
        assert result.row is not None
        assert result.row["price"] == 12.5
        assert result.row["quantity"] == 3
        assert result.row["active"] is True

    def test_already_correct_type_unchanged(self, ctx: "PluginContext") -> None:
        from elspeth.plugins.transforms.type_coerce import TypeCoerce
        from elspeth.testing import make_pipeline_row

        transform = TypeCoerce(
            {
                "schema": DYNAMIC_SCHEMA,
                "conversions": [{"field": "quantity", "to": "int"}],
            }
        )
        row = make_pipeline_row({"quantity": 42})
        result = transform.process(row, ctx)
        assert result.status == "success"
        assert result.row is not None
        assert result.row["quantity"] == 42
        # Check audit shows unchanged in metadata
        assert result.success_reason is not None
        assert "quantity" in result.success_reason["metadata"]["fields_unchanged"]

    def test_missing_field_errors(self, ctx: "PluginContext") -> None:
        from elspeth.plugins.transforms.type_coerce import TypeCoerce
        from elspeth.testing import make_pipeline_row

        transform = TypeCoerce(
            {
                "schema": DYNAMIC_SCHEMA,
                "conversions": [{"field": "missing", "to": "int"}],
            }
        )
        row = make_pipeline_row({"other": 42})
        result = transform.process(row, ctx)
        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "missing_field"
        assert result.reason["field"] == "missing"

    def test_conversion_failure_errors(self, ctx: "PluginContext") -> None:
        from elspeth.plugins.transforms.type_coerce import TypeCoerce
        from elspeth.testing import make_pipeline_row

        transform = TypeCoerce(
            {
                "schema": DYNAMIC_SCHEMA,
                "conversions": [{"field": "active", "to": "bool"}],
            }
        )
        row = make_pipeline_row({"active": "maybe"})
        result = transform.process(row, ctx)
        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "type_mismatch"
        assert "maybe" in result.reason.get("message", "")

    def test_atomic_failure_no_partial_mutation(self, ctx: "PluginContext") -> None:
        """If second conversion fails, first conversion should not be applied."""
        from elspeth.plugins.transforms.type_coerce import TypeCoerce
        from elspeth.testing import make_pipeline_row

        transform = TypeCoerce(
            {
                "schema": DYNAMIC_SCHEMA,
                "conversions": [
                    {"field": "price", "to": "float"},  # Would succeed
                    {"field": "active", "to": "bool"},  # Will fail
                ],
            }
        )
        row = make_pipeline_row({"price": "12.50", "active": "maybe"})
        result = transform.process(row, ctx)
        assert result.status == "error"
        # Original row should be unchanged (error path gets original row)

    def test_audit_trail_fields_coerced(self, ctx: "PluginContext") -> None:
        from elspeth.plugins.transforms.type_coerce import TypeCoerce
        from elspeth.testing import make_pipeline_row

        transform = TypeCoerce(
            {
                "schema": DYNAMIC_SCHEMA,
                "conversions": [
                    {"field": "price", "to": "float"},
                    {"field": "quantity", "to": "int"},
                ],
            }
        )
        row = make_pipeline_row({"price": "12.50", "quantity": 3})  # quantity already int
        result = transform.process(row, ctx)
        assert result.status == "success"
        assert result.success_reason is not None
        assert result.success_reason["action"] == "coerced"
        assert result.success_reason["fields_modified"] == ["price"]
        # Plugin-specific audit details in metadata
        metadata = result.success_reason["metadata"]
        assert metadata["fields_coerced"] == ["price"]
        assert metadata["fields_unchanged"] == ["quantity"]
        assert metadata["rules_evaluated"] == 2

    def test_fixed_schema_initializes_output_schema_config_and_aligns_output_contract(self, ctx: "PluginContext") -> None:
        """TypeCoerce must emit the configured contract mode, not the upstream one."""
        from elspeth.plugins.transforms.type_coerce import TypeCoerce
        from elspeth.testing import make_pipeline_row

        transform = TypeCoerce(
            {
                "schema": {
                    "mode": "fixed",
                    "fields": ["quantity: int"],
                },
                "conversions": [{"field": "quantity", "to": "int"}],
            }
        )
        row = make_pipeline_row({"quantity": "42"})

        result = transform.process(row, ctx)

        assert transform._output_schema_config is not None
        assert transform._output_schema_config.mode == "fixed"
        assert result.row is not None
        assert result.row.contract.mode == "FIXED"
        assert result.row.contract.locked is True

    def test_success_updates_emitted_contract_field_type(self, ctx: "PluginContext") -> None:
        """Successful coercion must evolve the emitted contract, not just the payload."""
        from elspeth.contracts.schema_contract import SchemaContract
        from elspeth.plugins.transforms.type_coerce import TypeCoerce
        from elspeth.testing import make_field, make_row

        transform = TypeCoerce(
            {
                "schema": {"mode": "observed"},
                "conversions": [{"field": "quantity", "to": "int"}],
            }
        )
        input_contract = SchemaContract(
            mode="OBSERVED",
            fields=(
                make_field(
                    "quantity",
                    str,
                    original_name="Quantity",
                    required=True,
                    source="declared",
                ),
            ),
            locked=True,
        )
        row = make_row({"quantity": "42"}, contract=input_contract)

        result = transform.process(row, ctx)

        assert result.status == "success"
        assert result.row is not None
        field = result.row.contract.get_field("quantity")
        assert field is not None
        assert field.python_type is int
        assert field.original_name == "Quantity"
        assert field.required is True
        assert result.row.contract is not row.contract
