"""Unit tests for narrow_contract_to_output function."""

import pytest
from structlog.testing import capture_logs

from elspeth.contracts.contract_propagation import narrow_contract_to_output
from elspeth.contracts.schema_contract import SchemaContract
from elspeth.testing import make_field


def test_narrow_contract_field_removal():
    """Test field removal: input has [a, b, c], output has [a, c]."""
    input_contract = SchemaContract(
        mode="FLEXIBLE",
        fields=(
            make_field("a", str, original_name="a", required=True, source="declared"),
            make_field("b", int, original_name="b", required=True, source="declared"),
            make_field("c", float, original_name="c", required=True, source="declared"),
        ),
        locked=True,
    )

    output_row = {"a": "value", "c": 3.14}

    result = narrow_contract_to_output(input_contract, output_row)

    assert len(result.fields) == 2
    assert {f.normalized_name for f in result.fields} == {"a", "c"}
    assert result.mode == "FLEXIBLE"
    assert result.locked is True


def test_narrow_contract_field_addition():
    """Test field addition: input has [a], output has [a, b]."""
    input_contract = SchemaContract(
        mode="FLEXIBLE",
        fields=(make_field("a", str, original_name="a", required=True, source="declared"),),
        locked=True,
    )

    output_row = {"a": "value", "b": 42}

    result = narrow_contract_to_output(input_contract, output_row)

    assert len(result.fields) == 2
    assert {f.normalized_name for f in result.fields} == {"a", "b"}

    # Find the new field
    new_field = next(f for f in result.fields if f.normalized_name == "b")
    assert new_field.python_type is int
    assert new_field.required is False
    assert new_field.source == "inferred"


def test_narrow_contract_field_rename():
    """Test field rename: input has [old], output has [new]."""
    input_contract = SchemaContract(
        mode="FLEXIBLE",
        fields=(make_field("old", str, original_name="old", required=True, source="declared"),),
        locked=True,
    )

    output_row = {"new": "value"}

    result = narrow_contract_to_output(input_contract, output_row)

    assert len(result.fields) == 1
    assert result.fields[0].normalized_name == "new"
    assert result.fields[0].python_type is str
    assert result.fields[0].source == "inferred"


def test_narrow_contract_rename_over_existing_target_preserves_source_metadata():
    """Rename-overwrite keeps source metadata so the output row validates."""
    input_contract = SchemaContract(
        mode="FLEXIBLE",
        fields=(
            make_field("name", str, original_name="Name", required=True, source="declared"),
            make_field("age", int, original_name="Age", required=True, source="declared"),
        ),
        locked=True,
    )

    output_row = {"age": "Alice"}

    result = narrow_contract_to_output(
        input_contract,
        output_row,
        renamed_fields={"name": "age"},
    )

    assert len(result.fields) == 1
    renamed_field = result.get_field("age")
    assert renamed_field.original_name == "Name"
    assert renamed_field.python_type is str
    assert renamed_field.source == "declared"
    assert renamed_field.required is True
    assert result.validate(output_row) == []


def test_narrow_contract_rename_unknown_source_fails_closed():
    """A rename map is engine-owned evidence and must resolve to a source field."""
    input_contract = SchemaContract(
        mode="FLEXIBLE",
        fields=(make_field("name", str, original_name="Name", required=True, source="declared"),),
        locked=True,
    )

    with pytest.raises(ValueError, match="unknown source field"):
        narrow_contract_to_output(
            input_contract,
            {"target": "Alice"},
            renamed_fields={"missing": "target"},
        )


def test_narrow_contract_mixed_operations():
    """Test mixed: input has [a, old, c], output has [a, new, d]."""
    input_contract = SchemaContract(
        mode="FLEXIBLE",
        fields=(
            make_field("a", str, original_name="a", required=True, source="declared"),
            make_field("old", int, original_name="old", required=True, source="declared"),
            make_field("c", float, original_name="c", required=True, source="declared"),
        ),
        locked=True,
    )

    output_row = {"a": "value", "new": 42, "d": True}

    result = narrow_contract_to_output(input_contract, output_row)

    assert len(result.fields) == 3
    assert {f.normalized_name for f in result.fields} == {"a", "new", "d"}

    # Check kept field
    kept_field = next(f for f in result.fields if f.normalized_name == "a")
    assert kept_field.source == "declared"
    assert kept_field.required is True

    # Check inferred fields
    inferred_fields = [f for f in result.fields if f.source == "inferred"]
    assert len(inferred_fields) == 2
    assert {f.normalized_name for f in inferred_fields} == {"new", "d"}


def test_narrow_contract_preserves_dict_and_list_as_object():
    """Dict/list additions are preserved as inferred object fields."""
    input_contract = SchemaContract(
        mode="FLEXIBLE",
        fields=(make_field("a", str, original_name="a", required=True, source="declared"),),
        locked=True,
    )

    output_row = {"a": "value", "dict_field": {"nested": "data"}, "list_field": [1, 2, 3]}

    result = narrow_contract_to_output(input_contract, output_row)

    assert len(result.fields) == 3
    assert {f.normalized_name for f in result.fields} == {"a", "dict_field", "list_field"}

    dict_field = next(f for f in result.fields if f.normalized_name == "dict_field")
    assert dict_field.python_type is object
    assert dict_field.source == "inferred"
    assert dict_field.required is False

    list_field = next(f for f in result.fields if f.normalized_name == "list_field")
    assert list_field.python_type is object
    assert list_field.source == "inferred"
    assert list_field.required is False


def test_narrow_contract_preserves_unsupported_type_as_object():
    """Unsupported types are preserved as object in contract (not silently dropped).

    Regression test for P1 bug: narrow_contract_to_output silently dropped fields
    with unsupported types, causing contract/data divergence and FIXED mode
    access failures.
    """

    class _CustomUnsupported:
        pass

    input_contract = SchemaContract(
        mode="FLEXIBLE",
        fields=(make_field("a", str, original_name="a", required=True, source="declared"),),
        locked=True,
    )

    output_row = {"a": "value", "custom": _CustomUnsupported()}

    result = narrow_contract_to_output(input_contract, output_row)

    assert len(result.fields) == 2
    assert {f.normalized_name for f in result.fields} == {"a", "custom"}
    custom_field = next(f for f in result.fields if f.normalized_name == "custom")
    assert custom_field.python_type is object
    assert custom_field.source == "inferred"
    assert custom_field.required is False


def test_narrow_contract_preserves_decimal_as_object():
    """Decimal values are preserved as object in narrowed contract."""
    from decimal import Decimal

    input_contract = SchemaContract(
        mode="FLEXIBLE",
        fields=(make_field("a", str, original_name="a", required=True, source="declared"),),
        locked=True,
    )

    output_row = {"a": "value", "price": Decimal("12.34")}

    result = narrow_contract_to_output(input_contract, output_row)

    assert len(result.fields) == 2
    assert {f.normalized_name for f in result.fields} == {"a", "price"}
    price_field = next(f for f in result.fields if f.normalized_name == "price")
    assert price_field.python_type is object


def test_narrow_contract_rejects_non_finite_float():
    """Narrowing uses the same non-finite inference policy as propagation."""
    input_contract = SchemaContract(
        mode="FLEXIBLE",
        fields=(make_field("a", str, original_name="a", required=True, source="declared"),),
        locked=True,
    )

    output_row = {"a": "value", "bad": float("nan")}

    with pytest.raises(ValueError, match="non-finite float"):
        narrow_contract_to_output(input_contract, output_row)


def test_narrow_contract_does_not_log_field_level_decisions():
    """Contracts L0 does not emit row/field narrowing decisions to structlog."""
    input_contract = SchemaContract(
        mode="FLEXIBLE",
        fields=(
            make_field("a", str, original_name="a", required=True, source="declared"),
            make_field("old", int, original_name="Original Old", required=True, source="declared"),
        ),
        locked=True,
    )
    output_row = {"a": "value", "new": 42}

    with capture_logs() as logs:
        result = narrow_contract_to_output(
            input_contract,
            output_row,
            renamed_fields={"Original Old": "new"},
        )

    assert logs == []
    assert {field.normalized_name for field in result.fields} == {"a", "new"}


def test_narrow_contract_preserves_mode():
    """Test that contract mode is preserved from input."""
    for mode in ["FIXED", "FLEXIBLE", "OBSERVED"]:
        input_contract = SchemaContract(
            mode=mode,
            fields=(make_field("a", str, original_name="a", required=True, source="declared"),),
            locked=True,
        )

        output_row = {"a": "value", "b": 42}

        result = narrow_contract_to_output(input_contract, output_row)

        assert result.mode == mode


def test_narrow_contract_empty_output():
    """Test edge case: output removes all fields."""
    input_contract = SchemaContract(
        mode="FLEXIBLE",
        fields=(make_field("a", str, original_name="a", required=True, source="declared"),),
        locked=True,
    )

    output_row = {"b": 42}  # 'a' removed, 'b' added

    result = narrow_contract_to_output(input_contract, output_row)

    assert len(result.fields) == 1
    assert result.fields[0].normalized_name == "b"
    assert result.fields[0].source == "inferred"
