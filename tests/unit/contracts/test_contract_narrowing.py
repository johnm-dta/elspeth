"""Unit tests for narrow_contract_to_output function."""

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


def test_narrow_contract_skips_unsupported_non_dict_list_type():
    """Unsupported non-dict/list values preserve existing skip behavior."""

    class _CustomUnsupported:
        pass

    input_contract = SchemaContract(
        mode="FLEXIBLE",
        fields=(make_field("a", str, original_name="a", required=True, source="declared"),),
        locked=True,
    )

    output_row = {"a": "value", "custom": _CustomUnsupported()}

    result = narrow_contract_to_output(input_contract, output_row)

    assert len(result.fields) == 1
    assert result.fields[0].normalized_name == "a"


def test_narrow_contract_skips_non_finite_float():
    """Non-finite floats remain excluded in narrowing inference."""
    input_contract = SchemaContract(
        mode="FLEXIBLE",
        fields=(make_field("a", str, original_name="a", required=True, source="declared"),),
        locked=True,
    )

    output_row = {"a": "value", "bad": float("nan")}

    result = narrow_contract_to_output(input_contract, output_row)

    assert len(result.fields) == 1
    assert result.fields[0].normalized_name == "a"


def test_narrow_contract_preserves_mode():
    """Test that contract mode is preserved from input."""
    for mode in ["FIXED", "FLEXIBLE", "OBSERVED"]:
        input_contract = SchemaContract(
            mode=mode,  # type: ignore[arg-type]
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
