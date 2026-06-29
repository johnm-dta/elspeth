"""Regression tests for public elspeth.testing factories."""

from __future__ import annotations

from elspeth.testing import make_success_multi


def test_make_success_multi_dict_rows_builds_union_contract() -> None:
    """Dict-row shorthand builds one shared contract from the union of row keys."""
    result = make_success_multi([{"a": 1}, {"b": "x"}])

    assert result.is_multi_row
    assert result.rows is not None
    assert len(result.rows) == 2

    contract = result.rows[0].contract
    assert result.rows[1].contract is contract
    assert {field.normalized_name: field.python_type for field in contract.fields} == {
        "a": object,
        "b": object,
    }
