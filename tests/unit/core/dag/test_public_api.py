"""Tests for the DAG package facade."""

from __future__ import annotations

import pytest


def test_dag_facade_does_not_export_coalesce_merge_internals() -> None:
    import elspeth.core.dag as dag

    internal_helpers = {
        "merge_guaranteed_fields",
        "merge_union_contracts",
        "merge_union_fields",
    }

    assert internal_helpers.isdisjoint(dag.__all__)
    for helper_name in internal_helpers:
        assert not hasattr(dag, helper_name)
        with pytest.raises(ImportError):
            exec(f"from elspeth.core.dag import {helper_name}", {})
