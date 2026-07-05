"""Shared declaration-contract helper tests."""

from __future__ import annotations

import pytest

from elspeth.contracts.errors import FrameworkBugError
from elspeth.contracts.schema_contract import FieldContract, PipelineRow, SchemaContract
from elspeth.engine.executors.declaration_flags import _runtime_observed_fields


def _contract(fields: tuple[str, ...]) -> SchemaContract:
    return SchemaContract(
        mode="OBSERVED",
        fields=tuple(
            FieldContract(
                normalized_name=name,
                original_name=name,
                python_type=str,
                required=True,
                source="inferred",
                nullable=False,
            )
            for name in fields
        ),
        locked=True,
    )


def test_runtime_observed_fields_intersects_contract_fields_and_payload_keys() -> None:
    emitted = PipelineRow(
        {"shared": "v", "payload_only": "v"},
        _contract(("shared", "contract_only")),
    )

    assert _runtime_observed_fields(emitted, plugin_name="Plugin") == frozenset({"shared"})


def test_runtime_observed_fields_rejects_missing_contract() -> None:
    emitted = PipelineRow({"field": "v"}, None)

    with pytest.raises(FrameworkBugError, match=r"Plugin.*no contract"):
        _runtime_observed_fields(emitted, plugin_name="Plugin")
