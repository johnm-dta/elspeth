"""Regression tests for MCP contract analyzer functions.

Bug fix covered:
- P1-2026-02-14: explain_field returns wrong field when lookup key matches
  one field's original_name and a different field's normalized_name.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Literal, cast

from elspeth.contracts.schema_contract import FieldContract, SchemaContract
from elspeth.mcp.analyzers.contracts import explain_field
from elspeth.mcp.types import FieldNotFoundError

SchemaMode = Literal["FIXED", "FLEXIBLE", "OBSERVED"]


def _make_contract(*fields: FieldContract, mode: SchemaMode = "OBSERVED") -> SchemaContract:
    """Create a SchemaContract from field specs."""
    return SchemaContract(mode=mode, fields=tuple(fields), locked=True)


class _RunLifecycleDouble:
    def __init__(self, contract: SchemaContract) -> None:
        self._contract = contract

    def get_run(self, _run_id: str) -> SimpleNamespace:
        return SimpleNamespace()

    def get_run_source_resume_records(self, _run_id: str) -> dict[str, SimpleNamespace]:
        return {
            "source-only": SimpleNamespace(
                source_node_id="source-only",
                source_name="primary",
                source_schema_json="{}",
                schema_contract=self._contract,
            ),
        }


class _RepositoryFactoryDouble:
    def __init__(self, contract: SchemaContract) -> None:
        self.run_lifecycle = _RunLifecycleDouble(contract)


class _DatabaseDouble:
    pass


def _make_factory(contract: SchemaContract) -> _RepositoryFactoryDouble:
    return _RepositoryFactoryDouble(contract)


class TestExplainFieldPrecedence:
    """Verify explain_field uses canonical name resolution.

    Regression: P1-2026-02-14 — explain_field used a linear scan with
    first-match-wins on (normalized_name OR original_name), so when a
    lookup key matched one field's original_name and a different field's
    normalized_name, the wrong field was returned.

    The fix delegates to SchemaContract.find_name() which uses deterministic
    precedence: normalized_name match takes priority over original_name match.
    """

    def test_normalized_name_takes_precedence_over_original_name(self) -> None:
        """When 'x' is both Field1.original_name and Field2.normalized_name,
        explain_field('x') must return Field2 (normalized match wins).
        """
        field_a = FieldContract(
            normalized_name="a",
            original_name="x",  # 'x' as original name
            python_type=int,
            required=True,
            source="inferred",
        )
        field_x = FieldContract(
            normalized_name="x",  # 'x' as normalized name
            original_name="y",
            python_type=str,
            required=True,
            source="inferred",
        )

        contract = _make_contract(field_a, field_x)

        # ADR-025 §3 Decision 5 (G6): MCP layer resolves contracts through
        # ``get_run_source_resume_records`` keyed by source_node_id; the
        # deleted singleton ``get_run_contract`` is gone.
        db = _DatabaseDouble()
        factory = _make_factory(contract)

        result = explain_field(db, factory, "run-123", "x")
        assert "error" not in result

        # Must return field_x (normalized_name='x'), not field_a (original_name='x')
        assert result["normalized_name"] == "x"
        assert result["original_name"] == "y"
        assert result["python_type"] == "str"

    def test_original_name_lookup_still_works(self) -> None:
        """Lookup by original_name still works when there is no normalized_name collision."""
        field_a = FieldContract(
            normalized_name="normalized_a",
            original_name="Original A",
            python_type=float,
            required=False,
            source="declared",
        )

        contract = _make_contract(field_a)

        # ADR-025 §3 Decision 5 (G6): MCP layer resolves contracts through
        # ``get_run_source_resume_records`` keyed by source_node_id; the
        # deleted singleton ``get_run_contract`` is gone.
        db = _DatabaseDouble()
        factory = _make_factory(contract)

        result = explain_field(db, factory, "run-123", "Original A")
        assert "error" not in result

        assert result["normalized_name"] == "normalized_a"
        assert result["original_name"] == "Original A"

    def test_field_not_found_returns_available_fields(self) -> None:
        """Missing fields return error with available field list."""
        field_a = FieldContract(
            normalized_name="amount",
            original_name="Amount",
            python_type=float,
            required=True,
            source="declared",
        )

        contract = _make_contract(field_a)

        # ADR-025 §3 Decision 5 (G6): MCP layer resolves contracts through
        # ``get_run_source_resume_records`` keyed by source_node_id; the
        # deleted singleton ``get_run_contract`` is gone.
        db = _DatabaseDouble()
        factory = _make_factory(contract)

        raw_result = explain_field(db, factory, "run-123", "nonexistent")
        result = cast(FieldNotFoundError, raw_result)
        assert "nonexistent" in result["error"]
        assert "amount" in result["available_fields"]
