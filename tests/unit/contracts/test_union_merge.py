# tests/unit/contracts/test_union_merge.py
"""Unit tests for merge_union_contracts (runtime policy-aware union merge).

merge_union_contracts is the runtime sibling of the build-time
merge_union_fields: both delegate per-field flag computation to the canonical
merge_union_field_flags algorithm. These tests pin the policy matrix
(require_all x collision_policy), the runtime-only attribute layering (mode,
locked, source, original_name), and the delegation invariant for
SchemaContract.merge_for_batch.
"""

from __future__ import annotations

import pytest

from elspeth.contracts.errors import ContractMergeError
from elspeth.contracts.schema_contract import FieldContract, SchemaContract
from elspeth.contracts.union_merge import merge_union_contracts


def _field(
    name: str,
    python_type: type = int,
    *,
    original: str | None = None,
    required: bool = False,
    source: str = "inferred",
    nullable: bool = False,
) -> FieldContract:
    return FieldContract(
        normalized_name=name,
        original_name=original if original is not None else name.upper(),
        python_type=python_type,
        required=required,
        source=source,  # type: ignore[arg-type]
        nullable=nullable,
    )


def _contract(
    *fields: FieldContract,
    mode: str = "FLEXIBLE",
    locked: bool = False,
) -> SchemaContract:
    return SchemaContract(mode=mode, fields=fields, locked=locked)  # type: ignore[arg-type]


class TestRequiredSemantics:
    """require_all → OR-required; other policies → AND-required."""

    def test_require_all_or_required_shared_field(self) -> None:
        a = _contract(_field("x", required=True))
        b = _contract(_field("x", required=False))
        merged = merge_union_contracts({"a": a, "b": b}, require_all=True, branch_order=("a", "b"))
        assert merged.get_field("x").required is True

    def test_best_effort_and_required_shared_field(self) -> None:
        a = _contract(_field("x", required=True))
        b = _contract(_field("x", required=False))
        merged = merge_union_contracts({"a": a, "b": b}, require_all=False, branch_order=("a", "b"))
        assert merged.get_field("x").required is False

    def test_require_all_exclusive_field_keeps_source_flags(self) -> None:
        """Branch always arrives under require_all → exclusive fields keep their flags."""
        a = _contract(_field("x", required=True), _field("a_only", str, required=True, nullable=False))
        b = _contract(_field("x", required=True))
        merged = merge_union_contracts({"a": a, "b": b}, require_all=True, branch_order=("a", "b"))
        a_only = merged.get_field("a_only")
        assert a_only.required is True
        assert a_only.nullable is False

    def test_best_effort_exclusive_field_forced_optional_nullable(self) -> None:
        """Branch may not arrive under best_effort → exclusive fields forced optional+nullable."""
        a = _contract(_field("x", required=True), _field("a_only", str, required=True, nullable=False))
        b = _contract(_field("x", required=True))
        merged = merge_union_contracts({"a": a, "b": b}, require_all=False, branch_order=("a", "b"))
        a_only = merged.get_field("a_only")
        assert a_only.required is False
        assert a_only.nullable is True


class TestNullableSemantics:
    """Shared-field nullable: collision_policy-aware under require_all, OR otherwise."""

    def _branches(self) -> dict[str, SchemaContract]:
        # a declares x nullable, b declares x non-nullable
        return {
            "a": _contract(_field("x", required=True, nullable=True)),
            "b": _contract(_field("x", required=True, nullable=False)),
        }

    def test_require_all_first_wins_uses_first_branch_nullable(self) -> None:
        merged = merge_union_contracts(self._branches(), require_all=True, collision_policy="first_wins", branch_order=("a", "b"))
        assert merged.get_field("x").nullable is True

    def test_require_all_last_wins_uses_last_branch_nullable(self) -> None:
        merged = merge_union_contracts(self._branches(), require_all=True, collision_policy="last_wins", branch_order=("a", "b"))
        assert merged.get_field("x").nullable is False

    def test_require_all_fail_uses_or(self) -> None:
        merged = merge_union_contracts(self._branches(), require_all=True, collision_policy="fail", branch_order=("a", "b"))
        assert merged.get_field("x").nullable is True

    @pytest.mark.parametrize("collision_policy", ["first_wins", "last_wins", "fail"])
    def test_non_require_all_always_or(self, collision_policy: str) -> None:
        """P1 soundness: under partial arrival, any nullable branch forces nullable."""
        merged = merge_union_contracts(
            self._branches(),
            require_all=False,
            collision_policy=collision_policy,  # type: ignore[arg-type]
            branch_order=("a", "b"),
        )
        assert merged.get_field("x").nullable is True

    def test_branch_order_controls_first_last(self) -> None:
        """Declaration order (branch_order), not dict order, decides first/last."""
        merged = merge_union_contracts(
            self._branches(),
            require_all=True,
            collision_policy="first_wins",
            branch_order=("b", "a"),  # b first despite dict order
        )
        assert merged.get_field("x").nullable is False


class TestAllObservedProductionShape:
    """All-OBSERVED inferred-field contracts: the production coalesce fold input.

    Production OBSERVED contracts only carry required=False/nullable=False
    inferred fields (with_field hardcodes those). Pins the one production-
    observable change of the schema-merge collapse: under require_all,
    branch-exclusive fields keep (required=False, nullable=False) instead of
    the old forced (False, True).
    """

    def _branches(self) -> dict[str, SchemaContract]:
        a = _contract(
            _field("shared", required=False, nullable=False),
            _field("a_only", str, required=False, nullable=False),
            mode="OBSERVED",
            locked=True,
        )
        b = _contract(
            _field("shared", required=False, nullable=False),
            _field("b_only", float, required=False, nullable=False),
            mode="OBSERVED",
            locked=True,
        )
        return {"a": a, "b": b}

    def test_require_all_exclusive_fields_keep_flags(self) -> None:
        merged = merge_union_contracts(self._branches(), require_all=True, branch_order=("a", "b"))
        for name in ("a_only", "b_only"):
            fc = merged.get_field(name)
            assert fc.required is False
            assert fc.nullable is False, f"require_all: exclusive field '{name}' keeps nullable=False (branch always arrives)"

    def test_non_require_all_exclusive_fields_forced(self) -> None:
        merged = merge_union_contracts(self._branches(), require_all=False, branch_order=("a", "b"))
        for name in ("a_only", "b_only"):
            fc = merged.get_field(name)
            assert fc.required is False
            assert fc.nullable is True, f"best_effort: exclusive field '{name}' forced nullable (branch may not arrive)"

    def test_mode_and_locked(self) -> None:
        merged = merge_union_contracts(self._branches(), require_all=True, branch_order=("a", "b"))
        assert merged.mode == "OBSERVED"
        assert merged.locked is True


class TestRuntimeAttributeLayering:
    """Mode precedence, locked-OR, source OR-declared, original_name resolution."""

    def test_mode_most_restrictive_wins(self) -> None:
        fixed = _contract(mode="FIXED")
        flexible = _contract(mode="FLEXIBLE")
        observed = _contract(mode="OBSERVED")
        assert merge_union_contracts({"a": fixed, "b": observed}, require_all=True).mode == "FIXED"
        assert merge_union_contracts({"a": observed, "b": fixed}, require_all=True).mode == "FIXED"
        assert merge_union_contracts({"a": flexible, "b": observed}, require_all=False).mode == "FLEXIBLE"

    def test_locked_or(self) -> None:
        locked = _contract(locked=True)
        unlocked = _contract(locked=False)
        assert merge_union_contracts({"a": locked, "b": unlocked}, require_all=True).locked is True
        assert merge_union_contracts({"a": unlocked, "b": locked}, require_all=False).locked is True
        assert merge_union_contracts({"a": unlocked, "b": _contract(locked=False)}, require_all=True).locked is False

    def test_source_declared_if_any_branch_declares(self) -> None:
        a = _contract(_field("x", source="inferred"))
        b = _contract(_field("x", source="declared"))
        merged = merge_union_contracts({"a": a, "b": b}, require_all=True, branch_order=("a", "b"))
        assert merged.get_field("x").source == "declared"

    def test_exclusive_field_keeps_own_source(self) -> None:
        a = _contract(_field("a_only", source="inferred"))
        b = _contract(_field("b_only", source="declared"))
        merged = merge_union_contracts({"a": a, "b": b}, require_all=False, branch_order=("a", "b"))
        assert merged.get_field("a_only").source == "inferred"
        assert merged.get_field("b_only").source == "declared"

    def test_original_name_from_first_declared_branch(self) -> None:
        """Shared-field original_name comes from the first branch in branch_order."""
        a = _contract(_field("x", original="X from A"))
        b = _contract(_field("x", original="X from B"))
        merged_ab = merge_union_contracts({"a": a, "b": b}, require_all=True, branch_order=("a", "b"))
        merged_ba = merge_union_contracts({"a": a, "b": b}, require_all=True, branch_order=("b", "a"))
        assert merged_ab.get_field("x").original_name == "X from A"
        assert merged_ba.get_field("x").original_name == "X from B"

    def test_fields_sorted_by_normalized_name(self) -> None:
        a = _contract(_field("zeta"), _field("bravo", str))
        b = _contract(_field("alpha", float), _field("yankee", bool))
        merged = merge_union_contracts({"a": a, "b": b}, require_all=True, branch_order=("a", "b"))
        assert [fc.normalized_name for fc in merged.fields] == ["alpha", "bravo", "yankee", "zeta"]


class TestEdgeCases:
    def test_empty_field_contract_still_forces_exclusives(self) -> None:
        """An arrived branch with zero fields counts as contributing (old fold parity).

        Unlike the build-time wrapper (which skips observed/fields-None
        branches), every runtime branch contributes: its rows arrived and lack
        the siblings' fields, so those fields must go optional+nullable under
        non-require_all.
        """
        a = _contract(_field("a_only", required=True, nullable=False))
        empty = _contract(mode="OBSERVED")
        merged = merge_union_contracts({"a": a, "empty": empty}, require_all=False, branch_order=("a", "empty"))
        fc = merged.get_field("a_only")
        assert fc.required is False
        assert fc.nullable is True

    def test_type_conflict_raises_contract_merge_error(self) -> None:
        a = _contract(_field("x", int))
        b = _contract(_field("x", str))
        with pytest.raises(ContractMergeError, match="conflicting types"):
            merge_union_contracts({"a": a, "b": b}, require_all=True, branch_order=("a", "b"))

    def test_type_conflict_error_carries_field_and_types(self) -> None:
        a = _contract(_field("amount", int))
        b = _contract(_field("amount", str))
        with pytest.raises(ContractMergeError) as exc_info:
            merge_union_contracts({"a": a, "b": b}, require_all=False)
        assert exc_info.value.field == "amount"
        assert {exc_info.value.type_a, exc_info.value.type_b} == {"int", "str"}

    def test_three_branch_merge(self) -> None:
        a = _contract(_field("x", required=True), _field("a_only", str, required=True))
        b = _contract(_field("x", required=True))
        c = _contract(_field("x", required=False), _field("c_only", float, required=True))
        merged = merge_union_contracts({"a": a, "b": b, "c": c}, require_all=True, branch_order=("a", "b", "c"))
        # Shared across all three: OR-required
        assert merged.get_field("x").required is True
        # Exclusive fields keep flags under require_all
        assert merged.get_field("a_only").required is True
        assert merged.get_field("c_only").required is True

        merged_be = merge_union_contracts({"a": a, "b": b, "c": c}, require_all=False, branch_order=("a", "b", "c"))
        assert merged_be.get_field("x").required is False  # AND: c does not require
        assert merged_be.get_field("a_only").required is False
        assert merged_be.get_field("a_only").nullable is True

    def test_empty_branch_map_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one branch contract"):
            merge_union_contracts({}, require_all=True)

    def test_branch_order_skips_missing_names(self) -> None:
        """branch_order may name branches that did not arrive (e.g., lost)."""
        a = _contract(_field("x", required=True))
        merged = merge_union_contracts({"a": a}, require_all=False, branch_order=("a", "lost_branch"))
        assert merged.get_field("x").required is True  # only contributing branch


class TestMergeForBatchDelegation:
    """merge_for_batch must be a thin delegate of merge_union_contracts."""

    def test_delegation_parity(self) -> None:
        a = _contract(
            _field("shared", required=True, nullable=False),
            _field("a_only", str, required=True, source="declared"),
            mode="FLEXIBLE",
            locked=True,
        )
        b = _contract(
            _field("shared", required=False, nullable=True),
            _field("b_only", float, required=True),
            mode="OBSERVED",
        )
        via_method = a.merge_for_batch(b)
        via_function = merge_union_contracts({"self": a, "other": b}, require_all=False, branch_order=("self", "other"))
        assert via_method == via_function
        assert via_method.version_hash() == via_function.version_hash()
