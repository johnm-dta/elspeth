"""Canonical union-merge algorithm for coalesce schema/contract merging.

This module is the single source of truth for policy-aware union merge
semantics. Two thin wrappers consume the core algorithm:

1. merge_union_fields (core/dag/coalesce_merge.py): build-time merge of
   SchemaConfig field definitions during DAG construction.
2. merge_union_contracts (this module): runtime merge of SchemaContract
   instances for all-OBSERVED union coalesces.

Both derive required/nullable flags from merge_union_field_flags so build-time
and runtime merges cannot diverge.

Policy semantics (shared by both wrappers):

- require_all (OR semantics): A field is required if required in ANY branch.
  Since all branches always arrive under require_all, any branch's guarantee
  is honored in the merged output. Branch-exclusive fields keep their source
  branch's flags (the branch always arrives).

- other policies (AND semantics): A field is required only if required in ALL
  branches. Since some branches may be lost, only shared guarantees survive.
  Branch-exclusive fields are forced optional and nullable (the branch may
  not arrive, leaving the field absent/None).

Nullable semantics for shared fields under require_all depend on
collision_policy (D5 fix): first_wins/last_wins use the winning branch's
nullable; fail uses OR of all branches (conservative). Under non-require_all,
nullable is always OR of all branches (P1 soundness: any branch might be the
one that arrives).
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from typing import Literal, cast

from elspeth.contracts.errors import ContractMergeError
from elspeth.contracts.schema_contract import FieldContract, SchemaContract


class UnionTypeConflictError(Exception):
    """Two branches contribute incompatible types for the same field.

    Neutral conflict signal raised by merge_union_field_flags. Wrappers
    translate it to their domain error (GraphValidationError at build time,
    ContractMergeError at runtime).

    Attributes:
        field: The field name with conflicting types
        type_a: The type key from the first-seen branch
        type_b: The type key from the conflicting branch
        branch_a: The branch that first contributed the field
        branch_b: The branch with the conflicting type
    """

    def __init__(
        self,
        *,
        field: str,
        type_a: Hashable,
        type_b: Hashable,
        branch_a: str,
        branch_b: str,
    ) -> None:
        self.field = field
        self.type_a = type_a
        self.type_b = type_b
        self.branch_a = branch_a
        self.branch_b = branch_b
        super().__init__(
            f"Incompatible types for field '{field}' in union merge: "
            f"branch '{branch_a}' has {type_a!r}, branch '{branch_b}' has {type_b!r}."
        )


def merge_union_field_flags(
    branch_fields: Mapping[str, Sequence[tuple[str, Hashable, bool, bool]]],
    *,
    require_all: bool,
    collision_policy: Literal["last_wins", "first_wins", "fail"] = "last_wins",
    branch_order: Sequence[str] | None = None,
) -> dict[str, tuple[Hashable, bool, bool, str]]:
    """Merge per-field (type, required, nullable) flags across branches.

    The canonical per-field union algorithm. Type keys are opaque hashables:
    the build-time wrapper passes field_type strings, the runtime wrapper
    passes Python types — the algorithm is shared, the field dataclasses
    are not.

    Args:
        branch_fields: Map of branch name to sequence of
            (name, type_key, required, nullable) tuples. EVERY key counts as
            a contributing branch (even with an empty sequence) — callers
            control membership by pre-filtering non-contributing branches.
        require_all: If True, use OR semantics for required; if False, use AND
        collision_policy: How shared-field nullable resolves under require_all.
        branch_order: Iteration order for branches. Names absent from
            branch_fields are skipped; if None, uses dict iteration order.

    Returns:
        Map of field name to (type_key, required, nullable,
        first_contributing_branch), in first-seen insertion order.

    Raises:
        UnionTypeConflictError: If branches have incompatible type keys for
            the same field name.
    """
    seen_types: dict[str, tuple[Hashable, bool, bool, str]] = {}
    branches_with_field: dict[str, set[str]] = {}
    contributing_branches: set[str] = set()

    branch_names = branch_order if branch_order is not None else list(branch_fields.keys())

    for branch_name in branch_names:
        if branch_name not in branch_fields:
            # branch_order may include branches the caller filtered out (e.g.,
            # observed branches at build time, lost branches at runtime).
            continue
        contributing_branches.add(branch_name)
        for name, type_key, required, nullable in branch_fields[branch_name]:
            if name not in branches_with_field:
                branches_with_field[name] = set()
            branches_with_field[name].add(branch_name)

            if name in seen_types:
                prior_type, prior_req, prior_nullable, prior_branch = seen_types[name]
                if prior_type != type_key:
                    raise UnionTypeConflictError(
                        field=name,
                        type_a=prior_type,
                        type_b=type_key,
                        branch_a=prior_branch,
                        branch_b=branch_name,
                    )
                if require_all:
                    # OR for required: required if required in ANY branch.
                    merged_req = prior_req or required
                    # Nullable depends on collision_policy (D5 fix):
                    # - first_wins: keep first branch's nullable (prior_nullable)
                    # - last_wins: use current branch's nullable
                    # - fail: OR of all (conservative; collisions fail at runtime)
                    if collision_policy == "first_wins":
                        merged_nullable = prior_nullable  # First seen wins
                    elif collision_policy == "last_wins":
                        merged_nullable = nullable  # Last seen wins
                    else:  # "fail" — conservative OR
                        merged_nullable = prior_nullable or nullable
                else:
                    # AND: optional if optional in ANY branch.
                    merged_req = prior_req and required
                    # Under partial-arrival (non-require_all), any branch might be
                    # the one that arrives. The collision_policy determines VALUE
                    # resolution if multiple arrive, but the SCHEMA must be sound
                    # for all arrival combinations. If ANY branch can produce None,
                    # the merged schema must be nullable. (P1 soundness fix)
                    merged_nullable = prior_nullable or nullable
                seen_types[name] = (prior_type, merged_req, merged_nullable, prior_branch)
            else:
                seen_types[name] = (type_key, required, nullable, branch_name)

    # Branch-exclusive field handling (post-loop pass):
    # - require_all: keep the source-branch flags (branch always arrives)
    # - other policies: force optional + nullable (branch may not arrive)
    if not require_all:
        for field_name in list(seen_types):
            if branches_with_field[field_name] != contributing_branches:
                ftype, _, _, first_branch = seen_types[field_name]
                seen_types[field_name] = (ftype, False, True, first_branch)

    return seen_types


def merge_union_contracts(
    branch_contracts: Mapping[str, SchemaContract],
    *,
    require_all: bool,
    collision_policy: Literal["last_wins", "first_wins", "fail"] = "last_wins",
    branch_order: Sequence[str] | None = None,
    coalesce_id: str | None = None,
) -> SchemaContract:
    """Merge runtime SchemaContracts at a union coalesce, policy-aware.

    Runtime sibling of merge_union_fields (core/dag/coalesce_merge.py) — both
    delegate flag computation to merge_union_field_flags, so build-time and
    runtime union merges share one algorithm.

    Unlike the build-time wrapper, EVERY branch contributes — an arrived
    branch with zero fields still forces siblings' exclusive fields optional
    under non-require_all policies (it arrived; its rows lack those fields).

    Runtime-only attributes are layered on top of the core flags:
    - mode: most restrictive across branches (FIXED > FLEXIBLE > OBSERVED)
    - locked: True if any branch is locked
    - source: 'declared' if any branch with the field declares it
    - original_name: from the first contributing branch (in branch_order)
      that carries the field

    Args:
        branch_contracts: Map of branch name to that branch's SchemaContract
        require_all: If True, use OR semantics for required; if False, use AND
        collision_policy: How shared-field nullable resolves under require_all
        branch_order: Declaration order of branches for deterministic
            first/last semantics. Names absent from branch_contracts are
            skipped; if None, uses dict iteration order.
        coalesce_id: Optional node ID (reserved for error context; the raised
            ContractMergeError carries field/type detail only)

    Returns:
        New merged SchemaContract with fields sorted by normalized_name.

    Raises:
        ContractMergeError: If branches have incompatible types for the same
            field name.
        ValueError: If branch_contracts is empty (nothing to merge).
    """
    if not branch_contracts:
        node_desc = f"'{coalesce_id}'" if coalesce_id else "coalesce node"
        raise ValueError(f"merge_union_contracts at {node_desc} requires at least one branch contract")

    branch_names = branch_order if branch_order is not None else list(branch_contracts.keys())
    ordered_names = [name for name in branch_names if name in branch_contracts]

    if len(ordered_names) == 1:
        # Single contributing branch (e.g., policy='first' merging the first
        # arrival): the union of one contract is that contract. Return it
        # unchanged — no flag rewrites apply (every field is "shared" w.r.t.
        # the contributing set) and callers rely on field order/identity
        # passing through.
        return branch_contracts[ordered_names[0]]

    branch_fields: dict[str, list[tuple[str, type, bool, bool]]] = {
        name: [(fc.normalized_name, fc.python_type, fc.required, fc.nullable) for fc in branch_contracts[name].fields]
        for name in ordered_names
    }

    try:
        merged_flags = merge_union_field_flags(
            branch_fields,
            require_all=require_all,
            collision_policy=collision_policy,
            branch_order=ordered_names,
        )
    except UnionTypeConflictError as e:
        # Type keys here are always Python types (built from fc.python_type)
        raise ContractMergeError(
            field=e.field,
            type_a=cast(type, e.type_a).__name__,
            type_b=cast(type, e.type_b).__name__,
        ) from e

    # Mode precedence: FIXED > FLEXIBLE > OBSERVED (most restrictive wins)
    mode_order: dict[str, int] = {"FIXED": 0, "FLEXIBLE": 1, "OBSERVED": 2}
    merged_mode = min(
        (branch_contracts[name].mode for name in ordered_names),
        key=lambda m: mode_order[m],
    )
    merged_locked = any(branch_contracts[name].locked for name in ordered_names)

    merged_fields: list[FieldContract] = []
    for field_name in sorted(merged_flags):
        type_key, required, nullable, _first_branch = merged_flags[field_name]
        carriers = [fc for name in ordered_names if (fc := branch_contracts[name].find_field(field_name)) is not None]
        # carriers is non-empty: the field came from at least one branch's fields
        merged_fields.append(
            FieldContract(
                normalized_name=field_name,
                original_name=carriers[0].original_name,
                python_type=cast(type, type_key),
                required=required,
                source="declared" if any(fc.source == "declared" for fc in carriers) else "inferred",
                nullable=nullable,
            )
        )

    return SchemaContract(
        mode=merged_mode,
        fields=tuple(merged_fields),
        locked=merged_locked,
    )
