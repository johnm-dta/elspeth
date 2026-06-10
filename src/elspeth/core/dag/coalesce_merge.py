# src/elspeth/core/dag/coalesce_merge.py
"""Coalesce merge logic for schema computation.

Extracted from builder.py to enable direct testing without reimplementation.
This module contains the build-time wrappers for merging schemas at coalesce
points.

Two operations are provided:

1. merge_guaranteed_fields: Combines effective guarantees from branch schemas
   using union (require_all) or intersection (other policies).

2. merge_union_fields: Combines typed field definitions with policy-aware
   required/optional semantics for union merge strategy. The per-field
   algorithm lives in elspeth.contracts.union_merge (merge_union_field_flags)
   and is shared with the runtime merge_union_contracts, so build-time and
   runtime union merges cannot diverge.

Both functions are called by builder.py during graph construction and can
be called directly by tests to verify semantics without reimplementing logic.

merge_union_contracts is re-exported here for discoverability next to its
build-time sibling.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from typing import Literal

from elspeth.contracts.schema import FieldDefinition, SchemaConfig
from elspeth.contracts.union_merge import (
    UnionTypeConflictError,
    merge_union_contracts,
    merge_union_field_flags,
)
from elspeth.core.dag.models import GraphValidationError

__all__ = [
    "merge_guaranteed_fields",
    "merge_union_contracts",
    "merge_union_fields",
]


def merge_guaranteed_fields(
    branch_schemas: Mapping[str, SchemaConfig],
    *,
    require_all: bool,
) -> tuple[str, ...] | None:
    """Merge guaranteed fields from branch schemas.

    Computes the merged guaranteed_fields tuple for a coalesce node based on
    the effective guarantees from each branch and the coalesce policy.

    Args:
        branch_schemas: Map of branch name to SchemaConfig
        require_all: If True, use union semantics (all branches always arrive,
            so any branch's guarantee survives). If False, use intersection
            semantics (some branches may be lost, so only shared guarantees
            survive).

    Returns:
        Merged guaranteed fields tuple, or None if no branch has effective
        guarantees (abstention semantics — the coalesce makes no claim).

    Note:
        The None-vs-empty-tuple distinction is semantic:
        - None = no branch has effective guarantees (abstain from vote)
        - () = branches have guarantees but merge is empty set (explicit zero)
    """
    guaranteed_sets: list[set[str]] = []
    for schema_cfg in branch_schemas.values():
        if schema_cfg.has_effective_guarantees:
            guaranteed_sets.append(set(schema_cfg.get_effective_guaranteed_fields()))

    if not guaranteed_sets:
        return None

    if require_all:
        merged = set.union(*guaranteed_sets)
    else:
        merged = set.intersection(*guaranteed_sets)

    return tuple(sorted(merged)) if merged else ()


def merge_union_fields(
    branch_schemas: Mapping[str, SchemaConfig],
    *,
    require_all: bool,
    collision_policy: Literal["last_wins", "first_wins", "fail"] = "last_wins",
    branch_order: Sequence[str] | None = None,
    coalesce_id: str | None = None,
    guaranteed_fields: tuple[str, ...] | None = None,
    audit_fields: tuple[str, ...] | None = None,
) -> SchemaConfig:
    """Merge typed fields from branch schemas using union merge strategy.

    Combines field definitions from all branches with policy-aware handling
    of required/optional semantics:

    - require_all (OR semantics): A field is required if required in ANY
      branch. Since all branches always arrive under require_all, any branch's
      guarantee is honored in the merged output.

    - other policies (AND semantics): A field is required only if required in
      ALL branches. Since some branches may be lost, only shared guarantees
      survive.

    Branch-exclusive fields (present in only one branch) are handled specially:
    - require_all: Preserve source branch's required flag (branch always arrives)
    - other policies: Force optional (branch may not arrive)

    Nullable semantics for shared fields depend on collision_policy (D5 fix):
    - first_wins: Use first branch's nullable (first branch's value always wins)
    - last_wins: Use last branch's nullable (last branch's value always wins)
    - fail: Use OR of all branches (conservative; collisions fail at runtime)

    Args:
        branch_schemas: Map of branch name to SchemaConfig
        require_all: If True, use OR semantics for required; if False, use AND
        collision_policy: How to resolve field-level collisions in union merge.
            Affects nullable computation for shared fields.
        branch_order: Declaration order of branches. If provided, branches are
            processed in this order for deterministic first/last semantics.
            If None, uses dict iteration order (insertion order in Python 3.7+).
        coalesce_id: Optional node ID for error messages (default: generic)
        guaranteed_fields: Pre-computed guaranteed_fields to include in result
        audit_fields: Pre-computed audit_fields to include in result

    Returns:
        Merged SchemaConfig. If all branches are observed-mode or no branches
        contribute fields, returns an observed-mode schema. Otherwise returns
        a flexible-mode schema with merged field definitions.

    Raises:
        GraphValidationError: If branches have incompatible types for the same
            field name.
    """
    # Check if ALL branches are observed BEFORE processing. Only return observed
    # mode if every branch is observed - otherwise process the typed branches.
    # Previous bug: the loop would break on the FIRST observed branch, silently
    # discarding typed fields from subsequent branches.
    all_observed = all(schema_cfg.is_observed for schema_cfg in branch_schemas.values())
    if all_observed:
        # Early return: all branches are observed, no typed fields to merge
        return SchemaConfig(
            mode="observed",
            fields=None,
            guaranteed_fields=guaranteed_fields,
            audit_fields=audit_fields,
        )

    # Build the core algorithm's input, EXCLUDING non-contributing branches:
    # observed branches contribute no typed fields (mixed observed/explicit is
    # handled by processing only the explicit branches; upstream validation may
    # reject mixed schemas at a higher level), and fields=None branches have
    # nothing to contribute. A typed branch with fields=() still counts as
    # contributing (it can force siblings' exclusive fields optional).
    #
    # Iteration order: branch_order if provided (declaration order from config),
    # otherwise dict iteration order. This is critical for first_wins/last_wins
    # nullable semantics — the first/last branch in declaration order determines
    # the winning value. branch_order may include branches not in branch_schemas
    # (e.g., not wired up yet); the core skips names absent from its input.
    branch_fields: dict[str, list[tuple[str, Hashable, bool, bool]]] = {
        branch_name: [(fd.name, fd.field_type, fd.required, fd.nullable) for fd in schema_cfg.fields]
        for branch_name, schema_cfg in branch_schemas.items()
        if not schema_cfg.is_observed and schema_cfg.fields is not None
    }

    try:
        merged_flags = merge_union_field_flags(
            branch_fields,
            require_all=require_all,
            collision_policy=collision_policy,
            branch_order=branch_order,
        )
    except UnionTypeConflictError as e:
        node_desc = f"'{coalesce_id}'" if coalesce_id else "coalesce node"
        raise GraphValidationError(
            f"Coalesce node {node_desc} receives incompatible "
            f"types for field '{e.field}' in union merge: "
            f"branch '{e.branch_a}' has {e.type_a!r}, "
            f"branch '{e.branch_b}' has {e.type_b!r}. "
            "Union merge requires compatible types on shared fields.",
            component_id=coalesce_id,
            component_type="coalesce",
        ) from e

    if not merged_flags:
        # No typed fields from any branch (all branches had fields=None or were
        # observed and skipped). Return observed mode.
        return SchemaConfig(
            mode="observed",
            fields=None,
            guaranteed_fields=guaranteed_fields,
            audit_fields=audit_fields,
        )

    merged_fields = tuple(
        FieldDefinition(name=name, field_type=ftype, required=req, nullable=is_nullable)  # type: ignore[arg-type]
        for name, (ftype, req, is_nullable, _) in merged_flags.items()
    )
    return SchemaConfig(
        mode="flexible",
        fields=merged_fields,
        guaranteed_fields=guaranteed_fields,
        audit_fields=audit_fields,
    )
