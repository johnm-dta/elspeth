# tests/integration/contracts/test_build_runtime_consistency.py
"""Build↔runtime parity tests for the canonical union-merge algorithm.

The DAG builder precomputes coalesce schemas during compilation using
merge_union_fields() (SchemaConfig level). The coalesce executor merges
all-OBSERVED branch contracts at runtime using merge_union_contracts()
(SchemaContract level). Both are thin wrappers over the same core algorithm
(elspeth.contracts.union_merge.merge_union_field_flags), so for any policy
configuration they must produce identical per-field (type, required, nullable)
results on equivalent inputs.

This suite pins that parity across the full policy matrix:
require_all x collision_policy x branch_order.
"""

from __future__ import annotations

from typing import Literal

from hypothesis import given, settings
from hypothesis import strategies as st

from elspeth.contracts.schema import FieldDefinition, SchemaConfig
from elspeth.contracts.schema_contract_factory import create_contract_from_config
from elspeth.contracts.union_merge import merge_union_contracts
from elspeth.core.dag.coalesce_merge import merge_union_fields

# Type alias matching FieldDefinition.field_type
_FieldType = Literal["str", "int", "float", "bool", "any"]
_CollisionPolicy = Literal["last_wins", "first_wins", "fail"]

# Mapping from SchemaConfig field types to runtime contract types
_TYPE_MAP: dict[str, type] = {"int": int, "str": str, "float": float, "bool": bool, "any": object}


# =============================================================================
# Hypothesis Strategies
# =============================================================================


field_types: st.SearchStrategy[_FieldType] = st.sampled_from(["int", "str", "float", "bool"])

collision_policies: st.SearchStrategy[_CollisionPolicy] = st.sampled_from(["last_wins", "first_wins", "fail"])


@st.composite
def schema_configs_for_merge(draw: st.DrawFn) -> tuple[SchemaConfig, SchemaConfig]:
    """Generate two SchemaConfigs suitable for merging.

    Ensures:
    - Shared fields have the same type (required for merge)
    - At least one field in each config
    """
    # Generate shared field names and types
    n_shared = draw(st.integers(min_value=1, max_value=3))
    n_only_a = draw(st.integers(min_value=0, max_value=2))
    n_only_b = draw(st.integers(min_value=0, max_value=2))

    shared_names = [f"s{i}" for i in range(n_shared)]
    only_a_names = [f"a{i}" for i in range(n_only_a)]
    only_b_names = [f"b{i}" for i in range(n_only_b)]

    # Shared fields: same type in both
    shared_type_map = {name: draw(field_types) for name in shared_names}

    fields_a: list[FieldDefinition] = []
    fields_b: list[FieldDefinition] = []

    # Add shared fields
    for name in shared_names:
        fields_a.append(
            FieldDefinition(
                name=name,
                field_type=shared_type_map[name],
                required=draw(st.booleans()),
                nullable=draw(st.booleans()),
            )
        )
        fields_b.append(
            FieldDefinition(
                name=name,
                field_type=shared_type_map[name],  # Same type
                required=draw(st.booleans()),
                nullable=draw(st.booleans()),
            )
        )

    # Add exclusive fields
    for name in only_a_names:
        fields_a.append(
            FieldDefinition(
                name=name,
                field_type=draw(field_types),
                required=draw(st.booleans()),
                nullable=draw(st.booleans()),
            )
        )
    for name in only_b_names:
        fields_b.append(
            FieldDefinition(
                name=name,
                field_type=draw(field_types),
                required=draw(st.booleans()),
                nullable=draw(st.booleans()),
            )
        )

    config_a = SchemaConfig(mode="flexible", fields=tuple(fields_a))
    config_b = SchemaConfig(mode="flexible", fields=tuple(fields_b))

    return config_a, config_b


# =============================================================================
# Build↔Runtime Parity
# =============================================================================


class TestBuildRuntimeParity:
    """merge_union_fields and merge_union_contracts must agree on equivalent inputs.

    Strictly stronger than the retired divergence suite: every policy
    combination is compared on the full per-field (type, required, nullable)
    map, not just nullable under collision_policy='fail'.
    """

    @given(
        configs=schema_configs_for_merge(),
        require_all=st.booleans(),
        collision_policy=collision_policies,
        a_first=st.booleans(),
    )
    @settings(max_examples=300)
    def test_parity_property(
        self,
        configs: tuple[SchemaConfig, SchemaConfig],
        require_all: bool,
        collision_policy: _CollisionPolicy,
        a_first: bool,
    ) -> None:
        """Property: build-time and runtime merges produce identical field flags."""
        config_a, config_b = configs
        branch_order = ("a", "b") if a_first else ("b", "a")

        build_merged = merge_union_fields(
            {"a": config_a, "b": config_b},
            require_all=require_all,
            collision_policy=collision_policy,
            branch_order=branch_order,
        )

        runtime_merged = merge_union_contracts(
            {"a": create_contract_from_config(config_a), "b": create_contract_from_config(config_b)},
            require_all=require_all,
            collision_policy=collision_policy,
            branch_order=branch_order,
        )

        assert build_merged.fields is not None
        build_flags = {f.name: (_TYPE_MAP[f.field_type], f.required, f.nullable) for f in build_merged.fields}
        runtime_flags = {f.normalized_name: (f.python_type, f.required, f.nullable) for f in runtime_merged.fields}

        assert set(build_flags) == set(runtime_flags), f"Field set mismatch: build={set(build_flags)}, runtime={set(runtime_flags)}"
        assert build_flags == runtime_flags, f"Per-field flag mismatch: build={build_flags}, runtime={runtime_flags}"

    def test_require_all_exclusive_field_keeps_source_flags(self) -> None:
        """Concrete example: under require_all, a branch-exclusive field keeps its flags in BOTH merges."""
        config_a = SchemaConfig(
            mode="flexible",
            fields=(
                FieldDefinition(name="shared", field_type="int", required=True, nullable=False),
                FieldDefinition(name="a_only", field_type="str", required=True, nullable=False),
            ),
        )
        config_b = SchemaConfig(
            mode="flexible",
            fields=(FieldDefinition(name="shared", field_type="int", required=True, nullable=False),),
        )

        build_merged = merge_union_fields({"a": config_a, "b": config_b}, require_all=True, branch_order=("a", "b"))
        runtime_merged = merge_union_contracts(
            {"a": create_contract_from_config(config_a), "b": create_contract_from_config(config_b)},
            require_all=True,
            branch_order=("a", "b"),
        )

        assert build_merged.fields is not None
        build_a_only = next(f for f in build_merged.fields if f.name == "a_only")
        runtime_a_only = next(f for f in runtime_merged.fields if f.normalized_name == "a_only")

        # Branch always arrives under require_all → source flags preserved
        assert (build_a_only.required, build_a_only.nullable) == (True, False)
        assert (runtime_a_only.required, runtime_a_only.nullable) == (True, False)

    def test_best_effort_exclusive_field_forced_optional_nullable(self) -> None:
        """Concrete example: under best_effort, a branch-exclusive field is forced optional+nullable in BOTH merges."""
        config_a = SchemaConfig(
            mode="flexible",
            fields=(
                FieldDefinition(name="shared", field_type="int", required=True, nullable=False),
                FieldDefinition(name="a_only", field_type="str", required=True, nullable=False),
            ),
        )
        config_b = SchemaConfig(
            mode="flexible",
            fields=(FieldDefinition(name="shared", field_type="int", required=True, nullable=False),),
        )

        build_merged = merge_union_fields({"a": config_a, "b": config_b}, require_all=False, branch_order=("a", "b"))
        runtime_merged = merge_union_contracts(
            {"a": create_contract_from_config(config_a), "b": create_contract_from_config(config_b)},
            require_all=False,
            branch_order=("a", "b"),
        )

        assert build_merged.fields is not None
        build_a_only = next(f for f in build_merged.fields if f.name == "a_only")
        runtime_a_only = next(f for f in runtime_merged.fields if f.normalized_name == "a_only")

        # Branch may not arrive → forced optional and nullable
        assert (build_a_only.required, build_a_only.nullable) == (False, True)
        assert (runtime_a_only.required, runtime_a_only.nullable) == (False, True)
