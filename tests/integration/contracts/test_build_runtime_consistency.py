# tests/integration/contracts/test_build_runtime_consistency.py
"""Integration tests for build→runtime schema contract consistency.

D6 fix: Verifies that the build-time schema computation (merge_union_fields)
produces equivalent contracts to what runtime merge (SchemaContract.merge)
would produce. This ensures precomputed schemas match runtime behavior.

The gap this addresses: The DAG builder precomputes coalesce schemas during
compilation using merge_union_fields(). The coalesce executor has a fallback
path using SchemaContract.merge(). If these diverge, a pipeline compiled with
one schema could behave differently at runtime.
"""

from __future__ import annotations

from typing import Literal

from hypothesis import given, settings
from hypothesis import strategies as st

from elspeth.contracts.schema import FieldDefinition, SchemaConfig
from elspeth.contracts.schema_contract import SchemaContract
from elspeth.contracts.schema_contract_factory import create_contract_from_config
from elspeth.core.dag.coalesce_merge import merge_union_fields

# Type alias matching FieldDefinition.field_type
_FieldType = Literal["str", "int", "float", "bool", "any"]


# =============================================================================
# Test Helpers
# =============================================================================


def schema_config_to_contract(config: SchemaConfig) -> SchemaContract:
    """Convert SchemaConfig to SchemaContract for comparison.

    Uses the production factory to ensure test matches real behavior.
    """
    return create_contract_from_config(config)


def assert_contracts_equivalent(
    build_time: SchemaConfig,
    runtime: SchemaContract,
    *,
    check_mode: bool = True,
) -> None:
    """Assert that a SchemaConfig and SchemaContract represent equivalent schemas.

    Compares:
    - Field set (normalized names)
    - Field types
    - Field nullable flags
    - Optionally, mode (though mode semantics differ slightly between config and contract)

    Note: required field handling differs between merge_union_fields() and
    SchemaContract.merge() based on require_all flag, so we don't compare
    required here - that's policy-dependent.
    """
    # Convert build-time SchemaConfig to comparable form
    build_fields = {f.name: f for f in build_time.fields} if build_time.fields else {}
    runtime_fields = {f.normalized_name: f for f in runtime.fields}

    # Field sets must match
    assert set(build_fields.keys()) == set(runtime_fields.keys()), (
        f"Field set mismatch: build={set(build_fields.keys())}, runtime={set(runtime_fields.keys())}"
    )

    # Field types must match (via type string comparison)
    type_map = {"int": int, "str": str, "float": float, "bool": bool, "any": object}
    for name in build_fields:
        build_type = type_map[build_fields[name].field_type]
        runtime_type = runtime_fields[name].python_type
        assert build_type == runtime_type, f"Type mismatch for field '{name}': build={build_type}, runtime={runtime_type}"

    # Nullable must match (D7 fix)
    for name in build_fields:
        build_nullable = build_fields[name].nullable
        runtime_nullable = runtime_fields[name].nullable
        assert build_nullable == runtime_nullable, (
            f"Nullable mismatch for field '{name}': build={build_nullable}, runtime={runtime_nullable}"
        )


# =============================================================================
# Hypothesis Strategies
# =============================================================================


field_types: st.SearchStrategy[_FieldType] = st.sampled_from(["int", "str", "float", "bool"])


@st.composite
def field_definitions(draw: st.DrawFn) -> FieldDefinition:
    """Generate a FieldDefinition for testing."""
    name = draw(st.text(alphabet="abcdefghij", min_size=1, max_size=4))
    return FieldDefinition(
        name=name,
        field_type=draw(field_types),
        required=draw(st.booleans()),
        nullable=draw(st.booleans()),
    )


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
# Build→Runtime Consistency Tests
# =============================================================================


class TestBuildRuntimeNullableConsistency:
    """Test that build-time and runtime merge produce consistent nullable flags.

    This is the D6 gap: ensuring precomputed schemas behave correctly at runtime.

    Note: merge_union_fields() has policy-aware nullable semantics:
    - first_wins: uses first branch's nullable
    - last_wins: uses last branch's nullable (default)
    - fail: uses OR semantics

    SchemaContract.merge() always uses OR semantics, but per the code comment at
    schema_contract.py:489-494, this fallback is "never reached in production"
    since the precomputed schema is always used. Therefore:

    1. With collision_policy='fail', both use OR → should match
    2. With 'first_wins'/'last_wins', they intentionally differ
    3. The key invariant is that the executor uses the precomputed schema
    """

    def test_fail_policy_matches_runtime_or_semantics(self) -> None:
        """With collision_policy='fail', build-time uses OR, matching runtime merge.

        This is the only collision policy where both merge algorithms produce
        identical nullable results for shared fields.
        """
        config_a = SchemaConfig(
            mode="flexible",
            fields=(FieldDefinition(name="x", field_type="int", required=True, nullable=False),),
        )
        config_b = SchemaConfig(
            mode="flexible",
            fields=(FieldDefinition(name="x", field_type="int", required=True, nullable=True),),
        )

        # Build-time merge with 'fail' policy (uses OR semantics)
        build_merged = merge_union_fields(
            {"a": config_a, "b": config_b},
            require_all=True,
            collision_policy="fail",
        )

        # Runtime merge (always uses OR semantics)
        contract_a = schema_config_to_contract(config_a)
        contract_b = schema_config_to_contract(config_b)
        runtime_merged = contract_a.merge(contract_b)

        # Both must produce nullable=True (OR semantics: False OR True = True)
        assert build_merged.fields is not None
        build_x = next(f for f in build_merged.fields if f.name == "x")
        runtime_x = next(f for f in runtime_merged.fields if f.normalized_name == "x")

        assert build_x.nullable is True, "Build-time merge with 'fail' policy should use OR"
        assert runtime_x.nullable is True, "Runtime merge uses OR semantics"
        assert build_x.nullable == runtime_x.nullable

    def test_last_wins_policy_uses_last_branch_nullable(self) -> None:
        """With collision_policy='last_wins', build-time uses last branch's nullable.

        This intentionally differs from SchemaContract.merge() OR semantics.
        The precomputed schema is correct for the policy; runtime merge is a
        fallback that would give different (less accurate) results.
        """
        config_a = SchemaConfig(
            mode="flexible",
            fields=(FieldDefinition(name="x", field_type="int", required=True, nullable=True),),
        )
        config_b = SchemaConfig(
            mode="flexible",
            fields=(FieldDefinition(name="x", field_type="int", required=True, nullable=False),),
        )

        # Build-time merge with 'last_wins' (default)
        build_merged = merge_union_fields(
            {"a": config_a, "b": config_b},
            require_all=True,
            collision_policy="last_wins",
            branch_order=["a", "b"],  # b is last
        )

        assert build_merged.fields is not None
        build_x = next(f for f in build_merged.fields if f.name == "x")

        # Last branch (b) has nullable=False, so result should be False
        assert build_x.nullable is False, "last_wins should use last branch's nullable"

    def test_first_wins_policy_uses_first_branch_nullable(self) -> None:
        """With collision_policy='first_wins', build-time uses first branch's nullable."""
        config_a = SchemaConfig(
            mode="flexible",
            fields=(FieldDefinition(name="x", field_type="int", required=True, nullable=True),),
        )
        config_b = SchemaConfig(
            mode="flexible",
            fields=(FieldDefinition(name="x", field_type="int", required=True, nullable=False),),
        )

        # Build-time merge with 'first_wins'
        build_merged = merge_union_fields(
            {"a": config_a, "b": config_b},
            require_all=True,
            collision_policy="first_wins",
            branch_order=["a", "b"],  # a is first
        )

        assert build_merged.fields is not None
        build_x = next(f for f in build_merged.fields if f.name == "x")

        # First branch (a) has nullable=True, so result should be True
        assert build_x.nullable is True, "first_wins should use first branch's nullable"

    def test_branch_exclusive_nullable_consistency(self) -> None:
        """Build-time and runtime must agree on forced nullable for exclusive fields.

        Under best_effort (require_all=False), branch-exclusive fields become nullable
        because the source branch may not arrive. Both paths must produce the same result.
        """
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

        # Build-time merge with best_effort (AND semantics)
        build_merged = merge_union_fields({"a": config_a, "b": config_b}, require_all=False)

        # Runtime merge
        contract_a = schema_config_to_contract(config_a)
        contract_b = schema_config_to_contract(config_b)
        runtime_merged = contract_a.merge(contract_b)

        assert build_merged.fields is not None

        # Shared field: non-nullable in both → non-nullable
        build_shared = next(f for f in build_merged.fields if f.name == "shared")
        runtime_shared = next(f for f in runtime_merged.fields if f.normalized_name == "shared")
        assert build_shared.nullable == runtime_shared.nullable

        # Branch-exclusive field: forced nullable in both
        build_a_only = next(f for f in build_merged.fields if f.name == "a_only")
        runtime_a_only = next(f for f in runtime_merged.fields if f.normalized_name == "a_only")
        assert build_a_only.nullable is True, "Build: branch-exclusive forced nullable"
        assert runtime_a_only.nullable is True, "Runtime: branch-exclusive forced nullable"

    @given(configs=schema_configs_for_merge())
    @settings(max_examples=100)
    def test_fail_policy_nullable_consistency_property(self, configs: tuple[SchemaConfig, SchemaConfig]) -> None:
        """Property: With collision_policy='fail', build and runtime nullables match.

        The 'fail' policy uses OR semantics, which matches SchemaContract.merge().
        This is the only policy where both merge algorithms produce identical results.
        """
        config_a, config_b = configs

        # Build-time merge with 'fail' policy (OR semantics)
        build_merged = merge_union_fields(
            {"a": config_a, "b": config_b},
            require_all=False,
            collision_policy="fail",
        )

        # Runtime merge (always OR semantics)
        contract_a = schema_config_to_contract(config_a)
        contract_b = schema_config_to_contract(config_b)
        runtime_merged = contract_a.merge(contract_b)

        # Compare nullable for all fields
        assert build_merged.fields is not None
        build_nullable = {f.name: f.nullable for f in build_merged.fields}
        runtime_nullable = {f.normalized_name: f.nullable for f in runtime_merged.fields}

        assert build_nullable == runtime_nullable, (
            f"Nullable mismatch with 'fail' policy: build={build_nullable}, runtime={runtime_nullable}"
        )


class TestBuildRuntimeTypeConsistency:
    """Test that build-time and runtime merge produce the same field types."""

    @given(configs=schema_configs_for_merge())
    @settings(max_examples=100)
    def test_type_consistency_property(self, configs: tuple[SchemaConfig, SchemaConfig]) -> None:
        """Property: Build-time and runtime merges produce the same field types."""
        config_a, config_b = configs

        # Build-time merge
        build_merged = merge_union_fields({"a": config_a, "b": config_b}, require_all=False)

        # Runtime merge
        contract_a = schema_config_to_contract(config_a)
        contract_b = schema_config_to_contract(config_b)
        runtime_merged = contract_a.merge(contract_b)

        # Compare types
        type_map = {"int": int, "str": str, "float": float, "bool": bool, "any": object}
        assert build_merged.fields is not None
        build_types = {f.name: type_map[f.field_type] for f in build_merged.fields}
        runtime_types = {f.normalized_name: f.python_type for f in runtime_merged.fields}

        assert build_types == runtime_types


class TestBuildRuntimeFieldSetConsistency:
    """Test that build-time and runtime merge produce the same field sets."""

    @given(configs=schema_configs_for_merge())
    @settings(max_examples=100)
    def test_field_set_consistency_property(self, configs: tuple[SchemaConfig, SchemaConfig]) -> None:
        """Property: Build-time and runtime merges produce the same field set."""
        config_a, config_b = configs

        # Build-time merge
        build_merged = merge_union_fields({"a": config_a, "b": config_b}, require_all=False)

        # Runtime merge
        contract_a = schema_config_to_contract(config_a)
        contract_b = schema_config_to_contract(config_b)
        runtime_merged = contract_a.merge(contract_b)

        # Compare field sets
        assert build_merged.fields is not None
        build_fields = {f.name for f in build_merged.fields}
        runtime_fields = {f.normalized_name for f in runtime_merged.fields}

        assert build_fields == runtime_fields
