# tests/property/core/test_branch_transform_properties.py
"""Property-based tests for ARCH-15: per-branch transforms between fork and coalesce.

These tests verify fundamental invariants of the branch transform feature:
- Config normalization equivalence (list format ≡ identity dict)
- Token identity preservation through transform chains
- Branch first-node mapping completeness

Uses Hypothesis for randomized input generation. Property tests are
allowed to use make_graph_fork() for lightweight graph construction
(see tests/fixtures/factories.py tier rules).
"""

from __future__ import annotations

from typing import cast

from hypothesis import given, settings
from hypothesis import strategies as st

from elspeth.core.config import CoalesceSettings, GateSettings, SourceSettings
from elspeth.core.dag import ExecutionGraph
from elspeth.core.dag.models import WiredTransform
from elspeth.plugins.protocols import SinkProtocol, SourceProtocol
from elspeth.testing import make_pipeline_row, make_token_info
from tests.fixtures.factories import wire_transforms
from tests.fixtures.plugins import CollectSink, ListSource, PassTransform

# =============================================================================
# Strategies
# =============================================================================

# Valid branch names: must start with a letter, alphanumeric + underscores.
# Constrained to avoid validation failures from reserved names or special chars.
branch_name_strategy = st.from_regex(r"[a-z][a-z0-9_]{2,10}", fullmatch=True)

# Lists of unique branch names (min 2 for coalesce)
branch_name_lists = st.lists(
    branch_name_strategy,
    min_size=2,
    max_size=6,
    unique=True,
)


@st.composite
def branch_configs_with_transforms(draw: st.DrawFn) -> tuple[list[str], list[bool]]:
    """Generate valid branch names with random transform presence flags.

    Returns (branch_names, has_transform_flags) where each flag indicates
    whether the branch has transforms (True) or is identity-mapped (False).
    """
    num_branches = draw(st.integers(min_value=2, max_value=5))
    branch_names = [f"branch_{i}" for i in range(num_branches)]
    has_transforms = draw(st.lists(st.booleans(), min_size=num_branches, max_size=num_branches))
    return branch_names, has_transforms


# =============================================================================
# Config Normalization Properties
# =============================================================================


class TestConfigNormalization:
    """Property tests for CoalesceSettings branches normalization."""

    @given(branch_names=branch_name_lists)
    @settings(max_examples=100)
    def test_config_list_dict_equivalence(self, branch_names: list[str]) -> None:
        """Property: list format and identity dict format produce identical CoalesceSettings.

        branches: [a, b, c] must be equivalent to branches: {a: a, b: b, c: c}.
        This is the core ergonomics invariant of the ARCH-15 config change.
        """
        list_config = CoalesceSettings(
            name="test_merge",
            branches=branch_names,
            policy="require_all",
            merge="nested",
        )
        dict_config = CoalesceSettings(
            name="test_merge",
            branches={b: b for b in branch_names},
            policy="require_all",
            merge="nested",
        )

        assert list_config.branches == dict_config.branches, (
            f"List format {branch_names} produced {list_config.branches}, dict format produced {dict_config.branches}"
        )

    @given(branch_names=branch_name_lists)
    @settings(max_examples=100)
    def test_normalized_branches_are_dict(self, branch_names: list[str]) -> None:
        """Property: After normalization, branches is always a dict[str, str]."""
        config = CoalesceSettings(
            name="test_merge",
            branches=branch_names,
            policy="require_all",
            merge="nested",
        )

        assert isinstance(config.branches, dict)
        for key, value in config.branches.items():
            assert isinstance(key, str)
            assert isinstance(value, str)

    @given(branch_names=branch_name_lists)
    @settings(max_examples=100)
    def test_identity_dict_keys_equal_values(self, branch_names: list[str]) -> None:
        """Property: Identity-mapped branches have key == value for all entries."""
        config = CoalesceSettings(
            name="test_merge",
            branches=branch_names,
            policy="require_all",
            merge="nested",
        )

        for key, value in config.branches.items():
            assert key == value, f"Identity mapping violated: key={key!r}, value={value!r}"


# =============================================================================
# Token Identity Preservation Properties
# =============================================================================


class TestTokenBranchIdentity:
    """Property tests for branch_name preservation through transform operations."""

    @given(
        branch_name=branch_name_strategy,
        num_updates=st.integers(min_value=1, max_value=10),
        values=st.lists(st.integers(min_value=-1000, max_value=1000), min_size=1, max_size=10),
    )
    @settings(max_examples=100)
    def test_token_branch_identity_preserved_through_transforms(
        self,
        branch_name: str,
        num_updates: int,
        values: list[int],
    ) -> None:
        """Property: branch_name survives through N with_updated_data() calls.

        TokenInfo is a frozen dataclass; with_updated_data() creates a new
        instance with updated row_data but must preserve all lineage fields
        including branch_name. This invariant is critical for coalesce — the
        coalesce executor uses token.branch_name to identify which branch
        each arriving token belongs to.
        """
        token = make_token_info(branch_name=branch_name)

        for value in values[:num_updates]:
            new_data = make_pipeline_row({"value": value})
            token = token.with_updated_data(new_data)

            assert token.branch_name == branch_name, (
                f"branch_name changed from {branch_name!r} to {token.branch_name!r} after with_updated_data()"
            )

    @given(
        branch_name=branch_name_strategy,
        data_keys=st.lists(
            st.from_regex(r"[a-z][a-z0-9_]{0,8}", fullmatch=True),
            min_size=1,
            max_size=5,
            unique=True,
        ),
    )
    @settings(max_examples=100)
    def test_token_branch_name_independent_of_data_content(
        self,
        branch_name: str,
        data_keys: list[str],
    ) -> None:
        """Property: branch_name is independent of row data field names.

        Even if row data happens to contain a 'branch_name' field, the
        token's branch_name identity must come from the lineage metadata,
        not from row content.
        """
        row_data = {key: f"val_{i}" for i, key in enumerate(data_keys)}
        row_data["branch_name"] = "not_the_real_branch"

        token = make_token_info(branch_name=branch_name)
        token = token.with_updated_data(make_pipeline_row(row_data))

        assert token.branch_name == branch_name, "branch_name was corrupted by row data containing a 'branch_name' field"


# =============================================================================
# Branch First-Node Mapping Properties
# =============================================================================


class TestBranchFirstNodes:
    """Property tests for get_branch_first_nodes() completeness."""

    @given(config=branch_configs_with_transforms())
    @settings(max_examples=50, deadline=None)
    def test_branch_first_nodes_covers_all_branches(
        self,
        config: tuple[list[str], list[bool]],
    ) -> None:
        """Property: get_branch_first_nodes() returns an entry for every branch.

        For any valid fork/coalesce configuration (with or without per-branch
        transforms), the branch_first_node mapping must cover ALL branches.
        Identity branches map to the coalesce node; transform branches map
        to the first transform in the chain.
        """
        branch_names, has_transforms = config

        # Build coalesce config: transformed branches get unique output connections
        branches_dict: dict[str, str] = {}
        for name, has_transform in zip(branch_names, has_transforms, strict=True):
            if has_transform:
                branches_dict[name] = f"done_{name}"
            else:
                branches_dict[name] = name  # identity mapping

        coalesce = CoalesceSettings(
            name="test_merge",
            branches=branches_dict,
            policy="require_all",
            merge="nested",
            on_success="output",
        )
        gate = GateSettings(
            name="fork_gate",
            input="gate_in",
            condition="True",
            routes={"true": "fork", "false": "output"},
            fork_to=branch_names,
        )

        # Build transforms for branches that have them
        all_wired: list[WiredTransform] = []
        for name, has_transform in zip(branch_names, has_transforms, strict=True):
            if not has_transform:
                continue
            transform = PassTransform(name=f"transform_{name}")
            final_connection = branches_dict[name]
            branch_wired = wire_transforms(
                [transform],
                source_connection=name,
                final_sink=final_connection,
                names=[f"{name}_0"],
            )
            all_wired.extend(branch_wired)

        source = ListSource([], on_success="gate_in")
        source_settings = SourceSettings(plugin="list_source", on_success="gate_in", options={})
        sinks: dict[str, CollectSink] = {"output": CollectSink("output")}

        graph = ExecutionGraph.from_plugin_instances(
            source=cast("SourceProtocol", source),
            source_settings=source_settings,
            transforms=all_wired,
            sinks=cast("dict[str, SinkProtocol]", sinks),
            aggregations={},
            gates=[gate],
            coalesce_settings=[coalesce],
        )

        # THE PROPERTY: every branch in the config has a first-node entry
        result = graph.get_branch_first_nodes()
        assert set(result.keys()) == set(branch_names), (
            f"Branch first-node mapping missing branches. Expected: {sorted(branch_names)}, got: {sorted(result.keys())}"
        )

    @given(config=branch_configs_with_transforms())
    @settings(max_examples=50, deadline=None)
    def test_identity_branches_map_to_coalesce_node(
        self,
        config: tuple[list[str], list[bool]],
    ) -> None:
        """Property: Identity branches (no transforms) map to the coalesce node ID.

        When a branch has no transforms, the token goes directly to the
        coalesce node. The branch_first_node mapping must reflect this.
        """
        branch_names, has_transforms = config

        branches_dict: dict[str, str] = {}
        for name, has_transform in zip(branch_names, has_transforms, strict=True):
            if has_transform:
                branches_dict[name] = f"done_{name}"
            else:
                branches_dict[name] = name

        coalesce = CoalesceSettings(
            name="test_merge",
            branches=branches_dict,
            policy="require_all",
            merge="nested",
            on_success="output",
        )
        gate = GateSettings(
            name="fork_gate",
            input="gate_in",
            condition="True",
            routes={"true": "fork", "false": "output"},
            fork_to=branch_names,
        )

        all_wired: list[WiredTransform] = []
        for name, has_transform in zip(branch_names, has_transforms, strict=True):
            if not has_transform:
                continue
            transform = PassTransform(name=f"transform_{name}")
            branch_wired = wire_transforms(
                [transform],
                source_connection=name,
                final_sink=branches_dict[name],
                names=[f"{name}_0"],
            )
            all_wired.extend(branch_wired)

        source = ListSource([], on_success="gate_in")
        source_settings = SourceSettings(plugin="list_source", on_success="gate_in", options={})
        sinks: dict[str, CollectSink] = {"output": CollectSink("output")}

        graph = ExecutionGraph.from_plugin_instances(
            source=cast("SourceProtocol", source),
            source_settings=source_settings,
            transforms=all_wired,
            sinks=cast("dict[str, SinkProtocol]", sinks),
            aggregations={},
            gates=[gate],
            coalesce_settings=[coalesce],
        )

        result = graph.get_branch_first_nodes()
        coalesce_nid = graph.get_coalesce_id_map()["test_merge"]

        for name, has_transform in zip(branch_names, has_transforms, strict=True):
            if not has_transform:
                assert result[name] == coalesce_nid, (
                    f"Identity branch '{name}' should map to coalesce node '{coalesce_nid}', got '{result[name]}'"
                )

    @given(config=branch_configs_with_transforms())
    @settings(max_examples=50, deadline=None)
    def test_transform_branches_map_to_non_coalesce_node(
        self,
        config: tuple[list[str], list[bool]],
    ) -> None:
        """Property: Transform branches map to a node that is NOT the coalesce node.

        When a branch has transforms, the first-node should be the first
        transform in the chain, not the coalesce node itself.
        """
        branch_names, has_transforms = config

        # Skip if no branches have transforms (nothing to test)
        if not any(has_transforms):
            return

        branches_dict: dict[str, str] = {}
        for name, has_transform in zip(branch_names, has_transforms, strict=True):
            if has_transform:
                branches_dict[name] = f"done_{name}"
            else:
                branches_dict[name] = name

        coalesce = CoalesceSettings(
            name="test_merge",
            branches=branches_dict,
            policy="require_all",
            merge="nested",
            on_success="output",
        )
        gate = GateSettings(
            name="fork_gate",
            input="gate_in",
            condition="True",
            routes={"true": "fork", "false": "output"},
            fork_to=branch_names,
        )

        all_wired: list[WiredTransform] = []
        for name, has_transform in zip(branch_names, has_transforms, strict=True):
            if not has_transform:
                continue
            transform = PassTransform(name=f"transform_{name}")
            branch_wired = wire_transforms(
                [transform],
                source_connection=name,
                final_sink=branches_dict[name],
                names=[f"{name}_0"],
            )
            all_wired.extend(branch_wired)

        source = ListSource([], on_success="gate_in")
        source_settings = SourceSettings(plugin="list_source", on_success="gate_in", options={})
        sinks: dict[str, CollectSink] = {"output": CollectSink("output")}

        graph = ExecutionGraph.from_plugin_instances(
            source=cast("SourceProtocol", source),
            source_settings=source_settings,
            transforms=all_wired,
            sinks=cast("dict[str, SinkProtocol]", sinks),
            aggregations={},
            gates=[gate],
            coalesce_settings=[coalesce],
        )

        result = graph.get_branch_first_nodes()
        coalesce_nid = graph.get_coalesce_id_map()["test_merge"]

        for name, has_transform in zip(branch_names, has_transforms, strict=True):
            if has_transform:
                assert result[name] != coalesce_nid, (
                    f"Transform branch '{name}' should NOT map to coalesce node '{coalesce_nid}' — it should map to the first transform"
                )
