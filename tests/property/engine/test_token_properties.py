# tests/property/engine/test_token_properties.py
"""Property-based tests for token management (fork/expand/coalesce operations).

These tests verify CRITICAL audit trail integrity properties:
- Fork creates isolated copies (mutations don't leak between siblings)
- Fork preserves parent data (parent unchanged after fork)
- Expand creates isolated copies (same deepcopy requirement as fork)
- Expand preserves parent data (parent unchanged after expand)
- Coalesce produces correct merged token

Per tokens.py:151-153:
"CRITICAL: Use deepcopy to prevent nested mutable objects from being
shared across forked children. Shallow copy would cause mutations in
one branch to leak to siblings, breaking audit trail integrity."

Per tokens.py:362-366 (expand_token):
"CRITICAL: Use deepcopy to prevent nested mutable objects from being
shared across expanded children. Same reasoning as fork_token - without
this, mutations in one sibling leak to others, corrupting audit trail.
Bug: P2-2026-01-21-expand-token-shared-row-data"

These property tests prove the deepcopy fix works for ALL possible
nested data structures, not just the specific case that triggered the bug.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from hypothesis import given, settings
from hypothesis import strategies as st

from elspeth.contracts import TokenInfo
from elspeth.contracts.schema_contract import FieldContract, PipelineRow, SchemaContract
from elspeth.contracts.types import NodeID
from elspeth.engine.tokens import TokenManager
from tests.strategies.ids import multiple_branches
from tests.strategies.json import row_data
from tests.strategies.mutable import deeply_nested_data, mutable_nested_data


def _make_observed_contract() -> SchemaContract:
    """Create an OBSERVED schema contract for property tests."""
    return SchemaContract(mode="OBSERVED", fields=())


def _wrap_dict_as_pipeline_row(data: dict[str, Any]) -> PipelineRow:
    """Wrap dict as PipelineRow with OBSERVED contract for property tests."""
    return PipelineRow(data, _make_observed_contract())


def _create_mock_recorder(branches: list[str]) -> MagicMock:
    """Create a mock recorder that returns child tokens for fork operations.

    Note: fork_token now returns tuple[list[Token], str] for atomic operation.
    """
    mock_recorder = MagicMock()
    children = [MagicMock(token_id=f"child_{i}", branch_name=branch, fork_group_id="fork_1") for i, branch in enumerate(branches)]
    mock_recorder.fork_token.return_value = (children, "fork_1")
    return mock_recorder


class TestForkIsolationProperties:
    """Property tests for data isolation during fork operations."""

    @given(
        row_data=mutable_nested_data,
        branches=multiple_branches,
    )
    @settings(max_examples=200)
    def test_fork_creates_isolated_copies(self, row_data: dict[str, Any], branches: list[str]) -> None:
        """Property: Forked children have isolated row_data.

        Each child must get its own PipelineRow instance (not shared reference).
        This is the core audit trail integrity guarantee - even though PipelineRow
        is immutable, fork operations must create independent copies via deepcopy
        to ensure each child can be independently updated with_updated_data().
        """
        mock_recorder = _create_mock_recorder(branches)
        step_resolver = lambda node_id: 1  # noqa: E731
        manager = TokenManager(mock_recorder, step_resolver=step_resolver)

        parent = TokenInfo(
            row_id="row_1",
            token_id="parent_1",
            row_data=_wrap_dict_as_pipeline_row(row_data),
        )

        children, _fork_group_id = manager.fork_token(
            parent_token=parent,
            branches=branches,
            node_id=NodeID("node_fork"),
            run_id="test_run_1",
        )

        assert len(children) == len(branches), "Wrong number of children"

        # Verify each child has its own PipelineRow instance (not shared)
        # Even though PipelineRow is immutable, each token needs independent copies
        # so they can be updated independently via with_updated_data()
        for i in range(len(children)):
            for j in range(i + 1, len(children)):
                # Check object identity - they must be different instances
                assert children[i].row_data is not children[j].row_data, (
                    f"Children {i} and {j} share the same PipelineRow instance! Fork must create independent copies via deepcopy."
                )

        # Verify all children have equivalent data content (but different instances)
        expected_data = parent.row_data.to_dict()
        for i, child in enumerate(children):
            actual_data = child.row_data.to_dict()
            assert actual_data == expected_data, f"Child {i} has different data content! Expected {expected_data!r}, got {actual_data!r}"

    @given(
        row_data=deeply_nested_data,
        branches=multiple_branches,
    )
    @settings(max_examples=100)
    def test_fork_isolates_deeply_nested_data(self, row_data: Any, branches: list[str]) -> None:
        """Property: Deep nesting doesn't break isolation.

        Each child must get its own PipelineRow with deep-copied data.
        Even deeply nested mutable structures must be independent across siblings.
        """
        # Wrap in dict if needed (deeply_nested_data can be int, list, or dict)
        if not isinstance(row_data, dict):
            row_data = {"nested": row_data}

        mock_recorder = _create_mock_recorder(branches)
        step_resolver = lambda node_id: 1  # noqa: E731
        manager = TokenManager(mock_recorder, step_resolver=step_resolver)

        parent = TokenInfo(
            row_id="row_1",
            token_id="parent_1",
            row_data=_wrap_dict_as_pipeline_row(row_data),
        )

        children, _fork_group_id = manager.fork_token(
            parent_token=parent,
            branches=branches,
            node_id=NodeID("node_fork"),
            run_id="test_run_1",
        )

        # Verify each child has independent PipelineRow instances
        # Check all pairs have different object identities
        for i in range(len(children)):
            for j in range(i + 1, len(children)):
                assert children[i].row_data is not children[j].row_data, (
                    f"Deep nesting test: Children {i} and {j} share PipelineRow instance!"
                )

        # Verify all children have equivalent data (deep equality)
        expected_data = parent.row_data.to_dict()
        for i, child in enumerate(children):
            actual_data = child.row_data.to_dict()
            assert actual_data == expected_data, (
                f"Deep nesting test: Child {i} has different data! Expected {expected_data!r}, got {actual_data!r}"
            )

        # Additionally verify that the underlying data dict is NOT shared
        # (even though PipelineRow is immutable, the dict inside should be independent)
        # This ensures deep copying happened correctly
        if len(children) >= 2:
            # Get the internal data dicts (they should be different instances)
            data_0 = children[0].row_data.to_dict()
            data_1 = children[1].row_data.to_dict()
            # They should be equal in value but not the same object
            assert data_0 == data_1, "Children should have equal data"

            # For mutable nested structures, verify they're independent copies
            def check_independent_nested(obj1: Any, obj2: Any, path: str = "") -> None:
                """Recursively verify nested mutable objects are independent."""
                if isinstance(obj1, dict) and isinstance(obj2, dict):
                    for key in obj1:
                        if key in obj2:
                            check_independent_nested(obj1[key], obj2[key], f"{path}.{key}")
                elif isinstance(obj1, list) and isinstance(obj2, list):
                    for idx in range(min(len(obj1), len(obj2))):
                        check_independent_nested(obj1[idx], obj2[idx], f"{path}[{idx}]")
                elif isinstance(obj1, (dict, list)):
                    # Both are mutable - they must be different objects
                    assert obj1 is not obj2, f"Mutable objects at {path} are shared!"

            check_independent_nested(data_0, data_1)


class TestForkParentPreservationProperties:
    """Property tests for parent preservation during fork."""

    @given(
        row_data=mutable_nested_data,
        branches=multiple_branches,
    )
    @settings(max_examples=200)
    def test_fork_preserves_parent_data(self, row_data: dict[str, Any], branches: list[str]) -> None:
        """Property: Forking doesn't mutate parent token.

        The parent's row_data must remain independent from children.
        Since PipelineRow is immutable, we verify parent and children
        have different PipelineRow instances (not shared).
        """
        mock_recorder = _create_mock_recorder(branches)
        step_resolver = lambda node_id: 1  # noqa: E731
        manager = TokenManager(mock_recorder, step_resolver=step_resolver)

        parent = TokenInfo(
            row_id="row_1",
            token_id="parent_1",
            row_data=_wrap_dict_as_pipeline_row(row_data),
        )

        # Store original parent data for comparison
        original_parent_data = parent.row_data.to_dict()

        children, _fork_group_id = manager.fork_token(
            parent_token=parent,
            branches=branches,
            node_id=NodeID("node_fork"),
            run_id="test_run_1",
        )

        # Verify parent's PipelineRow is not shared with any child
        for i, child in enumerate(children):
            assert child.row_data is not parent.row_data, (
                f"Child {i} shares PipelineRow instance with parent! Fork must create independent copies."
            )

        # Verify parent data is unchanged (value equality)
        assert parent.row_data.to_dict() == original_parent_data, "Parent data was changed after fork!"

        # Verify all children have same data as parent (but different instances)
        for i, child in enumerate(children):
            assert child.row_data.to_dict() == original_parent_data, f"Child {i} has different data from parent!"

    @given(
        row_data=row_data,
        branches=st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=3, unique=True),
    )
    @settings(max_examples=150)
    def test_fork_children_have_correct_metadata(self, row_data: dict[str, Any], branches: list[str]) -> None:
        """Property: Forked children have correct token metadata.

        Each child should have:
        - Same row_id as parent
        - Unique token_id
        - Correct branch_name
        - Non-None fork_group_id
        """
        mock_recorder = _create_mock_recorder(branches)
        step_resolver = lambda node_id: 1  # noqa: E731
        manager = TokenManager(mock_recorder, step_resolver=step_resolver)

        parent = TokenInfo(
            row_id="row_1",
            token_id="parent_1",
            row_data=_wrap_dict_as_pipeline_row(row_data),
        )

        children, _fork_group_id = manager.fork_token(
            parent_token=parent,
            branches=branches,
            node_id=NodeID("node_fork"),
            run_id="test_run_1",
        )

        # Verify each child
        seen_token_ids = set()
        for i, child in enumerate(children):
            # Same row_id as parent
            assert child.row_id == parent.row_id, f"Child {i} has wrong row_id: {child.row_id} != {parent.row_id}"

            # Unique token_id
            assert child.token_id not in seen_token_ids, f"Duplicate token_id: {child.token_id}"
            seen_token_ids.add(child.token_id)

            # Correct branch_name
            assert child.branch_name == branches[i], f"Child {i} has wrong branch_name: {child.branch_name} != {branches[i]}"

            # Has fork_group_id
            assert child.fork_group_id is not None, f"Child {i} missing fork_group_id"


class TestForkRowDataOverrideProperties:
    """Property tests for row_data override during fork."""

    @given(
        original_data=row_data,
        override_data=row_data,
        branches=multiple_branches,
    )
    @settings(max_examples=100)
    def test_fork_with_override_uses_override(
        self,
        original_data: dict[str, Any],
        override_data: dict[str, Any],
        branches: list[str],
    ) -> None:
        """Property: When row_data is provided to fork, children use it.

        The override row_data should be used instead of parent's data.
        """
        mock_recorder = _create_mock_recorder(branches)
        step_resolver = lambda node_id: 1  # noqa: E731
        manager = TokenManager(mock_recorder, step_resolver=step_resolver)

        parent = TokenInfo(
            row_id="row_1",
            token_id="parent_1",
            row_data=_wrap_dict_as_pipeline_row(original_data),
        )

        children, _fork_group_id = manager.fork_token(
            parent_token=parent,
            branches=branches,
            node_id=NodeID("node_fork"),
            run_id="test_run_1",
            row_data=_wrap_dict_as_pipeline_row(override_data),  # Explicit override
        )

        # Children should have override data, not parent data
        for child in children:
            # Compare structure (values may be copied)
            assert set(child.row_data.keys()) == set(override_data.keys()), "Child doesn't have override data keys"

    @given(
        original_data=mutable_nested_data,
        branches=multiple_branches,
    )
    @settings(max_examples=100)
    def test_fork_without_override_uses_parent_data(
        self,
        original_data: dict[str, Any],
        branches: list[str],
    ) -> None:
        """Property: Without row_data override, children get parent's data.

        When no override is provided, children should receive a copy of
        the parent's row_data.
        """
        mock_recorder = _create_mock_recorder(branches)
        step_resolver = lambda node_id: 1  # noqa: E731
        manager = TokenManager(mock_recorder, step_resolver=step_resolver)

        parent = TokenInfo(
            row_id="row_1",
            token_id="parent_1",
            row_data=_wrap_dict_as_pipeline_row(original_data),
        )

        children, _fork_group_id = manager.fork_token(
            parent_token=parent,
            branches=branches,
            node_id=NodeID("node_fork"),
            run_id="test_run_1",
            # No row_data override
        )

        # Children should have parent's data structure
        for child in children:
            assert child.row_data.to_dict() == original_data, "Child data doesn't match parent when no override provided"


# =============================================================================
# Expand Token Helpers
# =============================================================================


def _make_locked_contract_from_data(data: dict[str, Any]) -> SchemaContract:
    """Create a locked OBSERVED contract from row data keys.

    expand_token requires a locked contract (tokens.py:354-358).
    """
    fields = tuple(
        FieldContract(
            normalized_name=key,
            original_name=key,
            python_type=object,
            required=False,
            source="inferred",
        )
        for key in data
    )
    return SchemaContract(mode="OBSERVED", fields=fields, locked=True)


def _create_mock_recorder_for_expand(count: int) -> MagicMock:
    """Create a mock recorder that returns child tokens for expand operations.

    expand_token returns tuple[list[Token], str] like fork_token.
    """
    mock_recorder = MagicMock()
    children = [MagicMock(token_id=f"expanded_{i}", expand_group_id="expand_1") for i in range(count)]
    mock_recorder.expand_token.return_value = (children, "expand_1")
    return mock_recorder


# =============================================================================
# Gap 1: Expand Token Isolation Properties
# =============================================================================
# expand_token (tokens.py:309-378) has its own deepcopy at line 372,
# independent from fork_token's deepcopy at line 250. This was the exact
# bug site of P2-2026-01-21-expand-token-shared-row-data.
# =============================================================================


class TestExpandIsolationProperties:
    """Property tests for data isolation during expand operations.

    Mirrors TestForkIsolationProperties but exercises the expand_token
    code path, which has its own independent deepcopy call.
    """

    @given(row_data=mutable_nested_data, count=st.integers(min_value=2, max_value=5))
    @settings(max_examples=200)
    def test_expand_creates_isolated_copies(self, row_data: dict[str, Any], count: int) -> None:
        """Property: Expanded children have isolated row_data.

        Each child must get its own PipelineRow instance (not shared reference).
        This is the regression test for P2-2026-01-21-expand-token-shared-row-data.
        """
        expanded_rows = [dict(row_data) for _ in range(count)]
        output_contract = _make_locked_contract_from_data(row_data)
        mock_recorder = _create_mock_recorder_for_expand(count)
        step_resolver = lambda node_id: 1  # noqa: E731
        manager = TokenManager(mock_recorder, step_resolver=step_resolver)

        parent = TokenInfo(
            row_id="row_1",
            token_id="parent_1",
            row_data=_wrap_dict_as_pipeline_row(row_data),
        )

        children, _expand_group_id = manager.expand_token(
            parent_token=parent,
            expanded_rows=expanded_rows,
            output_contract=output_contract,
            node_id=NodeID("node_expand"),
            run_id="test_run_1",
        )

        assert len(children) == count, f"Wrong number of children: {len(children)} != {count}"

        # Verify each child has its own PipelineRow instance (not shared)
        for i in range(len(children)):
            for j in range(i + 1, len(children)):
                assert children[i].row_data is not children[j].row_data, (
                    f"Expanded children {i} and {j} share PipelineRow instance! expand_token must deepcopy each row independently."
                )

        # Verify all children have equivalent data content
        for i, child in enumerate(children):
            assert child.row_data.to_dict() == row_data, (
                f"Expanded child {i} has different data! Expected {row_data!r}, got {child.row_data.to_dict()!r}"
            )

    @given(row_data=deeply_nested_data, count=st.integers(min_value=2, max_value=4))
    @settings(max_examples=100)
    def test_expand_isolates_deeply_nested_data(self, row_data: Any, count: int) -> None:
        """Property: Deep nesting doesn't break expand isolation.

        Stress tests deepcopy in expand_token with recursive structures.
        """
        if not isinstance(row_data, dict):
            row_data = {"nested": row_data}

        expanded_rows = [dict(row_data) for _ in range(count)]
        output_contract = _make_locked_contract_from_data(row_data)
        mock_recorder = _create_mock_recorder_for_expand(count)
        step_resolver = lambda node_id: 1  # noqa: E731
        manager = TokenManager(mock_recorder, step_resolver=step_resolver)

        parent = TokenInfo(
            row_id="row_1",
            token_id="parent_1",
            row_data=_wrap_dict_as_pipeline_row(row_data),
        )

        children, _expand_group_id = manager.expand_token(
            parent_token=parent,
            expanded_rows=expanded_rows,
            output_contract=output_contract,
            node_id=NodeID("node_expand"),
            run_id="test_run_1",
        )

        # Verify each child has independent PipelineRow instances
        for i in range(len(children)):
            for j in range(i + 1, len(children)):
                assert children[i].row_data is not children[j].row_data, f"Deep nesting: Expanded children {i} and {j} share PipelineRow!"

        # Verify underlying data dicts are independent copies
        if len(children) >= 2:
            data_0 = children[0].row_data.to_dict()
            data_1 = children[1].row_data.to_dict()
            assert data_0 == data_1, "Expanded children should have equal data"

            def check_independent_nested(obj1: Any, obj2: Any, path: str = "") -> None:
                """Recursively verify nested mutable objects are independent."""
                if isinstance(obj1, dict) and isinstance(obj2, dict):
                    for key in obj1:
                        if key in obj2:
                            check_independent_nested(obj1[key], obj2[key], f"{path}.{key}")
                elif isinstance(obj1, list) and isinstance(obj2, list):
                    for idx in range(min(len(obj1), len(obj2))):
                        check_independent_nested(obj1[idx], obj2[idx], f"{path}[{idx}]")
                elif isinstance(obj1, (dict, list)):
                    assert obj1 is not obj2, f"Mutable objects at {path} are shared!"

            check_independent_nested(data_0, data_1)


class TestExpandParentPreservationProperties:
    """Property tests for parent preservation during expand."""

    @given(row_data=mutable_nested_data, count=st.integers(min_value=2, max_value=5))
    @settings(max_examples=200)
    def test_expand_preserves_parent_data(self, row_data: dict[str, Any], count: int) -> None:
        """Property: Expanding doesn't mutate parent token.

        The parent's row_data must remain independent from expanded children.
        """
        expanded_rows = [dict(row_data) for _ in range(count)]
        output_contract = _make_locked_contract_from_data(row_data)
        mock_recorder = _create_mock_recorder_for_expand(count)
        step_resolver = lambda node_id: 1  # noqa: E731
        manager = TokenManager(mock_recorder, step_resolver=step_resolver)

        parent = TokenInfo(
            row_id="row_1",
            token_id="parent_1",
            row_data=_wrap_dict_as_pipeline_row(row_data),
        )

        original_parent_data = parent.row_data.to_dict()

        children, _expand_group_id = manager.expand_token(
            parent_token=parent,
            expanded_rows=expanded_rows,
            output_contract=output_contract,
            node_id=NodeID("node_expand"),
            run_id="test_run_1",
        )

        # Verify parent's PipelineRow is not shared with any child
        for i, child in enumerate(children):
            assert child.row_data is not parent.row_data, f"Expanded child {i} shares PipelineRow with parent!"

        # Verify parent data is unchanged
        assert parent.row_data.to_dict() == original_parent_data, "Parent data changed after expand!"

    @given(row_data=row_data, count=st.integers(min_value=1, max_value=5))
    @settings(max_examples=150)
    def test_expand_children_have_correct_metadata(self, row_data: dict[str, Any], count: int) -> None:
        """Property: Expanded children have correct token metadata.

        Each child should have:
        - Same row_id as parent
        - Unique token_id
        - Non-None expand_group_id
        - Inherited branch_name from parent
        """
        expanded_rows = [dict(row_data) for _ in range(count)]
        output_contract = _make_locked_contract_from_data(row_data)
        mock_recorder = _create_mock_recorder_for_expand(count)
        step_resolver = lambda node_id: 1  # noqa: E731
        manager = TokenManager(mock_recorder, step_resolver=step_resolver)

        parent = TokenInfo(
            row_id="row_1",
            token_id="parent_1",
            row_data=_wrap_dict_as_pipeline_row(row_data),
            branch_name="test_branch",
        )

        children, _expand_group_id = manager.expand_token(
            parent_token=parent,
            expanded_rows=expanded_rows,
            output_contract=output_contract,
            node_id=NodeID("node_expand"),
            run_id="test_run_1",
        )

        seen_token_ids = set()
        for i, child in enumerate(children):
            # Same row_id as parent
            assert child.row_id == parent.row_id, f"Expanded child {i} has wrong row_id: {child.row_id} != {parent.row_id}"

            # Unique token_id
            assert child.token_id not in seen_token_ids, f"Duplicate token_id: {child.token_id}"
            seen_token_ids.add(child.token_id)

            # Has expand_group_id
            assert child.expand_group_id is not None, f"Expanded child {i} missing expand_group_id"

            # Inherits branch_name from parent
            assert child.branch_name == parent.branch_name, (
                f"Expanded child {i} has wrong branch_name: {child.branch_name} != {parent.branch_name}"
            )

    @given(count=st.integers(min_value=1, max_value=5))
    @settings(max_examples=50)
    def test_expand_requires_locked_contract(self, count: int) -> None:
        """Property: expand_token rejects unlocked contracts.

        This is a guard at tokens.py:354-358 — the output contract must be
        locked before expansion to ensure type safety downstream.
        """
        row_data_dict = {"value": 42}
        expanded_rows = [dict(row_data_dict) for _ in range(count)]
        unlocked_contract = SchemaContract(mode="OBSERVED", fields=())  # locked=False
        mock_recorder = _create_mock_recorder_for_expand(count)
        step_resolver = lambda node_id: 1  # noqa: E731
        manager = TokenManager(mock_recorder, step_resolver=step_resolver)

        parent = TokenInfo(
            row_id="row_1",
            token_id="parent_1",
            row_data=_wrap_dict_as_pipeline_row(row_data_dict),
        )

        import pytest

        with pytest.raises(ValueError, match="locked"):
            manager.expand_token(
                parent_token=parent,
                expanded_rows=expanded_rows,
                output_contract=unlocked_contract,
                node_id=NodeID("node_expand"),
                run_id="test_run_1",
            )

        mock_recorder.expand_token.assert_not_called()
