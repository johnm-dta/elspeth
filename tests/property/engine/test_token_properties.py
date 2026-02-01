# tests/property/engine/test_token_properties.py
"""Property-based tests for token management (fork/coalesce operations).

These tests verify CRITICAL audit trail integrity properties:
- Fork creates isolated copies (mutations don't leak between siblings)
- Fork preserves parent data (parent unchanged after fork)
- Coalesce produces correct merged token

Per tokens.py:151-153:
"CRITICAL: Use deepcopy to prevent nested mutable objects from being
shared across forked children. Shallow copy would cause mutations in
one branch to leak to siblings, breaking audit trail integrity."

These property tests prove the deepcopy fix works for ALL possible
nested data structures, not just the specific case that triggered the bug.
"""

from __future__ import annotations

import copy
from typing import Any
from unittest.mock import MagicMock

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from elspeth.contracts import TokenInfo
from elspeth.engine.tokens import TokenManager
from tests.property.conftest import (
    deeply_nested_data,
    multiple_branches,
    mutable_nested_data,
    row_data,
)


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

        Mutating one child's row_data must NOT affect any sibling's data.
        This is the core audit trail integrity guarantee.
        """
        mock_recorder = _create_mock_recorder(branches)
        manager = TokenManager(mock_recorder)

        parent = TokenInfo(
            row_id="row_1",
            token_id="parent_1",
            row_data=row_data,
        )

        children, _fork_group_id = manager.fork_token(
            parent_token=parent,
            branches=branches,
            step_in_pipeline=1,
            run_id="test_run_1",
        )

        assert len(children) == len(branches), "Wrong number of children"

        # Get first key for mutation test
        if not children[0].row_data:
            assume(False)
            return

        first_key = next(iter(children[0].row_data))
        original_values = [copy.deepcopy(c.row_data.get(first_key)) for c in children]

        # Mutate first child's data (testing various mutation types)
        if isinstance(children[0].row_data[first_key], list):
            children[0].row_data[first_key].append(999999)
        elif isinstance(children[0].row_data[first_key], dict):
            children[0].row_data[first_key]["__mutated__"] = True
        else:
            children[0].row_data[first_key] = "__MUTATED__"

        # Verify ALL siblings are unaffected
        for i, child in enumerate(children[1:], start=1):
            actual_value = child.row_data.get(first_key)
            expected_value = original_values[i]
            assert actual_value == expected_value, (
                f"Child {i} was affected by mutation to child 0! Expected {expected_value!r}, got {actual_value!r}"
            )

    @given(
        row_data=deeply_nested_data,
        branches=multiple_branches,
    )
    @settings(max_examples=100)
    def test_fork_isolates_deeply_nested_data(self, row_data: Any, branches: list[str]) -> None:
        """Property: Deep nesting doesn't break isolation.

        Shallow copy only copies top-level references. This test ensures
        even deeply nested structures are properly isolated.
        """
        # Wrap in dict if needed (deeply_nested_data can be int, list, or dict)
        if not isinstance(row_data, dict):
            row_data = {"nested": row_data}

        mock_recorder = _create_mock_recorder(branches)
        manager = TokenManager(mock_recorder)

        parent = TokenInfo(
            row_id="row_1",
            token_id="parent_1",
            row_data=row_data,
        )

        children, _fork_group_id = manager.fork_token(
            parent_token=parent,
            branches=branches,
            step_in_pipeline=1,
            run_id="test_run_1",
        )

        # Store original state of all children
        original_states = [copy.deepcopy(c.row_data) for c in children]

        # Recursively mutate first child's data
        def mutate_deeply(obj: Any, depth: int = 0) -> bool:
            """Mutate the deepest mutable object found."""
            if depth > 10:
                return False

            if isinstance(obj, dict):
                for key, value in obj.items():
                    if isinstance(value, (dict, list)):
                        if mutate_deeply(value, depth + 1):
                            return True
                    else:
                        obj[key] = "__DEEP_MUTATED__"
                        return True
            elif isinstance(obj, list) and obj:
                if isinstance(obj[0], (dict, list)):
                    return mutate_deeply(obj[0], depth + 1)
                else:
                    obj[0] = "__DEEP_MUTATED__"
                    return True
            return False

        mutate_deeply(children[0].row_data)

        # Verify siblings unchanged
        for i, child in enumerate(children[1:], start=1):
            assert child.row_data == original_states[i], f"Child {i} was affected by deep mutation to child 0!"


class TestForkParentPreservationProperties:
    """Property tests for parent preservation during fork."""

    @given(
        row_data=mutable_nested_data,
        branches=multiple_branches,
    )
    @settings(max_examples=200)
    def test_fork_preserves_parent_data(self, row_data: dict[str, Any], branches: list[str]) -> None:
        """Property: Forking doesn't mutate parent token.

        The parent's row_data must be unchanged after fork,
        regardless of what happens to children.
        """
        mock_recorder = _create_mock_recorder(branches)
        manager = TokenManager(mock_recorder)

        parent = TokenInfo(
            row_id="row_1",
            token_id="parent_1",
            row_data=row_data,
        )

        # Deep copy parent data for comparison
        original_parent_data = copy.deepcopy(parent.row_data)

        children, _fork_group_id = manager.fork_token(
            parent_token=parent,
            branches=branches,
            step_in_pipeline=1,
            run_id="test_run_1",
        )

        # Mutate all children
        for child in children:
            if child.row_data:
                for key in list(child.row_data.keys()):
                    child.row_data[key] = "__CHILD_MUTATED__"

        # Parent must be unchanged
        assert parent.row_data == original_parent_data, "Parent data was mutated by child operations!"

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
        manager = TokenManager(mock_recorder)

        parent = TokenInfo(
            row_id="row_1",
            token_id="parent_1",
            row_data=row_data,
        )

        children, _fork_group_id = manager.fork_token(
            parent_token=parent,
            branches=branches,
            step_in_pipeline=1,
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
        manager = TokenManager(mock_recorder)

        parent = TokenInfo(
            row_id="row_1",
            token_id="parent_1",
            row_data=original_data,
        )

        children, _fork_group_id = manager.fork_token(
            parent_token=parent,
            branches=branches,
            step_in_pipeline=1,
            run_id="test_run_1",
            row_data=override_data,  # Explicit override
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
        manager = TokenManager(mock_recorder)

        parent = TokenInfo(
            row_id="row_1",
            token_id="parent_1",
            row_data=original_data,
        )

        children, _fork_group_id = manager.fork_token(
            parent_token=parent,
            branches=branches,
            step_in_pipeline=1,
            run_id="test_run_1",
            # No row_data override
        )

        # Children should have parent's data structure
        for child in children:
            assert child.row_data == original_data, "Child data doesn't match parent when no override provided"
