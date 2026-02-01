# tests/property/engine/test_coalesce_properties.py
"""Property-based tests for CoalesceExecutor merge policies and invariants.

These tests verify the fundamental properties of ELSPETH's coalesce system:

Merge Policy Properties:
- require_all: Merges only when ALL branches arrive
- first: Merges immediately on first arrival
- quorum: Merges when at least quorum_count branches arrive
- best_effort: Merges on timeout with whatever arrived

Memory Properties:
- Completed keys bounded by _max_completed_keys (FIFO eviction)
- Late arrivals after merge return consistent failure

Data Merge Properties:
- union: Combined fields from all branches (later overrides)
- nested: Each branch as nested object with correct hierarchy
- select: Only selected branch's data

The coalesce system is audit-critical - incorrect merging would orphan tokens
or produce incorrect audit trails.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from elspeth.contracts import TokenInfo
from elspeth.core.config import CoalesceSettings
from elspeth.engine.clock import MockClock
from elspeth.engine.coalesce_executor import CoalesceExecutor
from tests.property.conftest import row_data

# =============================================================================
# Strategies for generating coalesce configurations
# =============================================================================

# Branch names (simple alphanumeric)
branch_names = st.text(
    min_size=1,
    max_size=15,
    alphabet="abcdefghijklmnopqrstuvwxyz_",
).filter(lambda s: s[0].isalpha())


# Generate unique branch lists
@st.composite
def branch_lists(draw: st.DrawFn, min_size: int = 2, max_size: int = 5) -> list[str]:
    """Generate a list of unique branch names."""
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    branches = []
    for i in range(size):
        branches.append(f"branch_{i}")
    return branches


def make_token(
    token_id: str,
    row_id: str,
    branch_name: str,
    row_data: dict[str, Any],
) -> TokenInfo:
    """Create a TokenInfo for testing."""
    return TokenInfo(
        token_id=token_id,
        row_id=row_id,
        row_data=row_data,
        branch_name=branch_name,
    )


def make_mock_executor(clock: MockClock | None = None) -> CoalesceExecutor:
    """Create a CoalesceExecutor with mocked dependencies."""
    mock_recorder = MagicMock()
    mock_recorder.begin_node_state.return_value = MagicMock(state_id="state-001")
    mock_recorder.complete_node_state.return_value = None
    mock_recorder.record_token_outcome.return_value = None

    mock_span_factory = MagicMock()
    mock_token_manager = MagicMock()

    # Make coalesce_tokens return a merged token
    def mock_coalesce_tokens(parents, merged_data, step_in_pipeline):
        return TokenInfo(
            token_id=f"merged-{parents[0].row_id}",
            row_id=parents[0].row_id,
            row_data=merged_data,
            join_group_id=f"join-{parents[0].row_id}",
        )

    mock_token_manager.coalesce_tokens.side_effect = mock_coalesce_tokens

    return CoalesceExecutor(
        recorder=mock_recorder,
        span_factory=mock_span_factory,
        token_manager=mock_token_manager,
        run_id="test-run",
        clock=clock or MockClock(start=0.0),
    )


# =============================================================================
# Merge Policy Property Tests
# =============================================================================


class TestRequireAllPolicyProperties:
    """Property tests for require_all merge policy."""

    @given(branches=branch_lists(min_size=2, max_size=5))
    @settings(max_examples=50)
    def test_require_all_holds_until_all_arrive(self, branches: list[str]) -> None:
        """Property: require_all holds tokens until ALL branches arrive."""
        executor = make_mock_executor()
        settings = CoalesceSettings(
            name="test_coalesce",
            branches=branches,
            policy="require_all",
            merge="union",
        )
        executor.register_coalesce(settings, node_id="node-001")

        row_id = "row-001"

        # Send all but one branch
        for i, branch in enumerate(branches[:-1]):
            token = make_token(
                token_id=f"token-{i}",
                row_id=row_id,
                branch_name=branch,
                row_data={"field": i},
            )
            outcome = executor.accept(token, "test_coalesce", step_in_pipeline=0)
            assert outcome.held is True, f"Should hold after {i + 1}/{len(branches)} branches"
            assert outcome.merged_token is None

        # Send final branch - should merge
        final_token = make_token(
            token_id=f"token-{len(branches) - 1}",
            row_id=row_id,
            branch_name=branches[-1],
            row_data={"field": len(branches) - 1},
        )
        outcome = executor.accept(final_token, "test_coalesce", step_in_pipeline=0)

        assert outcome.held is False, "Should merge when all branches arrive"
        assert outcome.merged_token is not None
        assert len(outcome.consumed_tokens) == len(branches)

    @given(branches=branch_lists(min_size=3, max_size=5), missing_count=st.integers(min_value=1, max_value=2))
    @settings(max_examples=30)
    def test_require_all_never_partial_merge(self, branches: list[str], missing_count: int) -> None:
        """Property: require_all NEVER does partial merge, even on flush."""
        assume(missing_count < len(branches))

        executor = make_mock_executor()
        settings = CoalesceSettings(
            name="test_coalesce",
            branches=branches,
            policy="require_all",
            merge="union",
        )
        executor.register_coalesce(settings, node_id="node-001")

        row_id = "row-001"
        arriving_branches = branches[:-missing_count]

        # Send partial branches
        for i, branch in enumerate(arriving_branches):
            token = make_token(
                token_id=f"token-{i}",
                row_id=row_id,
                branch_name=branch,
                row_data={"field": i},
            )
            executor.accept(token, "test_coalesce", step_in_pipeline=0)

        # Flush pending - should fail, not merge
        outcomes = executor.flush_pending(step_map={"test_coalesce": 0})

        assert len(outcomes) == 1
        outcome = outcomes[0]
        assert outcome.failure_reason == "incomplete_branches"
        assert outcome.merged_token is None
        assert len(outcome.consumed_tokens) == len(arriving_branches)


class TestFirstPolicyProperties:
    """Property tests for first merge policy."""

    @given(branches=branch_lists(min_size=2, max_size=5), first_branch_idx=st.integers(min_value=0, max_value=4))
    @settings(max_examples=50)
    def test_first_merges_immediately(self, branches: list[str], first_branch_idx: int) -> None:
        """Property: first policy merges immediately on first arrival."""
        first_branch_idx = first_branch_idx % len(branches)

        executor = make_mock_executor()
        settings = CoalesceSettings(
            name="test_coalesce",
            branches=branches,
            policy="first",
            merge="union",
        )
        executor.register_coalesce(settings, node_id="node-001")

        # Send just one token
        token = make_token(
            token_id="token-0",
            row_id="row-001",
            branch_name=branches[first_branch_idx],
            row_data={"value": 42},
        )
        outcome = executor.accept(token, "test_coalesce", step_in_pipeline=0)

        assert outcome.held is False, "first policy should merge immediately"
        assert outcome.merged_token is not None
        assert len(outcome.consumed_tokens) == 1


class TestQuorumPolicyProperties:
    """Property tests for quorum merge policy."""

    @given(
        branch_count=st.integers(min_value=3, max_value=6),
        quorum_count=st.integers(min_value=2, max_value=5),
    )
    @settings(max_examples=50)
    def test_quorum_merges_at_exact_threshold(self, branch_count: int, quorum_count: int) -> None:
        """Property: quorum merges exactly when quorum_count branches arrive."""
        assume(quorum_count <= branch_count)

        branches = [f"branch_{i}" for i in range(branch_count)]
        executor = make_mock_executor()
        settings = CoalesceSettings(
            name="test_coalesce",
            branches=branches,
            policy="quorum",
            quorum_count=quorum_count,
            merge="union",
        )
        executor.register_coalesce(settings, node_id="node-001")

        row_id = "row-001"

        # Send branches up to quorum - 1 (should all hold)
        for i in range(quorum_count - 1):
            token = make_token(
                token_id=f"token-{i}",
                row_id=row_id,
                branch_name=branches[i],
                row_data={"field": i},
            )
            outcome = executor.accept(token, "test_coalesce", step_in_pipeline=0)
            assert outcome.held is True, f"Should hold at {i + 1} branches (quorum={quorum_count})"

        # Send quorum-th branch - should merge
        quorum_token = make_token(
            token_id=f"token-{quorum_count - 1}",
            row_id=row_id,
            branch_name=branches[quorum_count - 1],
            row_data={"field": quorum_count - 1},
        )
        outcome = executor.accept(quorum_token, "test_coalesce", step_in_pipeline=0)

        assert outcome.held is False, "Should merge when quorum is met"
        assert outcome.merged_token is not None
        assert len(outcome.consumed_tokens) == quorum_count

    @given(
        branch_count=st.integers(min_value=3, max_value=5),
        quorum_count=st.integers(min_value=2, max_value=4),
    )
    @settings(max_examples=30)
    def test_quorum_flush_fails_below_threshold(self, branch_count: int, quorum_count: int) -> None:
        """Property: quorum flush fails if quorum not met."""
        assume(quorum_count <= branch_count)
        assume(quorum_count > 1)  # Need at least 2 for meaningful test

        branches = [f"branch_{i}" for i in range(branch_count)]
        executor = make_mock_executor()
        settings = CoalesceSettings(
            name="test_coalesce",
            branches=branches,
            policy="quorum",
            quorum_count=quorum_count,
            merge="union",
        )
        executor.register_coalesce(settings, node_id="node-001")

        row_id = "row-001"

        # Send fewer than quorum
        arriving_count = quorum_count - 1
        for i in range(arriving_count):
            token = make_token(
                token_id=f"token-{i}",
                row_id=row_id,
                branch_name=branches[i],
                row_data={"field": i},
            )
            executor.accept(token, "test_coalesce", step_in_pipeline=0)

        # Flush - should fail
        outcomes = executor.flush_pending(step_map={"test_coalesce": 0})

        assert len(outcomes) == 1
        assert outcomes[0].failure_reason == "quorum_not_met"
        assert outcomes[0].merged_token is None


class TestBestEffortPolicyProperties:
    """Property tests for best_effort merge policy."""

    @given(branches=branch_lists(min_size=3, max_size=5))
    @settings(max_examples=30)
    def test_best_effort_merges_on_timeout(self, branches: list[str]) -> None:
        """Property: best_effort merges whatever arrived when timeout expires."""
        # Only send partial branches (not all), so timeout is the trigger
        arriving_count = len(branches) - 1

        clock = MockClock(start=0.0)
        executor = make_mock_executor(clock=clock)
        settings = CoalesceSettings(
            name="test_coalesce",
            branches=branches,
            policy="best_effort",
            merge="union",
            timeout_seconds=10.0,
        )
        executor.register_coalesce(settings, node_id="node-001")

        row_id = "row-001"

        # Send partial branches (not all - so merge doesn't happen immediately)
        for i in range(arriving_count):
            token = make_token(
                token_id=f"token-{i}",
                row_id=row_id,
                branch_name=branches[i],
                row_data={"field": i},
            )
            outcome = executor.accept(token, "test_coalesce", step_in_pipeline=0)
            # All should be held (best_effort waits for timeout or all branches)
            assert outcome.held is True, "Should hold when not all branches arrived"

        # Advance past timeout
        clock.advance(11.0)

        # Check timeouts - should merge
        outcomes = executor.check_timeouts("test_coalesce", step_in_pipeline=0)

        assert len(outcomes) == 1
        assert outcomes[0].merged_token is not None
        assert len(outcomes[0].consumed_tokens) == arriving_count


# =============================================================================
# Late Arrival Property Tests
# =============================================================================


class TestLateArrivalProperties:
    """Property tests for late arrival handling."""

    @given(branches=branch_lists(min_size=2, max_size=3))
    @settings(max_examples=30)
    def test_late_arrival_returns_failure(self, branches: list[str]) -> None:
        """Property: After merge completes, late arrivals return failure."""
        executor = make_mock_executor()
        settings = CoalesceSettings(
            name="test_coalesce",
            branches=branches,
            policy="require_all",
            merge="union",
        )
        executor.register_coalesce(settings, node_id="node-001")

        row_id = "row-001"

        # Complete the merge with all branches
        for i, branch in enumerate(branches):
            token = make_token(
                token_id=f"token-{i}",
                row_id=row_id,
                branch_name=branch,
                row_data={"field": i},
            )
            executor.accept(token, "test_coalesce", step_in_pipeline=0)

        # Now send a "late" token for same row_id (simulating duplicate/retry)
        late_token = make_token(
            token_id="token-late",
            row_id=row_id,
            branch_name=branches[0],
            row_data={"field": "late"},
        )
        outcome = executor.accept(late_token, "test_coalesce", step_in_pipeline=0)

        assert outcome.held is False
        assert outcome.failure_reason == "late_arrival_after_merge"
        assert outcome.merged_token is None

    @given(num_late=st.integers(min_value=1, max_value=5))
    @settings(max_examples=20)
    def test_multiple_late_arrivals_all_fail(self, num_late: int) -> None:
        """Property: Multiple late arrivals all consistently fail."""
        branches = ["branch_a", "branch_b"]
        executor = make_mock_executor()
        settings = CoalesceSettings(
            name="test_coalesce",
            branches=branches,
            policy="require_all",
            merge="union",
        )
        executor.register_coalesce(settings, node_id="node-001")

        row_id = "row-001"

        # Complete the merge
        for i, branch in enumerate(branches):
            token = make_token(
                token_id=f"token-{i}",
                row_id=row_id,
                branch_name=branch,
                row_data={"field": i},
            )
            executor.accept(token, "test_coalesce", step_in_pipeline=0)

        # Send multiple late arrivals
        for i in range(num_late):
            late_token = make_token(
                token_id=f"token-late-{i}",
                row_id=row_id,
                branch_name=branches[i % len(branches)],
                row_data={"field": f"late-{i}"},
            )
            outcome = executor.accept(late_token, "test_coalesce", step_in_pipeline=0)

            assert outcome.failure_reason == "late_arrival_after_merge", f"Late arrival {i} should fail consistently"


# =============================================================================
# Memory Bounded Property Tests
# =============================================================================


class TestMemoryBoundedProperties:
    """Property tests for bounded memory invariants."""

    def test_completed_keys_bounded_by_max(self) -> None:
        """Property: _completed_keys never exceeds _max_completed_keys."""
        executor = make_mock_executor()
        # Set a small max for testing
        executor._max_completed_keys = 100

        branches = ["branch_a", "branch_b"]
        settings = CoalesceSettings(
            name="test_coalesce",
            branches=branches,
            policy="require_all",
            merge="union",
        )
        executor.register_coalesce(settings, node_id="node-001")

        # Complete more merges than max_completed_keys
        for row_num in range(150):
            row_id = f"row-{row_num:05d}"
            for i, branch in enumerate(branches):
                token = make_token(
                    token_id=f"token-{row_num}-{i}",
                    row_id=row_id,
                    branch_name=branch,
                    row_data={"value": row_num},
                )
                executor.accept(token, "test_coalesce", step_in_pipeline=0)

        # Verify bounded
        assert len(executor._completed_keys) <= 100, f"_completed_keys has {len(executor._completed_keys)} entries, should be <= 100"

    def test_fifo_eviction_preserves_recent(self) -> None:
        """Property: FIFO eviction keeps most recent, evicts oldest."""
        executor = make_mock_executor()
        executor._max_completed_keys = 10

        branches = ["branch_a", "branch_b"]
        settings = CoalesceSettings(
            name="test_coalesce",
            branches=branches,
            policy="require_all",
            merge="union",
        )
        executor.register_coalesce(settings, node_id="node-001")

        # Complete 20 merges
        for row_num in range(20):
            row_id = f"row-{row_num:03d}"
            for i, branch in enumerate(branches):
                token = make_token(
                    token_id=f"token-{row_num}-{i}",
                    row_id=row_id,
                    branch_name=branch,
                    row_data={"value": row_num},
                )
                executor.accept(token, "test_coalesce", step_in_pipeline=0)

        # Most recent 10 should be retained
        for row_num in range(10, 20):
            key = ("test_coalesce", f"row-{row_num:03d}")
            assert key in executor._completed_keys, f"Recent key {key} should be retained"

        # Oldest 10 should be evicted
        for row_num in range(10):
            key = ("test_coalesce", f"row-{row_num:03d}")
            assert key not in executor._completed_keys, f"Old key {key} should be evicted"


# =============================================================================
# Merge Data Strategy Property Tests
# =============================================================================


class TestMergeDataProperties:
    """Property tests for merge data strategies."""

    @given(
        data_a=row_data,
        data_b=row_data,
    )
    @settings(max_examples=50)
    def test_union_merge_contains_all_fields(self, data_a: dict[str, Any], data_b: dict[str, Any]) -> None:
        """Property: union merge contains fields from all branches."""
        executor = make_mock_executor()
        branches = ["branch_a", "branch_b"]
        settings = CoalesceSettings(
            name="test_coalesce",
            branches=branches,
            policy="require_all",
            merge="union",
        )
        executor.register_coalesce(settings, node_id="node-001")

        # Send both tokens
        token_a = make_token("t-a", "row-001", "branch_a", data_a)
        token_b = make_token("t-b", "row-001", "branch_b", data_b)

        executor.accept(token_a, "test_coalesce", step_in_pipeline=0)
        outcome = executor.accept(token_b, "test_coalesce", step_in_pipeline=0)

        merged_data = outcome.merged_token.row_data

        # All keys from both dicts should be in merged (last write wins for conflicts)
        all_keys = set(data_a.keys()) | set(data_b.keys())
        for key in all_keys:
            assert key in merged_data, f"Key '{key}' missing from union merge"

    @given(
        data_a=row_data,
        data_b=row_data,
    )
    @settings(max_examples=50)
    def test_nested_merge_has_branch_hierarchy(self, data_a: dict[str, Any], data_b: dict[str, Any]) -> None:
        """Property: nested merge creates branch-keyed hierarchy."""
        executor = make_mock_executor()
        branches = ["branch_a", "branch_b"]
        settings = CoalesceSettings(
            name="test_coalesce",
            branches=branches,
            policy="require_all",
            merge="nested",
        )
        executor.register_coalesce(settings, node_id="node-001")

        token_a = make_token("t-a", "row-001", "branch_a", data_a)
        token_b = make_token("t-b", "row-001", "branch_b", data_b)

        executor.accept(token_a, "test_coalesce", step_in_pipeline=0)
        outcome = executor.accept(token_b, "test_coalesce", step_in_pipeline=0)

        merged_data = outcome.merged_token.row_data

        # Should have branch names as top-level keys
        assert "branch_a" in merged_data, "nested merge should have 'branch_a' key"
        assert "branch_b" in merged_data, "nested merge should have 'branch_b' key"
        assert merged_data["branch_a"] == data_a
        assert merged_data["branch_b"] == data_b

    @given(data_selected=row_data, data_other=row_data)
    @settings(max_examples=50)
    def test_select_merge_takes_only_selected_branch(self, data_selected: dict[str, Any], data_other: dict[str, Any]) -> None:
        """Property: select merge takes only the selected branch's data."""
        executor = make_mock_executor()
        branches = ["selected_branch", "other_branch"]
        settings = CoalesceSettings(
            name="test_coalesce",
            branches=branches,
            policy="require_all",
            merge="select",
            select_branch="selected_branch",
        )
        executor.register_coalesce(settings, node_id="node-001")

        token_selected = make_token("t-sel", "row-001", "selected_branch", data_selected)
        token_other = make_token("t-oth", "row-001", "other_branch", data_other)

        executor.accept(token_selected, "test_coalesce", step_in_pipeline=0)
        outcome = executor.accept(token_other, "test_coalesce", step_in_pipeline=0)

        merged_data = outcome.merged_token.row_data

        # Should be exactly the selected branch's data
        assert merged_data == data_selected, "select merge should use only selected branch"


# =============================================================================
# Token Conservation Property Tests
# =============================================================================


class TestTokenConservationProperties:
    """Property tests for token conservation during coalesce."""

    @given(branches=branch_lists(min_size=2, max_size=5))
    @settings(max_examples=30)
    def test_consumed_tokens_equals_arrived_tokens(self, branches: list[str]) -> None:
        """Property: consumed_tokens count matches number of arrived tokens."""
        executor = make_mock_executor()
        settings = CoalesceSettings(
            name="test_coalesce",
            branches=branches,
            policy="require_all",
            merge="union",
        )
        executor.register_coalesce(settings, node_id="node-001")

        row_id = "row-001"
        sent_tokens = []

        # Send all branches
        for i, branch in enumerate(branches):
            token = make_token(
                token_id=f"token-{i}",
                row_id=row_id,
                branch_name=branch,
                row_data={"field": i},
            )
            sent_tokens.append(token)
            outcome = executor.accept(token, "test_coalesce", step_in_pipeline=0)

        # Outcome from last accept has the merge result
        assert len(outcome.consumed_tokens) == len(branches)

        # Verify token IDs match
        consumed_ids = {t.token_id for t in outcome.consumed_tokens}
        sent_ids = {t.token_id for t in sent_tokens}
        assert consumed_ids == sent_ids, "All sent tokens should be consumed"


# =============================================================================
# Coalesce Metadata Property Tests
# =============================================================================


class TestCoalesceMetadataProperties:
    """Property tests for coalesce audit metadata."""

    @given(branches=branch_lists(min_size=2, max_size=4))
    @settings(max_examples=30)
    def test_metadata_contains_policy_and_strategy(self, branches: list[str]) -> None:
        """Property: coalesce metadata includes policy and merge strategy."""
        executor = make_mock_executor()
        settings = CoalesceSettings(
            name="test_coalesce",
            branches=branches,
            policy="require_all",
            merge="nested",
        )
        executor.register_coalesce(settings, node_id="node-001")

        row_id = "row-001"
        for i, branch in enumerate(branches):
            token = make_token(f"token-{i}", row_id, branch, {"field": i})
            outcome = executor.accept(token, "test_coalesce", step_in_pipeline=0)

        metadata = outcome.coalesce_metadata
        assert metadata is not None
        assert metadata["policy"] == "require_all"
        assert metadata["merge_strategy"] == "nested"
        assert set(metadata["expected_branches"]) == set(branches)
        assert set(metadata["branches_arrived"]) == set(branches)

    @given(branches=branch_lists(min_size=2, max_size=3))
    @settings(max_examples=20)
    def test_metadata_arrival_order_is_chronological(self, branches: list[str]) -> None:
        """Property: arrival_order metadata is sorted chronologically."""
        clock = MockClock(start=0.0)
        executor = make_mock_executor(clock=clock)
        settings = CoalesceSettings(
            name="test_coalesce",
            branches=branches,
            policy="require_all",
            merge="union",
        )
        executor.register_coalesce(settings, node_id="node-001")

        row_id = "row-001"

        # Send branches with time gaps
        for i, branch in enumerate(branches):
            clock.advance(1.0)  # 1 second between each
            token = make_token(f"token-{i}", row_id, branch, {"field": i})
            outcome = executor.accept(token, "test_coalesce", step_in_pipeline=0)

        arrival_order = outcome.coalesce_metadata["arrival_order"]

        # Verify chronological order
        offsets = [entry["arrival_offset_ms"] for entry in arrival_order]
        assert offsets == sorted(offsets), "arrival_order should be chronologically sorted"

        # Verify offsets are approximately correct (1000ms apart)
        for i, offset in enumerate(offsets):
            expected_offset = i * 1000  # First is 0, second is 1000, etc.
            assert abs(offset - expected_offset) < 1, f"Offset {i} should be ~{expected_offset}ms"
