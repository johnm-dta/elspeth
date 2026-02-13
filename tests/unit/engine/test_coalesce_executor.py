# tests/unit/engine/test_coalesce_executor.py
"""Comprehensive unit tests for CoalesceExecutor.

Tests merge policies (require_all, first, quorum, best_effort),
merge strategies (union, nested, select), timeout handling,
flush behaviour, branch loss notifications, late arrivals,
and audit trail recording.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, Mock, patch
from uuid import uuid4

import pytest

from elspeth.contracts import TokenInfo
from elspeth.contracts.enums import NodeStateStatus, RowOutcome
from elspeth.contracts.errors import OrchestrationInvariantError
from elspeth.contracts.schema_contract import PipelineRow, SchemaContract
from elspeth.core.config import CoalesceSettings
from elspeth.engine.clock import MockClock
from elspeth.engine.coalesce_executor import CoalesceExecutor, CoalesceOutcome, _PendingCoalesce
from elspeth.testing import make_field, make_row

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_state_counter = 0


def _next_state_id() -> str:
    global _state_counter
    _state_counter += 1
    return f"state_{_state_counter:04d}"


def _make_contract(fields: list[Any] | None = None) -> SchemaContract:
    """Create an OBSERVED contract for testing."""
    if fields is None:
        fields = [
            make_field(
                "amount",
                original_name="amount",
                python_type=int,
                required=True,
                source="declared",
            ),
        ]
    return SchemaContract(fields=tuple(fields), mode="OBSERVED", locked=True)


def _make_token(
    row_id="row_1",
    token_id="tok_1",
    branch_name="branch_a",
    data=None,
    contract=None,
):
    """Build a TokenInfo suitable for coalesce testing."""
    if data is None:
        data = {"amount": 100}
    if contract is None:
        contract = _make_contract()
    row_data = make_row(data, contract=contract)
    return TokenInfo(
        row_id=row_id,
        token_id=token_id,
        row_data=row_data,
        branch_name=branch_name,
    )


def _make_executor(clock=None, max_completed_keys: int = 10000):
    """Build a CoalesceExecutor with mocked dependencies.

    Returns (executor, recorder, token_manager, clock).
    """
    recorder = MagicMock()
    recorder.begin_node_state.side_effect = lambda **kw: Mock(state_id=_next_state_id())
    span_factory = MagicMock()
    token_manager = MagicMock()

    def coalesce_tokens_impl(parents, merged_data, node_id):
        return TokenInfo(
            row_id=parents[0].row_id,
            token_id=f"merged_{uuid4().hex[:8]}",
            row_data=merged_data,
            join_group_id=f"join_{uuid4().hex[:8]}",
        )

    token_manager.coalesce_tokens.side_effect = coalesce_tokens_impl

    if clock is None:
        clock = MockClock(start=100.0)

    def step_resolver(node_id: str) -> int:
        return 5

    executor = CoalesceExecutor(
        recorder,
        span_factory,
        token_manager,
        "run_1",
        step_resolver=step_resolver,
        clock=clock,
        max_completed_keys=max_completed_keys,
    )
    return executor, recorder, token_manager, clock


def _settings(
    name="merge",
    branches=None,
    policy="require_all",
    merge="union",
    timeout_seconds=None,
    quorum_count=None,
    select_branch=None,
):
    """Shorthand for building CoalesceSettings."""
    if branches is None:
        branches = ["a", "b"]
    return CoalesceSettings(
        name=name,
        branches=branches,
        policy=policy,
        merge=merge,
        timeout_seconds=timeout_seconds,
        quorum_count=quorum_count,
        select_branch=select_branch,
    )


# ===========================================================================
# CoalesceOutcome dataclass
# ===========================================================================


class TestCoalesceOutcome:
    def test_defaults(self):
        outcome = CoalesceOutcome(held=True)
        assert outcome.held is True
        assert outcome.merged_token is None
        assert outcome.consumed_tokens == []
        assert outcome.coalesce_metadata is None
        assert outcome.failure_reason is None
        assert outcome.coalesce_name is None
        assert outcome.outcomes_recorded is False

    def test_custom_values(self):
        token = _make_token()
        outcome = CoalesceOutcome(
            held=False,
            merged_token=token,
            consumed_tokens=[token],
            coalesce_metadata={"policy": "require_all"},
            failure_reason="late_arrival_after_merge",
            coalesce_name="merge",
            outcomes_recorded=True,
        )
        assert outcome.held is False
        assert outcome.merged_token is token
        assert outcome.consumed_tokens == [token]
        assert outcome.coalesce_metadata["policy"] == "require_all"
        assert outcome.failure_reason == "late_arrival_after_merge"
        assert outcome.coalesce_name == "merge"
        assert outcome.outcomes_recorded is True


# ===========================================================================
# register_coalesce / get_registered_names
# ===========================================================================


class TestRegisterCoalesce:
    def test_register_single(self):
        executor, *_ = _make_executor()
        executor.register_coalesce(_settings(name="merge1"), "node_1")
        assert executor.get_registered_names() == ["merge1"]

    def test_register_multiple(self):
        executor, *_ = _make_executor()
        executor.register_coalesce(_settings(name="m1"), "n1")
        executor.register_coalesce(_settings(name="m2"), "n2")
        assert set(executor.get_registered_names()) == {"m1", "m2"}

    def test_get_registered_names_empty(self):
        executor, *_ = _make_executor()
        assert executor.get_registered_names() == []


# ===========================================================================
# accept() -- basic validation
# ===========================================================================


class TestAcceptBasics:
    def test_unregistered_coalesce_raises(self):
        executor, *_ = _make_executor()
        token = _make_token(branch_name="a")
        with pytest.raises(ValueError, match="not registered"):
            executor.accept(token, "nonexistent")

    def test_token_without_branch_raises(self):
        executor, *_ = _make_executor()
        executor.register_coalesce(_settings(), "node_1")
        token = TokenInfo(
            row_id="row_1",
            token_id="tok_1",
            row_data=make_row({"amount": 1}),
            branch_name=None,
        )
        with pytest.raises(ValueError, match="no branch_name"):
            executor.accept(token, "merge")

    def test_unexpected_branch_raises(self):
        executor, *_ = _make_executor()
        executor.register_coalesce(_settings(branches=["a", "b"]), "node_1")
        token = _make_token(branch_name="c")
        with pytest.raises(ValueError, match="not in expected branches"):
            executor.accept(token, "merge")

    def test_duplicate_arrival_raises(self):
        executor, *_ = _make_executor()
        executor.register_coalesce(_settings(), "node_1")
        t1 = _make_token(branch_name="a", token_id="tok_1")
        t2 = _make_token(branch_name="a", token_id="tok_2")
        executor.accept(t1, "merge")
        with pytest.raises(ValueError, match="Duplicate arrival"):
            executor.accept(t2, "merge")

    def test_first_token_held(self):
        executor, *_ = _make_executor()
        executor.register_coalesce(_settings(), "node_1")
        token = _make_token(branch_name="a")
        outcome = executor.accept(token, "merge")
        assert outcome.held is True
        assert outcome.merged_token is None

    def test_outcome_has_coalesce_name(self):
        executor, *_ = _make_executor()
        executor.register_coalesce(_settings(name="my_merge"), "node_1")
        token = _make_token(branch_name="a")
        outcome = executor.accept(token, "my_merge")
        assert outcome.coalesce_name == "my_merge"


# ===========================================================================
# require_all policy
# ===========================================================================


class TestRequireAllPolicy:
    def _setup(self, branches=None):
        if branches is None:
            branches = ["a", "b"]
        executor, recorder, tm, clock = _make_executor()
        s = _settings(branches=branches, policy="require_all")
        executor.register_coalesce(s, "node_1")
        return executor, recorder, tm, clock

    def test_two_branches_first_held_second_merges(self):
        executor, _, _, _ = self._setup()
        o1 = executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        o2 = executor.accept(_make_token(branch_name="b", token_id="t2"), "merge")
        assert o1.held is True
        assert o2.held is False
        assert o2.merged_token is not None

    def test_three_branches(self):
        executor, _, _, _ = self._setup(branches=["a", "b", "c"])
        o1 = executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        o2 = executor.accept(_make_token(branch_name="b", token_id="t2"), "merge")
        o3 = executor.accept(_make_token(branch_name="c", token_id="t3"), "merge")
        assert o1.held is True
        assert o2.held is True
        assert o3.held is False
        assert o3.merged_token is not None

    def test_merged_token_in_outcome(self):
        executor, _, _, _ = self._setup()
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        o = executor.accept(_make_token(branch_name="b", token_id="t2"), "merge")
        assert o.merged_token is not None
        assert o.merged_token.row_id == "row_1"
        assert o.merged_token.join_group_id is not None

    def test_consumed_tokens_list(self):
        executor, _, _, _ = self._setup()
        t1 = _make_token(branch_name="a", token_id="t1")
        t2 = _make_token(branch_name="b", token_id="t2")
        executor.accept(t1, "merge")
        o = executor.accept(t2, "merge")
        consumed_ids = {t.token_id for t in o.consumed_tokens}
        assert consumed_ids == {"t1", "t2"}

    def test_coalesce_metadata(self):
        executor, _, _, _ = self._setup()
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        o = executor.accept(_make_token(branch_name="b", token_id="t2"), "merge")
        md = o.coalesce_metadata
        assert md["policy"] == "require_all"
        assert md["merge_strategy"] == "union"
        assert set(md["expected_branches"]) == {"a", "b"}
        assert set(md["branches_arrived"]) == {"a", "b"}

    def test_audit_begin_node_state_for_each_token(self):
        executor, recorder, _, _ = self._setup()
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        executor.accept(_make_token(branch_name="b", token_id="t2"), "merge")
        # begin_node_state called once per accepted token
        assert recorder.begin_node_state.call_count == 2

    def test_audit_complete_node_state_completed(self):
        executor, recorder, _, _ = self._setup()
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        executor.accept(_make_token(branch_name="b", token_id="t2"), "merge")
        # On merge, each consumed token's state is completed with COMPLETED
        completed_calls = [c for c in recorder.complete_node_state.call_args_list if c.kwargs.get("status") == NodeStateStatus.COMPLETED]
        assert len(completed_calls) == 2

    def test_audit_record_token_outcome_coalesced(self):
        executor, recorder, _, _ = self._setup()
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        executor.accept(_make_token(branch_name="b", token_id="t2"), "merge")
        outcome_calls = recorder.record_token_outcome.call_args_list
        assert len(outcome_calls) == 2
        for c in outcome_calls:
            assert c.kwargs["outcome"] == RowOutcome.COALESCED

    def test_token_manager_coalesce_tokens_called(self):
        executor, _, tm, _ = self._setup()
        t1 = _make_token(branch_name="a", token_id="t1")
        t2 = _make_token(branch_name="b", token_id="t2")
        executor.accept(t1, "merge")
        executor.accept(t2, "merge")
        tm.coalesce_tokens.assert_called_once()
        kw = tm.coalesce_tokens.call_args.kwargs
        assert kw["node_id"] == "node_1"
        parent_ids = {p.token_id for p in kw["parents"]}
        assert parent_ids == {"t1", "t2"}


# ===========================================================================
# first policy
# ===========================================================================


class TestFirstPolicy:
    def test_single_token_triggers_merge(self):
        executor, _, _, _ = _make_executor()
        s = _settings(policy="first")
        executor.register_coalesce(s, "node_1")
        t = _make_token(branch_name="a", token_id="t1")
        o = executor.accept(t, "merge")
        assert o.held is False
        assert o.merged_token is not None

    def test_only_one_consumed_token(self):
        executor, _, _, _ = _make_executor()
        s = _settings(policy="first")
        executor.register_coalesce(s, "node_1")
        t = _make_token(branch_name="a", token_id="t1")
        o = executor.accept(t, "merge")
        assert len(o.consumed_tokens) == 1
        assert o.consumed_tokens[0].token_id == "t1"

    def test_second_arrival_is_late(self):
        executor, _, _, _ = _make_executor()
        s = _settings(policy="first")
        executor.register_coalesce(s, "node_1")
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        o = executor.accept(_make_token(branch_name="b", token_id="t2"), "merge")
        assert o.held is False
        assert o.failure_reason == "late_arrival_after_merge"


# ===========================================================================
# quorum policy
# ===========================================================================


class TestQuorumPolicy:
    def test_quorum_met_triggers_merge(self):
        executor, _, _, _ = _make_executor()
        s = _settings(branches=["a", "b", "c"], policy="quorum", quorum_count=2)
        executor.register_coalesce(s, "node_1")
        o1 = executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        o2 = executor.accept(_make_token(branch_name="b", token_id="t2"), "merge")
        assert o1.held is True
        assert o2.held is False
        assert o2.merged_token is not None

    def test_third_arrival_is_late(self):
        executor, _, _, _ = _make_executor()
        s = _settings(branches=["a", "b", "c"], policy="quorum", quorum_count=2)
        executor.register_coalesce(s, "node_1")
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        executor.accept(_make_token(branch_name="b", token_id="t2"), "merge")
        o = executor.accept(_make_token(branch_name="c", token_id="t3"), "merge")
        assert o.failure_reason == "late_arrival_after_merge"

    def test_quorum_of_one_triggers_like_first(self):
        executor, _, _, _ = _make_executor()
        s = _settings(branches=["a", "b"], policy="quorum", quorum_count=1)
        executor.register_coalesce(s, "node_1")
        o = executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        assert o.held is False
        assert o.merged_token is not None


# ===========================================================================
# best_effort policy
# ===========================================================================


class TestBestEffortPolicy:
    def test_does_not_merge_on_partial_arrival(self):
        """best_effort requires timeout or all-accounted-for to merge."""
        executor, _, _, _ = _make_executor()
        s = _settings(policy="best_effort", timeout_seconds=60.0)
        executor.register_coalesce(s, "node_1")
        o = executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        assert o.held is True

    def test_merges_when_all_accounted_for(self):
        """best_effort merges when arrived + lost >= expected."""
        executor, _, _, _ = _make_executor()
        s = _settings(branches=["a", "b"], policy="best_effort", timeout_seconds=60.0)
        executor.register_coalesce(s, "node_1")
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        # Notify branch b lost
        result = executor.notify_branch_lost("merge", "row_1", "b", "error_routed")
        assert result is not None
        assert result.merged_token is not None

    def test_all_branches_arrived_triggers_merge(self):
        """best_effort merges immediately when all branches arrive."""
        executor, _, _, _ = _make_executor()
        s = _settings(branches=["a", "b"], policy="best_effort", timeout_seconds=60.0)
        executor.register_coalesce(s, "node_1")
        o1 = executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        o2 = executor.accept(_make_token(branch_name="b", token_id="t2"), "merge")
        assert o1.held is True
        assert o2.held is False
        assert o2.merged_token is not None


# ===========================================================================
# late arrival
# ===========================================================================


class TestLateArrival:
    def test_late_arrival_outcome(self):
        executor, _, _, _ = _make_executor()
        executor.register_coalesce(_settings(), "node_1")
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        executor.accept(_make_token(branch_name="b", token_id="t2"), "merge")
        # A new token with same row_id arriving at same coalesce is a late arrival
        late_token = _make_token(branch_name="a", token_id="t_late", row_id="row_1")
        o = executor.accept(late_token, "merge")
        assert o.held is False
        assert o.failure_reason == "late_arrival_after_merge"
        assert o.outcomes_recorded is True

    def test_late_arrival_records_failed_state_and_outcome(self):
        executor, recorder, _, _ = _make_executor()
        executor.register_coalesce(_settings(), "node_1")
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        executor.accept(_make_token(branch_name="b", token_id="t2"), "merge")
        recorder.reset_mock()
        late = _make_token(branch_name="a", token_id="t_late", row_id="row_1")
        executor.accept(late, "merge")

        # Should begin + complete with FAILED
        recorder.begin_node_state.assert_called_once()
        recorder.complete_node_state.assert_called_once()
        fail_call = recorder.complete_node_state.call_args
        assert fail_call.kwargs["status"] == NodeStateStatus.FAILED

        # Should record a terminal FAILED token outcome immediately
        recorder.record_token_outcome.assert_called_once()
        outcome_call = recorder.record_token_outcome.call_args
        assert outcome_call.kwargs["token_id"] == "t_late"
        assert outcome_call.kwargs["outcome"] == RowOutcome.FAILED
        assert isinstance(outcome_call.kwargs["error_hash"], str)
        assert len(outcome_call.kwargs["error_hash"]) == 16

    def test_late_arrival_consumed_tokens(self):
        executor, _, _, _ = _make_executor()
        executor.register_coalesce(_settings(), "node_1")
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        executor.accept(_make_token(branch_name="b", token_id="t2"), "merge")
        late = _make_token(branch_name="a", token_id="t_late", row_id="row_1")
        o = executor.accept(late, "merge")
        assert len(o.consumed_tokens) == 1
        assert o.consumed_tokens[0].token_id == "t_late"

    def test_late_arrival_metadata_has_policy(self):
        executor, _, _, _ = _make_executor()
        executor.register_coalesce(_settings(), "node_1")
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        executor.accept(_make_token(branch_name="b", token_id="t2"), "merge")
        late = _make_token(branch_name="a", token_id="t_late", row_id="row_1")
        o = executor.accept(late, "merge")
        assert o.coalesce_metadata["policy"] == "require_all"
        assert "reason" in o.coalesce_metadata


# ===========================================================================
# union merge
# ===========================================================================


class TestUnionMerge:
    def test_fields_from_both_branches(self):
        executor, _, tm, _ = _make_executor()
        executor.register_coalesce(_settings(merge="union"), "node_1")
        t1 = _make_token(branch_name="a", token_id="t1", data={"x": 1})
        t2 = _make_token(branch_name="b", token_id="t2", data={"y": 2})
        executor.accept(t1, "merge")
        executor.accept(t2, "merge")
        merged_data = tm.coalesce_tokens.call_args.kwargs["merged_data"]
        d = merged_data.to_dict()
        assert d["x"] == 1
        assert d["y"] == 2

    def test_last_branch_wins_on_collision(self):
        executor, _, tm, _ = _make_executor()
        executor.register_coalesce(_settings(branches=["a", "b"], merge="union"), "node_1")
        t1 = _make_token(branch_name="a", token_id="t1", data={"shared": "from_a"})
        t2 = _make_token(branch_name="b", token_id="t2", data={"shared": "from_b"})
        executor.accept(t1, "merge")
        executor.accept(t2, "merge")
        merged_data = tm.coalesce_tokens.call_args.kwargs["merged_data"]
        assert merged_data.to_dict()["shared"] == "from_b"

    def test_collision_metadata_recorded(self):
        executor, _, _, _ = _make_executor()
        executor.register_coalesce(_settings(branches=["a", "b"], merge="union"), "node_1")
        t1 = _make_token(branch_name="a", token_id="t1", data={"shared": "from_a"})
        t2 = _make_token(branch_name="b", token_id="t2", data={"shared": "from_b"})
        executor.accept(t1, "merge")
        o = executor.accept(t2, "merge")
        assert "union_field_collisions" in o.coalesce_metadata
        assert "shared" in o.coalesce_metadata["union_field_collisions"]

    def test_no_collisions_no_collision_metadata(self):
        """When there are no field collisions, collision metadata should be absent."""
        executor, _, _, _ = _make_executor()
        executor.register_coalesce(_settings(merge="union"), "node_1")
        t1 = _make_token(branch_name="a", token_id="t1", data={"x": 1})
        t2 = _make_token(branch_name="b", token_id="t2", data={"y": 2})
        executor.accept(t1, "merge")
        o = executor.accept(t2, "merge")
        assert "union_field_collisions" not in o.coalesce_metadata

    def test_collision_tracks_all_contributing_branches(self):
        """Collision metadata lists all branches that contributed the same field."""
        executor, _, _, _ = _make_executor()
        s = _settings(branches=["a", "b", "c"], merge="union", policy="require_all")
        executor.register_coalesce(s, "node_1")
        t1 = _make_token(branch_name="a", token_id="t1", data={"f": 1})
        t2 = _make_token(branch_name="b", token_id="t2", data={"f": 2})
        t3 = _make_token(branch_name="c", token_id="t3", data={"f": 3})
        executor.accept(t1, "merge")
        executor.accept(t2, "merge")
        o = executor.accept(t3, "merge")
        collision_branches = o.coalesce_metadata["union_field_collisions"]["f"]
        assert "a" in collision_branches
        assert "b" in collision_branches
        assert "c" in collision_branches


# ===========================================================================
# nested merge
# ===========================================================================


class TestNestedMerge:
    def test_each_branch_nested(self):
        executor, _, tm, _ = _make_executor()
        executor.register_coalesce(_settings(merge="nested"), "node_1")
        t1 = _make_token(branch_name="a", token_id="t1", data={"x": 1})
        t2 = _make_token(branch_name="b", token_id="t2", data={"y": 2})
        executor.accept(t1, "merge")
        executor.accept(t2, "merge")
        merged_data = tm.coalesce_tokens.call_args.kwargs["merged_data"]
        d = merged_data.to_dict()
        assert d["a"] == {"x": 1}
        assert d["b"] == {"y": 2}

    def test_only_arrived_branches_included(self):
        """With first policy, only the arrived branch appears in nested data."""
        executor, _, tm, _ = _make_executor()
        executor.register_coalesce(
            _settings(policy="first", merge="nested"),
            "node_1",
        )
        t = _make_token(branch_name="a", token_id="t1", data={"x": 1})
        executor.accept(t, "merge")
        merged_data = tm.coalesce_tokens.call_args.kwargs["merged_data"]
        d = merged_data.to_dict()
        assert "a" in d
        assert "b" not in d

    def test_nested_preserves_each_branch_data(self):
        """Nested merge preserves full row data as nested dict for each branch."""
        executor, _, tm, _ = _make_executor()
        executor.register_coalesce(_settings(merge="nested"), "node_1")
        t1 = _make_token(branch_name="a", token_id="t1", data={"x": 1, "y": 2})
        t2 = _make_token(branch_name="b", token_id="t2", data={"z": 3})
        executor.accept(t1, "merge")
        executor.accept(t2, "merge")
        merged_data = tm.coalesce_tokens.call_args.kwargs["merged_data"]
        d = merged_data.to_dict()
        assert d["a"]["x"] == 1
        assert d["a"]["y"] == 2
        assert d["b"]["z"] == 3


# ===========================================================================
# select merge
# ===========================================================================


class TestSelectMerge:
    def test_selected_branch_data(self):
        executor, _, tm, _ = _make_executor()
        s = _settings(merge="select", select_branch="a")
        executor.register_coalesce(s, "node_1")
        t1 = _make_token(branch_name="a", token_id="t1", data={"x": 10})
        t2 = _make_token(branch_name="b", token_id="t2", data={"y": 20})
        executor.accept(t1, "merge")
        executor.accept(t2, "merge")
        merged_data = tm.coalesce_tokens.call_args.kwargs["merged_data"]
        d = merged_data.to_dict()
        assert d == {"x": 10}

    def test_select_branch_not_arrived_failure(self):
        """If select_branch hasn't arrived but merge triggers, outcome is failure."""
        executor, _, _, _ = _make_executor()
        # quorum allows merge before select_branch arrives
        s = _settings(
            branches=["a", "b", "c"],
            policy="quorum",
            quorum_count=2,
            merge="select",
            select_branch="c",
        )
        executor.register_coalesce(s, "node_1")
        t1 = _make_token(branch_name="a", token_id="t1")
        t2 = _make_token(branch_name="b", token_id="t2")
        executor.accept(t1, "merge")
        o = executor.accept(t2, "merge")
        assert o.failure_reason == "select_branch_not_arrived"
        assert o.outcomes_recorded is True

    def test_select_ignores_other_branch_data(self):
        """Select merge returns only the selected branch's data."""
        executor, _, tm, _ = _make_executor()
        s = _settings(merge="select", select_branch="b")
        executor.register_coalesce(s, "node_1")
        t1 = _make_token(branch_name="a", token_id="t1", data={"a_val": 1})
        t2 = _make_token(branch_name="b", token_id="t2", data={"b_val": 2})
        executor.accept(t1, "merge")
        executor.accept(t2, "merge")
        merged_data = tm.coalesce_tokens.call_args.kwargs["merged_data"]
        d = merged_data.to_dict()
        assert d == {"b_val": 2}
        assert "a_val" not in d


# ===========================================================================
# check_timeouts
# ===========================================================================


class TestCheckTimeouts:
    def test_no_timeout_configured_returns_empty(self):
        executor, _, _, _ = _make_executor()
        executor.register_coalesce(_settings(policy="require_all"), "node_1")
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        results = executor.check_timeouts("merge")
        assert results == []

    def test_not_expired_returns_empty(self):
        executor, _, _, clock = _make_executor()
        s = _settings(policy="best_effort", timeout_seconds=10.0)
        executor.register_coalesce(s, "node_1")
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        clock.advance(5.0)  # Only 5s of 10s timeout
        results = executor.check_timeouts("merge")
        assert results == []

    def test_best_effort_expired_merges(self):
        executor, _, _, clock = _make_executor()
        s = _settings(policy="best_effort", timeout_seconds=10.0)
        executor.register_coalesce(s, "node_1")
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        clock.advance(11.0)
        results = executor.check_timeouts("merge")
        assert len(results) == 1
        assert results[0].merged_token is not None

    def test_best_effort_expired_cleans_pending(self):
        """Timeout-triggered merge removes the pending entry."""
        executor, _, _, clock = _make_executor()
        s = _settings(policy="best_effort", timeout_seconds=10.0)
        executor.register_coalesce(s, "node_1")
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        assert ("merge", "row_1") in executor._pending
        clock.advance(11.0)
        executor.check_timeouts("merge")
        assert ("merge", "row_1") not in executor._pending

    def test_quorum_expired_quorum_not_met_fails(self):
        executor, _, _, clock = _make_executor()
        s = _settings(
            branches=["a", "b", "c"],
            policy="quorum",
            quorum_count=2,
            timeout_seconds=10.0,
        )
        executor.register_coalesce(s, "node_1")
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        clock.advance(11.0)
        results = executor.check_timeouts("merge")
        assert len(results) == 1
        assert results[0].failure_reason == "quorum_not_met_at_timeout"
        assert results[0].outcomes_recorded is True

    def test_require_all_expired_fails(self):
        executor, _, _, clock = _make_executor()
        s = _settings(policy="require_all", timeout_seconds=5.0)
        executor.register_coalesce(s, "node_1")
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        clock.advance(6.0)
        results = executor.check_timeouts("merge")
        assert len(results) == 1
        assert results[0].failure_reason == "incomplete_branches"
        assert results[0].outcomes_recorded is True

    def test_multiple_pending_some_expired(self):
        executor, _, _, clock = _make_executor()
        s = _settings(policy="best_effort", timeout_seconds=10.0)
        executor.register_coalesce(s, "node_1")
        # First row arrives at t=100
        executor.accept(_make_token(branch_name="a", token_id="t1", row_id="row_1"), "merge")
        clock.advance(8.0)  # t=108
        # Second row arrives at t=108
        executor.accept(_make_token(branch_name="a", token_id="t2", row_id="row_2"), "merge")
        clock.advance(3.0)  # t=111 -- row_1 expired (11s > 10s), row_2 not (3s < 10s)
        results = executor.check_timeouts("merge")
        assert len(results) == 1  # Only row_1 expired

    def test_unregistered_coalesce_raises(self):
        executor, *_ = _make_executor()
        with pytest.raises(ValueError, match="not registered"):
            executor.check_timeouts("ghost")

    def test_exact_timeout_boundary_triggers(self):
        """Timeout check fires when elapsed == timeout_seconds."""
        executor, _, _, clock = _make_executor()
        s = _settings(policy="best_effort", timeout_seconds=10.0)
        executor.register_coalesce(s, "node_1")
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        clock.advance(10.0)  # Exactly 10s
        results = executor.check_timeouts("merge")
        assert len(results) == 1
        assert results[0].merged_token is not None


# ===========================================================================
# flush_pending
# ===========================================================================


class TestFlushPending:
    def test_best_effort_with_arrivals_merges(self):
        executor, _, _, _ = _make_executor()
        s = _settings(policy="best_effort", timeout_seconds=60.0)
        executor.register_coalesce(s, "node_1")
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        results = executor.flush_pending()
        assert len(results) == 1
        assert results[0].merged_token is not None

    def test_best_effort_one_lost_one_arrived_flush(self):
        """Flush merges arrived tokens even when some branches are lost."""
        executor, _, _, _ = _make_executor()
        s = _settings(branches=["a", "b", "c"], policy="best_effort", timeout_seconds=60.0)
        executor.register_coalesce(s, "node_1")
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        # Don't report any losses; flush should merge what's there
        results = executor.flush_pending()
        assert len(results) == 1
        assert results[0].merged_token is not None

    def test_quorum_not_met_at_flush_fails(self):
        executor, _, _, _ = _make_executor()
        s = _settings(
            branches=["a", "b", "c"],
            policy="quorum",
            quorum_count=2,
            timeout_seconds=60.0,
        )
        executor.register_coalesce(s, "node_1")
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        results = executor.flush_pending()
        assert len(results) == 1
        assert results[0].failure_reason == "quorum_not_met"

    def test_require_all_fails(self):
        executor, _, _, _ = _make_executor()
        s = _settings(policy="require_all", timeout_seconds=5.0)
        executor.register_coalesce(s, "node_1")
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        results = executor.flush_pending()
        assert len(results) == 1
        assert results[0].failure_reason == "incomplete_branches"

    def test_first_policy_with_pending_raises(self):
        executor, _, _, _ = _make_executor()
        s = _settings(policy="first")
        executor.register_coalesce(s, "node_1")
        # Normally impossible since first merges immediately.
        # Force a pending entry for the test.
        key = ("merge", "row_1")
        executor._pending[key] = _PendingCoalesce(
            arrived={"a": _make_token(branch_name="a")},
            arrival_times={"a": 100.0},
            first_arrival=100.0,
            pending_state_ids={"a": "state_fake"},
        )
        with pytest.raises(RuntimeError, match="Invariant violation"):
            executor.flush_pending()

    def test_flush_clears_completed_keys(self):
        executor, _, _, _ = _make_executor()
        s = _settings(policy="require_all", timeout_seconds=5.0)
        executor.register_coalesce(s, "node_1")
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        executor.accept(_make_token(branch_name="b", token_id="t2"), "merge")
        assert len(executor._completed_keys) == 1
        executor.flush_pending()
        assert len(executor._completed_keys) == 0

    def test_flush_no_pending_returns_empty(self):
        """Flush with no pending entries returns an empty list."""
        executor, _, _, _ = _make_executor()
        s = _settings(policy="require_all")
        executor.register_coalesce(s, "node_1")
        results = executor.flush_pending()
        assert results == []

    def test_flush_multiple_pending_rows(self):
        """Flush processes all pending entries across different rows."""
        executor, _, _, _ = _make_executor()
        s = _settings(policy="require_all", timeout_seconds=5.0)
        executor.register_coalesce(s, "node_1")
        executor.accept(_make_token(branch_name="a", token_id="t1", row_id="r1"), "merge")
        executor.accept(_make_token(branch_name="a", token_id="t2", row_id="r2"), "merge")
        results = executor.flush_pending()
        assert len(results) == 2
        for r in results:
            assert r.failure_reason == "incomplete_branches"


# ===========================================================================
# notify_branch_lost
# ===========================================================================


class TestNotifyBranchLost:
    def test_unregistered_coalesce_raises(self):
        executor, *_ = _make_executor()
        with pytest.raises(ValueError, match="not registered"):
            executor.notify_branch_lost("ghost", "row_1", "a", "reason")

    def test_unknown_branch_raises(self):
        executor, *_ = _make_executor()
        executor.register_coalesce(_settings(branches=["a", "b"]), "node_1")
        with pytest.raises(ValueError, match="not in expected branches"):
            executor.notify_branch_lost("merge", "row_1", "c", "reason")

    def test_require_all_any_loss_fails(self):
        executor, *_ = _make_executor()
        executor.register_coalesce(_settings(branches=["a", "b"], policy="require_all"), "node_1")
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        result = executor.notify_branch_lost("merge", "row_1", "b", "error_routed")
        assert result is not None
        assert result.failure_reason is not None
        assert "branch_lost" in result.failure_reason

    def test_require_all_loss_before_any_arrival_fails(self):
        """require_all: branch loss even before any arrivals triggers failure."""
        executor, *_ = _make_executor()
        executor.register_coalesce(_settings(branches=["a", "b"], policy="require_all"), "node_1")
        result = executor.notify_branch_lost("merge", "row_1", "b", "error_routed")
        assert result is not None
        assert "branch_lost" in result.failure_reason

    def test_quorum_loss_makes_impossible_fails(self):
        executor, *_ = _make_executor()
        s = _settings(branches=["a", "b"], policy="quorum", quorum_count=2)
        executor.register_coalesce(s, "node_1")
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        # 2 branches, quorum=2, one lost -> max_possible=1 < quorum=2 -> fail
        result = executor.notify_branch_lost("merge", "row_1", "b", "error_routed")
        assert result is not None
        assert "quorum_impossible" in result.failure_reason

    def test_quorum_loss_still_possible_returns_none(self):
        executor, *_ = _make_executor()
        s = _settings(branches=["a", "b", "c"], policy="quorum", quorum_count=2)
        executor.register_coalesce(s, "node_1")
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        # Loss of c -> max_possible = 3-1=2 >= quorum_count=2. arrived=1 < 2. None.
        result = executor.notify_branch_lost("merge", "row_1", "c", "error_routed")
        assert result is None

    def test_best_effort_all_accounted_with_arrivals_merges(self):
        executor, *_ = _make_executor()
        s = _settings(branches=["a", "b"], policy="best_effort", timeout_seconds=60.0)
        executor.register_coalesce(s, "node_1")
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        result = executor.notify_branch_lost("merge", "row_1", "b", "error_routed")
        assert result is not None
        assert result.merged_token is not None

    def test_best_effort_all_lost_fails(self):
        executor, *_ = _make_executor()
        s = _settings(branches=["a", "b"], policy="best_effort", timeout_seconds=60.0)
        executor.register_coalesce(s, "node_1")
        # Both lost, no arrivals
        executor.notify_branch_lost("merge", "row_1", "a", "error_routed")
        result = executor.notify_branch_lost("merge", "row_1", "b", "error_routed")
        assert result is not None
        assert result.failure_reason == "all_branches_lost"

    def test_best_effort_still_waiting_returns_none(self):
        executor, *_ = _make_executor()
        s = _settings(branches=["a", "b", "c"], policy="best_effort", timeout_seconds=60.0)
        executor.register_coalesce(s, "node_1")
        # One lost, two remaining
        result = executor.notify_branch_lost("merge", "row_1", "a", "error_routed")
        assert result is None

    def test_first_policy_returns_none(self):
        executor, *_ = _make_executor()
        s = _settings(policy="first")
        executor.register_coalesce(s, "node_1")
        result = executor.notify_branch_lost("merge", "row_1", "a", "error_routed")
        assert result is None

    def test_branch_arrived_then_lost_raises(self):
        executor, *_ = _make_executor()
        executor.register_coalesce(_settings(), "node_1")
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        with pytest.raises(ValueError, match="already arrived"):
            executor.notify_branch_lost("merge", "row_1", "a", "error_routed")

    def test_branch_lost_before_any_arrivals(self):
        executor, *_ = _make_executor()
        s = _settings(branches=["a", "b", "c"], policy="best_effort", timeout_seconds=60.0)
        executor.register_coalesce(s, "node_1")
        # No accept() yet; notify loss creates pending entry
        result = executor.notify_branch_lost("merge", "row_1", "a", "upstream_error")
        # 3 branches, 1 lost, 0 arrived -> accounted=1 < 3 -> still waiting
        assert result is None
        # Verify pending entry was created
        assert ("merge", "row_1") in executor._pending

    def test_already_completed_returns_none(self):
        executor, *_ = _make_executor()
        executor.register_coalesce(_settings(), "node_1")
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        executor.accept(_make_token(branch_name="b", token_id="t2"), "merge")
        # Key is now completed
        result = executor.notify_branch_lost("merge", "row_1", "a", "error")
        assert result is None

    def test_duplicate_branch_loss_raises(self):
        """Reporting the same branch lost twice should work (updates reason)."""
        executor, *_ = _make_executor()
        s = _settings(branches=["a", "b", "c"], policy="best_effort", timeout_seconds=60.0)
        executor.register_coalesce(s, "node_1")
        executor.notify_branch_lost("merge", "row_1", "a", "first_reason")
        # Second loss notification for same branch updates the reason
        result = executor.notify_branch_lost("merge", "row_1", "b", "second_reason")
        # 3 branches, 2 lost, 0 arrived -> accounted=2 < 3 -> still waiting
        assert result is None


# ===========================================================================
# _mark_completed (FIFO eviction)
# ===========================================================================


class TestMarkCompleted:
    def test_constructor_max_completed_keys_configurable(self):
        executor, *_ = _make_executor(max_completed_keys=7)
        assert executor._max_completed_keys == 7

    def test_constructor_non_positive_max_completed_keys_raises(self):
        with pytest.raises(OrchestrationInvariantError, match="must be > 0"):
            _make_executor(max_completed_keys=0)

    def test_bounded_at_max(self):
        executor, *_ = _make_executor()
        executor._max_completed_keys = 5
        for i in range(10):
            executor._mark_completed(("c", f"row_{i}"))
        assert len(executor._completed_keys) == 5

    def test_eviction_emits_structured_warning(self):
        executor, *_ = _make_executor(max_completed_keys=2)
        with patch("elspeth.engine.coalesce_executor.slog.warning") as warning_mock:
            executor._mark_completed(("c", "row_0"))
            executor._mark_completed(("c", "row_1"))
            executor._mark_completed(("c", "row_2"))  # Triggers eviction

        warning_mock.assert_called_once()
        assert warning_mock.call_args.kwargs["max_completed_keys"] == 2
        assert warning_mock.call_args.kwargs["evicted_count"] == 1

    def test_fifo_eviction_oldest_removed(self):
        executor, *_ = _make_executor()
        executor._max_completed_keys = 3
        for i in range(5):
            executor._mark_completed(("c", f"row_{i}"))
        # Oldest (row_0, row_1) should be evicted; row_2, row_3, row_4 remain
        assert ("c", "row_0") not in executor._completed_keys
        assert ("c", "row_1") not in executor._completed_keys
        assert ("c", "row_2") in executor._completed_keys
        assert ("c", "row_3") in executor._completed_keys
        assert ("c", "row_4") in executor._completed_keys

    def test_idempotent_mark(self):
        """Marking the same key twice does not create duplicates."""
        executor, *_ = _make_executor()
        executor._mark_completed(("c", "row_1"))
        executor._mark_completed(("c", "row_1"))
        assert len(executor._completed_keys) == 1

    def test_default_max_is_10000(self):
        """Default max_completed_keys should be 10000."""
        executor, *_ = _make_executor()
        assert executor._max_completed_keys == 10000


# ===========================================================================
# contract handling during merge
# ===========================================================================


class TestContractHandling:
    def test_token_without_contract_raises(self):
        """Merge crashes if any token has None contract (upstream bug)."""
        executor, _, _, _ = _make_executor()
        executor.register_coalesce(_settings(), "node_1")
        contract = _make_contract()
        t1 = _make_token(branch_name="a", token_id="t1", data={"amount": 1}, contract=contract)
        # Simulate a bug: token with None contract
        bad_row = MagicMock(spec=PipelineRow)
        bad_row.contract = None
        bad_row.to_dict.return_value = {"amount": 2}
        t2 = TokenInfo(row_id="row_1", token_id="t2", row_data=bad_row, branch_name="b")
        executor.accept(t1, "merge")
        with pytest.raises(ValueError, match="has no contract"):
            executor.accept(t2, "merge")

    def test_union_contracts_merged(self):
        """Union merge should merge contracts from all branches."""
        executor, _, tm, _ = _make_executor()
        executor.register_coalesce(_settings(merge="union"), "node_1")
        c_a = _make_contract(
            fields=[
                make_field("x", python_type=int, required=True, source="declared"),
            ]
        )
        c_b = _make_contract(
            fields=[
                make_field("y", python_type=str, required=True, source="declared"),
            ]
        )
        t1 = _make_token(branch_name="a", token_id="t1", data={"x": 1}, contract=c_a)
        t2 = _make_token(branch_name="b", token_id="t2", data={"y": "hi"}, contract=c_b)
        executor.accept(t1, "merge")
        executor.accept(t2, "merge")
        merged_data = tm.coalesce_tokens.call_args.kwargs["merged_data"]
        mc = merged_data.contract
        assert mc.get_field("x") is not None
        assert mc.get_field("y") is not None

    def test_nested_merge_branch_key_contract(self):
        """Nested merge produces FIXED contract with branch keys typed as object."""
        executor, _, tm, _ = _make_executor()
        executor.register_coalesce(_settings(merge="nested"), "node_1")
        t1 = _make_token(branch_name="a", token_id="t1", data={"x": 1})
        t2 = _make_token(branch_name="b", token_id="t2", data={"y": 2})
        executor.accept(t1, "merge")
        executor.accept(t2, "merge")
        merged_data = tm.coalesce_tokens.call_args.kwargs["merged_data"]
        mc = merged_data.contract
        assert mc.mode == "FIXED"
        field_a = mc.get_field("a")
        field_b = mc.get_field("b")
        assert field_a is not None
        assert field_b is not None
        assert field_a.python_type is object
        assert field_b.python_type is object

    def test_select_merge_uses_selected_branch_contract(self):
        """Select merge uses the selected branch's contract, not a merge."""
        executor, _, tm, _ = _make_executor()
        s = _settings(merge="select", select_branch="a")
        executor.register_coalesce(s, "node_1")
        c_a = _make_contract(
            fields=[
                make_field("chosen", python_type=str, required=True, source="declared"),
            ]
        )
        c_b = _make_contract(
            fields=[
                make_field("ignored", python_type=int, required=True, source="declared"),
            ]
        )
        t1 = _make_token(branch_name="a", token_id="t1", data={"chosen": "yes"}, contract=c_a)
        t2 = _make_token(branch_name="b", token_id="t2", data={"ignored": 0}, contract=c_b)
        executor.accept(t1, "merge")
        executor.accept(t2, "merge")
        merged_data = tm.coalesce_tokens.call_args.kwargs["merged_data"]
        assert merged_data.contract is c_a

    def test_conflicting_contracts_raise_orchestration_error(self):
        """Union merge with conflicting field types raises OrchestrationInvariantError."""
        executor, _, _, _ = _make_executor()
        executor.register_coalesce(_settings(merge="union"), "node_1")
        c_a = _make_contract(
            fields=[
                make_field("value", python_type=int, required=True, source="declared"),
            ]
        )
        c_b = _make_contract(
            fields=[
                make_field("value", python_type=str, required=True, source="declared"),
            ]
        )
        t1 = _make_token(branch_name="a", token_id="t1", data={"value": 1}, contract=c_a)
        t2 = _make_token(branch_name="b", token_id="t2", data={"value": "x"}, contract=c_b)
        executor.accept(t1, "merge")
        with pytest.raises(OrchestrationInvariantError, match="Contract merge failed"):
            executor.accept(t2, "merge")


# ===========================================================================
# Audit trail details
# ===========================================================================


class TestAuditTrailDetails:
    def test_begin_node_state_captures_input_data(self):
        """begin_node_state should pass the token's row data as input_data."""
        executor, recorder, _, _ = _make_executor()
        executor.register_coalesce(_settings(), "node_1")
        t = _make_token(branch_name="a", token_id="t1", data={"amount": 42})
        executor.accept(t, "merge")
        kw = recorder.begin_node_state.call_args.kwargs
        assert kw["token_id"] == "t1"
        assert kw["run_id"] == "run_1"
        assert kw["step_index"] == 5
        assert kw["input_data"]["amount"] == 42

    def test_begin_node_state_uses_correct_node_id(self):
        """begin_node_state should use the node_id from register_coalesce."""
        executor, recorder, _, _ = _make_executor()
        executor.register_coalesce(_settings(), "coalesce_node_42")
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        kw = recorder.begin_node_state.call_args.kwargs
        assert kw["node_id"] == "coalesce_node_42"

    def test_complete_node_state_duration_ms(self):
        """Completed node states should have a non-negative duration_ms."""
        executor, recorder, _, clock = _make_executor()
        executor.register_coalesce(_settings(), "node_1")
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        clock.advance(0.5)
        executor.accept(_make_token(branch_name="b", token_id="t2"), "merge")
        calls = recorder.complete_node_state.call_args_list
        durations = [c.kwargs["duration_ms"] for c in calls if c.kwargs.get("status") == NodeStateStatus.COMPLETED]
        assert len(durations) == 2
        assert all(d >= 0 for d in durations)
        # At least one should have waited ~500ms
        assert any(d >= 400 for d in durations)

    def test_complete_node_state_includes_coalesce_context(self):
        """Completed node states should include coalesce_context in context_after."""
        executor, recorder, _, _ = _make_executor()
        executor.register_coalesce(_settings(), "node_1")
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        executor.accept(_make_token(branch_name="b", token_id="t2"), "merge")
        for c in recorder.complete_node_state.call_args_list:
            if c.kwargs.get("status") == NodeStateStatus.COMPLETED:
                ctx = c.kwargs.get("context_after", {})
                assert "coalesce_context" in ctx

    def test_complete_node_state_output_data_merged_into(self):
        """Completed node states have output_data with merged_into token ID."""
        executor, recorder, _, _ = _make_executor()
        executor.register_coalesce(_settings(), "node_1")
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        executor.accept(_make_token(branch_name="b", token_id="t2"), "merge")
        for c in recorder.complete_node_state.call_args_list:
            if c.kwargs.get("status") == NodeStateStatus.COMPLETED:
                output = c.kwargs.get("output_data", {})
                assert "merged_into" in output
                assert output["merged_into"].startswith("merged_")

    def test_record_token_outcome_has_join_group_id(self):
        """Token outcomes should include join_group_id from merged token."""
        executor, recorder, _, _ = _make_executor()
        executor.register_coalesce(_settings(), "node_1")
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        executor.accept(_make_token(branch_name="b", token_id="t2"), "merge")
        for c in recorder.record_token_outcome.call_args_list:
            assert c.kwargs["join_group_id"] is not None

    def test_record_token_outcome_has_correct_token_ids(self):
        """Token outcomes should reference the original consumed token IDs."""
        executor, recorder, _, _ = _make_executor()
        executor.register_coalesce(_settings(), "node_1")
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        executor.accept(_make_token(branch_name="b", token_id="t2"), "merge")
        token_ids = {c.kwargs["token_id"] for c in recorder.record_token_outcome.call_args_list}
        assert token_ids == {"t1", "t2"}

    def test_merge_metadata_arrival_order(self):
        """Coalesce metadata should include arrival_order with offset_ms."""
        executor, _, _, clock = _make_executor()
        executor.register_coalesce(_settings(), "node_1")
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        clock.advance(0.2)
        o = executor.accept(_make_token(branch_name="b", token_id="t2"), "merge")
        arrival_order = o.coalesce_metadata["arrival_order"]
        assert len(arrival_order) == 2
        assert arrival_order[0]["branch"] == "a"
        assert arrival_order[0]["arrival_offset_ms"] == pytest.approx(0.0)
        assert arrival_order[1]["branch"] == "b"
        assert arrival_order[1]["arrival_offset_ms"] == pytest.approx(200.0)

    def test_merge_metadata_wait_duration(self):
        """Coalesce metadata should include total wait_duration_ms."""
        executor, _, _, clock = _make_executor()
        executor.register_coalesce(_settings(), "node_1")
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        clock.advance(1.5)
        o = executor.accept(_make_token(branch_name="b", token_id="t2"), "merge")
        assert o.coalesce_metadata["wait_duration_ms"] == pytest.approx(1500.0)

    def test_merge_metadata_branches_lost_empty_when_none_lost(self):
        """Branches_lost in metadata should be empty dict when all arrived."""
        executor, _, _, _ = _make_executor()
        executor.register_coalesce(_settings(), "node_1")
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        o = executor.accept(_make_token(branch_name="b", token_id="t2"), "merge")
        assert o.coalesce_metadata["branches_lost"] == {}


# ===========================================================================
# Default clock usage
# ===========================================================================


class TestDefaultClock:
    def test_uses_default_clock_when_none(self):
        """Constructor should use DEFAULT_CLOCK when clock=None."""
        from elspeth.engine.clock import DEFAULT_CLOCK

        recorder = MagicMock()
        recorder.begin_node_state.side_effect = lambda **kw: Mock(state_id="s1")
        executor = CoalesceExecutor(recorder, MagicMock(), MagicMock(), "run_1", step_resolver=lambda n: 0, clock=None)
        assert executor._clock is DEFAULT_CLOCK

    def test_uses_injected_clock(self):
        clock = MockClock(start=42.0)
        recorder = MagicMock()
        recorder.begin_node_state.side_effect = lambda **kw: Mock(state_id="s1")
        executor = CoalesceExecutor(recorder, MagicMock(), MagicMock(), "run_1", step_resolver=lambda n: 0, clock=clock)
        assert executor._clock is clock


# ===========================================================================
# Multi-row isolation
# ===========================================================================


class TestMultiRowIsolation:
    def test_different_rows_independent(self):
        """Tokens for different row_ids are tracked independently."""
        executor, _, _, _ = _make_executor()
        executor.register_coalesce(_settings(), "node_1")
        o1 = executor.accept(_make_token(row_id="r1", branch_name="a", token_id="t1"), "merge")
        o2 = executor.accept(_make_token(row_id="r2", branch_name="a", token_id="t2"), "merge")
        assert o1.held is True
        assert o2.held is True
        # Complete r1
        o3 = executor.accept(_make_token(row_id="r1", branch_name="b", token_id="t3"), "merge")
        assert o3.held is False
        assert o3.merged_token is not None
        # r2 still pending
        o4 = executor.accept(_make_token(row_id="r2", branch_name="b", token_id="t4"), "merge")
        assert o4.held is False
        assert o4.merged_token is not None

    def test_different_coalesce_points_independent(self):
        """Separate coalesce points do not interfere with each other."""
        executor, _, _, _ = _make_executor()
        s1 = _settings(name="m1", branches=["a", "b"])
        s2 = _settings(name="m2", branches=["x", "y"])
        executor.register_coalesce(s1, "n1")
        executor.register_coalesce(s2, "n2")
        o1 = executor.accept(_make_token(branch_name="a", token_id="t1"), "m1")
        o2 = executor.accept(_make_token(branch_name="x", token_id="t2"), "m2")
        assert o1.held is True
        assert o2.held is True


# ===========================================================================
# Failure outcomes via _fail_pending
# ===========================================================================


class TestFailPendingDetails:
    def test_failure_records_failed_node_states(self):
        executor, recorder, _, clock = _make_executor()
        s = _settings(policy="require_all", timeout_seconds=5.0)
        executor.register_coalesce(s, "node_1")
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        clock.advance(6.0)
        executor.check_timeouts("merge")
        # Check that complete_node_state was called with FAILED
        fail_calls = [c for c in recorder.complete_node_state.call_args_list if c.kwargs.get("status") == NodeStateStatus.FAILED]
        assert len(fail_calls) == 1

    def test_failure_records_token_outcomes_failed(self):
        executor, recorder, _, clock = _make_executor()
        s = _settings(policy="require_all", timeout_seconds=5.0)
        executor.register_coalesce(s, "node_1")
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        clock.advance(6.0)
        executor.check_timeouts("merge")
        outcome_calls = recorder.record_token_outcome.call_args_list
        assert len(outcome_calls) == 1
        assert outcome_calls[0].kwargs["outcome"] == RowOutcome.FAILED

    def test_failure_metadata_includes_policy(self):
        executor, _, _, clock = _make_executor()
        s = _settings(policy="require_all", timeout_seconds=5.0)
        executor.register_coalesce(s, "node_1")
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        clock.advance(6.0)
        results = executor.check_timeouts("merge")
        md = results[0].coalesce_metadata
        assert md["policy"] == "require_all"
        assert set(md["expected_branches"]) == {"a", "b"}

    def test_failure_removes_pending_entry(self):
        executor, _, _, clock = _make_executor()
        s = _settings(policy="require_all", timeout_seconds=5.0)
        executor.register_coalesce(s, "node_1")
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        assert ("merge", "row_1") in executor._pending
        clock.advance(6.0)
        executor.check_timeouts("merge")
        assert ("merge", "row_1") not in executor._pending

    def test_failure_marks_key_completed(self):
        executor, _, _, clock = _make_executor()
        s = _settings(policy="require_all", timeout_seconds=5.0)
        executor.register_coalesce(s, "node_1")
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        clock.advance(6.0)
        executor.check_timeouts("merge")
        assert ("merge", "row_1") in executor._completed_keys

    def test_failure_metadata_includes_lost_branches(self):
        executor, *_ = _make_executor()
        s = _settings(branches=["a", "b"], policy="require_all")
        executor.register_coalesce(s, "node_1")
        # Loss of b triggers require_all failure
        result = executor.notify_branch_lost("merge", "row_1", "b", "upstream_fail")
        assert "branches_lost" in result.coalesce_metadata
        assert "b" in result.coalesce_metadata["branches_lost"]

    def test_failure_metadata_includes_quorum_required(self):
        executor, *_ = _make_executor()
        s = _settings(branches=["a", "b", "c"], policy="quorum", quorum_count=3)
        executor.register_coalesce(s, "node_1")
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        # Loss of b -> max_possible=2 < quorum=3 -> fail
        result = executor.notify_branch_lost("merge", "row_1", "b", "error")
        assert result.coalesce_metadata["quorum_required"] == 3

    def test_require_all_timeout_metadata_has_timeout_seconds(self):
        executor, _, _, clock = _make_executor()
        s = _settings(policy="require_all", timeout_seconds=8.0)
        executor.register_coalesce(s, "node_1")
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        clock.advance(9.0)
        results = executor.check_timeouts("merge")
        assert results[0].coalesce_metadata["timeout_seconds"] == 8.0

    def test_failure_error_hash_is_deterministic(self):
        """The error_hash recorded for failed tokens should be consistent."""
        executor, recorder, _, clock = _make_executor()
        s = _settings(policy="require_all", timeout_seconds=5.0)
        executor.register_coalesce(s, "node_1")
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        clock.advance(6.0)
        executor.check_timeouts("merge")
        # record_token_outcome should have been called with an error_hash
        kw = recorder.record_token_outcome.call_args.kwargs
        assert "error_hash" in kw
        assert isinstance(kw["error_hash"], str)
        assert len(kw["error_hash"]) == 16  # sha256[:16]

    def test_failure_branches_arrived_in_metadata(self):
        """Failure metadata includes which branches had actually arrived."""
        executor, *_ = _make_executor()
        s = _settings(branches=["a", "b", "c"], policy="require_all", timeout_seconds=5.0)
        executor.register_coalesce(s, "node_1")
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        executor.accept(_make_token(branch_name="b", token_id="t2"), "merge")
        # require_all needs c, loss of c -> fail
        result = executor.notify_branch_lost("merge", "row_1", "c", "error")
        assert set(result.coalesce_metadata["branches_arrived"]) == {"a", "b"}
