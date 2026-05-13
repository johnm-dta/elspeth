# tests/unit/engine/test_coalesce_executor.py
"""Comprehensive unit tests for CoalesceExecutor.

Tests merge policies (require_all, first, quorum, best_effort),
merge strategies (union, nested, select), timeout handling,
flush behaviour, branch loss notifications, late arrivals,
and audit trail recording.
"""

from __future__ import annotations

import itertools
from typing import Any, Literal
from unittest.mock import MagicMock, Mock
from uuid import uuid4

import pytest

from elspeth.contracts import TokenInfo
from elspeth.contracts.coalesce_enums import CoalescePolicy, MergeStrategy
from elspeth.contracts.enums import NodeStateStatus, TerminalOutcome, TerminalPath
from elspeth.contracts.errors import (
    AuditIntegrityError,
    CoalesceCollisionError,
    OrchestrationInvariantError,
)
from elspeth.contracts.schema_contract import PipelineRow, SchemaContract
from elspeth.contracts.types import NodeID
from elspeth.core.config import CoalesceSettings
from elspeth.core.landscape.data_flow_repository import DataFlowRepository
from elspeth.core.landscape.execution_repository import ExecutionRepository
from elspeth.engine.clock import MockClock
from elspeth.engine.coalesce_executor import CoalesceExecutor, CoalesceOutcome, _BranchEntry, _PendingCoalesce
from elspeth.testing import make_field, make_row

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _TestCoalesceExecutor(CoalesceExecutor):
    """Test wrapper that auto-provides output_schema for union merge.

    Production code computes output_schema via the DAG builder's merge_union_fields().
    Tests bypass the DAG builder, so this wrapper provides an OBSERVED-mode schema
    by default, matching the contract mode used by test fixtures.

    This eliminates the need for the fallback path in CoalesceExecutor._execute_merge().
    """

    def register_coalesce(
        self,
        settings: CoalesceSettings,
        node_id: NodeID,
        branch_schemas: dict[str, tuple[str, ...]] | None = None,
        output_schema: SchemaContract | None = None,
    ) -> None:
        if settings.on_success is None:
            settings = settings.model_copy(update={"on_success": "default"})
        # Auto-provide OBSERVED schema for union merge if not specified.
        # OBSERVED mode = "infer schema from data" — matches test fixture contracts.
        if output_schema is None and settings.merge == "union":
            output_schema = SchemaContract(mode="OBSERVED", fields=(), locked=False)
        super().register_coalesce(settings, node_id, branch_schemas, output_schema)


_state_counter = itertools.count(1)


def _next_state_id() -> str:
    return f"state_{next(_state_counter):04d}"


def _make_contract(
    fields: list[Any] | None = None,
    *,
    mode: Literal["FIXED", "FLEXIBLE", "OBSERVED"],
) -> SchemaContract:
    """Create a schema contract for testing with explicit mode.

    Args:
        fields: List of FieldContract instances. Defaults to a single 'amount' field.
        mode: Schema enforcement mode (required, no default).
    """
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
    return SchemaContract(fields=tuple(fields), mode=mode, locked=True)


def _make_token(
    row_id: str = "row_1",
    token_id: str = "tok_1",
    branch_name: str = "branch_a",
    data: dict[str, Any] | None = None,
    contract: SchemaContract | None = None,
) -> TokenInfo:
    """Build a TokenInfo suitable for coalesce testing."""
    if data is None:
        data = {"amount": 100}
    if contract is None:
        contract = _make_contract(mode="FLEXIBLE")
    row_data = make_row(data, contract=contract)
    return TokenInfo(
        row_id=row_id,
        token_id=token_id,
        row_data=row_data,
        branch_name=branch_name,
    )


def _make_executor(
    clock: MockClock | None = None, max_completed_keys: int = 10000
) -> tuple[CoalesceExecutor, MagicMock, MagicMock, MagicMock, MockClock]:
    """Build a CoalesceExecutor with mocked dependencies.

    Returns (executor, execution, data_flow, token_manager, clock).
    """
    execution = MagicMock(spec=ExecutionRepository)
    execution.begin_node_state.side_effect = lambda **kw: Mock(state_id=_next_state_id())
    # Default: Landscape returns no completed coalesces (unit tests don't have a real DB).
    # Tests that exercise Landscape-based restoration override this per-test.
    execution.get_completed_row_ids_for_nodes.return_value = set()
    data_flow = MagicMock(spec=DataFlowRepository)
    span_factory = MagicMock()
    token_manager = MagicMock()

    def coalesce_tokens_impl(parents, merged_data, node_id, run_id):
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

    executor = _TestCoalesceExecutor(
        execution,
        span_factory,
        token_manager,
        "run_1",
        step_resolver=step_resolver,
        clock=clock,
        max_completed_keys=max_completed_keys,
        data_flow=data_flow,
    )
    return executor, execution, data_flow, token_manager, clock


def _make_raw_executor(
    clock: MockClock | None = None, max_completed_keys: int = 10000
) -> tuple[CoalesceExecutor, MagicMock, MagicMock, MagicMock, MockClock]:
    """Build the production CoalesceExecutor without the test on_success shim."""
    execution = MagicMock(spec=ExecutionRepository)
    execution.begin_node_state.side_effect = lambda **kw: Mock(state_id=_next_state_id())
    execution.get_completed_row_ids_for_nodes.return_value = set()
    data_flow = MagicMock(spec=DataFlowRepository)
    span_factory = MagicMock()
    token_manager = MagicMock()

    def coalesce_tokens_impl(parents, merged_data, node_id, run_id):
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
        execution,
        span_factory,
        token_manager,
        "run_1",
        step_resolver=step_resolver,
        clock=clock,
        max_completed_keys=max_completed_keys,
        data_flow=data_flow,
    )
    return executor, execution, data_flow, token_manager, clock


def _settings(
    name: str = "merge",
    branches: list[str] | None = None,
    policy: str = "require_all",
    merge: str = "union",
    timeout_seconds: float | None = None,
    quorum_count: int | None = None,
    select_branch: str | None = None,
    union_collision_policy: str = "last_wins",
) -> CoalesceSettings:
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
        union_collision_policy=union_collision_policy,
    )


# ===========================================================================
# CoalesceOutcome dataclass
# ===========================================================================


class TestCoalesceOutcome:
    def test_defaults(self):
        outcome = CoalesceOutcome(held=True)
        assert outcome.held is True
        assert outcome.merged_token is None
        assert outcome.consumed_tokens == ()
        assert outcome.coalesce_metadata is None
        assert outcome.failure_reason is None
        assert outcome.coalesce_name is None
        assert outcome.outcomes_recorded is False

    def test_merged_outcome(self):
        from elspeth.contracts.coalesce_metadata import CoalesceMetadata

        token = _make_token()
        metadata = CoalesceMetadata.for_late_arrival(policy=CoalescePolicy.REQUIRE_ALL, reason="test")
        outcome = CoalesceOutcome(
            held=False,
            merged_token=token,
            consumed_tokens=(token,),
            coalesce_metadata=metadata,
            coalesce_name="merge",
        )
        assert outcome.held is False
        assert outcome.merged_token is token
        assert outcome.consumed_tokens == (token,)
        assert outcome.coalesce_metadata.policy == CoalescePolicy.REQUIRE_ALL
        assert outcome.failure_reason is None
        assert outcome.coalesce_name == "merge"
        assert outcome.outcomes_recorded is False

    def test_failure_outcome(self):
        from elspeth.contracts.coalesce_metadata import CoalesceMetadata

        token = _make_token()
        metadata = CoalesceMetadata.for_late_arrival(policy=CoalescePolicy.REQUIRE_ALL, reason="test")
        outcome = CoalesceOutcome(
            held=False,
            consumed_tokens=(token,),
            coalesce_metadata=metadata,
            failure_reason="late_arrival_after_merge",
            coalesce_name="merge",
            outcomes_recorded=True,
        )
        assert outcome.held is False
        assert outcome.merged_token is None
        assert outcome.consumed_tokens == (token,)
        assert outcome.failure_reason == "late_arrival_after_merge"
        assert outcome.outcomes_recorded is True

    def test_invalid_held_with_merged_token(self):
        from elspeth.contracts.errors import OrchestrationInvariantError

        token = _make_token()
        with pytest.raises(OrchestrationInvariantError, match="held=True but merged_token"):
            CoalesceOutcome(held=True, merged_token=token)

    def test_invalid_merged_and_failed(self):
        from elspeth.contracts.errors import OrchestrationInvariantError

        token = _make_token()
        with pytest.raises(OrchestrationInvariantError, match="both merged_token and failure_reason"):
            CoalesceOutcome(held=False, merged_token=token, failure_reason="some_reason")


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
        with pytest.raises(OrchestrationInvariantError, match="not registered"):
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
        with pytest.raises(OrchestrationInvariantError, match="no branch_name"):
            executor.accept(token, "merge")

    def test_unexpected_branch_raises(self):
        executor, *_ = _make_executor()
        executor.register_coalesce(_settings(branches=["a", "b"]), "node_1")
        token = _make_token(branch_name="c")
        with pytest.raises(OrchestrationInvariantError, match="not in expected branches"):
            executor.accept(token, "merge")

    def test_duplicate_arrival_raises(self):
        executor, *_ = _make_executor()
        executor.register_coalesce(_settings(), "node_1")
        t1 = _make_token(branch_name="a", token_id="tok_1")
        t2 = _make_token(branch_name="a", token_id="tok_2")
        executor.accept(t1, "merge")
        with pytest.raises(OrchestrationInvariantError, match="Duplicate arrival"):
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
        executor, execution, data_flow, tm, clock = _make_executor()
        s = _settings(branches=branches, policy="require_all")
        executor.register_coalesce(s, "node_1")
        return executor, execution, data_flow, tm, clock

    def test_two_branches_first_held_second_merges(self):
        executor, _, _, _, _ = self._setup()
        o1 = executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        o2 = executor.accept(_make_token(branch_name="b", token_id="t2"), "merge")
        assert o1.held is True
        assert o2.held is False
        assert o2.merged_token is not None

    def test_three_branches(self):
        executor, _, _, _, _ = self._setup(branches=["a", "b", "c"])
        o1 = executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        o2 = executor.accept(_make_token(branch_name="b", token_id="t2"), "merge")
        o3 = executor.accept(_make_token(branch_name="c", token_id="t3"), "merge")
        assert o1.held is True
        assert o2.held is True
        assert o3.held is False
        assert o3.merged_token is not None

    def test_merged_token_in_outcome(self):
        executor, _, _, _, _ = self._setup()
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        o = executor.accept(_make_token(branch_name="b", token_id="t2"), "merge")
        assert o.merged_token is not None
        assert o.merged_token.row_id == "row_1"
        assert o.merged_token.join_group_id is not None

    def test_consumed_tokens_list(self):
        executor, _, _, _, _ = self._setup()
        t1 = _make_token(branch_name="a", token_id="t1")
        t2 = _make_token(branch_name="b", token_id="t2")
        executor.accept(t1, "merge")
        o = executor.accept(t2, "merge")
        consumed_ids = {t.token_id for t in o.consumed_tokens}
        assert consumed_ids == {"t1", "t2"}

    def test_coalesce_metadata(self):
        executor, _, _, _, _ = self._setup()
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        o = executor.accept(_make_token(branch_name="b", token_id="t2"), "merge")
        md = o.coalesce_metadata
        assert md.policy == CoalescePolicy.REQUIRE_ALL
        assert md.merge_strategy == MergeStrategy.UNION
        assert set(md.expected_branches) == {"a", "b"}
        assert set(md.branches_arrived) == {"a", "b"}

    def test_audit_begin_node_state_for_each_token(self):
        executor, execution, _, _, _ = self._setup()
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        executor.accept(_make_token(branch_name="b", token_id="t2"), "merge")
        # begin_node_state called once per accepted token
        assert execution.begin_node_state.call_count == 2

    def test_audit_complete_node_state_completed(self):
        executor, execution, _, _, _ = self._setup()
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        executor.accept(_make_token(branch_name="b", token_id="t2"), "merge")
        # On merge, each consumed token's state is completed with COMPLETED
        completed_calls = [c for c in execution.complete_node_state.call_args_list if c.kwargs.get("status") == NodeStateStatus.COMPLETED]
        assert len(completed_calls) == 2

    def test_audit_record_token_outcome_coalesced(self):
        executor, _, data_flow, _, _ = self._setup()
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        executor.accept(_make_token(branch_name="b", token_id="t2"), "merge")
        outcome_calls = data_flow.record_token_outcome.call_args_list
        assert len(outcome_calls) == 2
        for c in outcome_calls:
            assert c.kwargs["outcome"] == TerminalOutcome.SUCCESS
            assert c.kwargs["path"] == TerminalPath.COALESCED

    def test_non_terminal_coalesce_records_absorbed_branches_without_sink_witness(self):
        """Downstream coalesce flows have no terminal sink witness at merge time."""
        executor, _, data_flow, _, _ = _make_raw_executor()
        settings = _settings(branches=["a", "b"], policy="require_all")
        executor.register_coalesce(
            settings,
            "node_1",
            output_schema=SchemaContract(mode="OBSERVED", fields=(), locked=False),
        )

        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        outcome = executor.accept(_make_token(branch_name="b", token_id="t2"), "merge")

        assert outcome.merged_token is not None
        outcome_calls = data_flow.record_token_outcome.call_args_list
        assert len(outcome_calls) == 2
        for c in outcome_calls:
            assert c.kwargs["outcome"] == TerminalOutcome.SUCCESS
            assert c.kwargs["path"] == TerminalPath.COALESCED
            assert c.kwargs["sink_name"] is None
            assert c.kwargs["join_group_id"] == outcome.merged_token.join_group_id

    def test_terminal_coalesce_does_not_tag_absorbed_branches_with_sink_witness(self):
        """Only the merged token's later sink write should carry the sink discriminator."""
        executor, _, data_flow, _, _ = _make_raw_executor()
        settings = _settings(branches=["a", "b"], policy="require_all").model_copy(update={"on_success": "output"})
        executor.register_coalesce(
            settings,
            "node_1",
            output_schema=SchemaContract(mode="OBSERVED", fields=(), locked=False),
        )

        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        outcome = executor.accept(_make_token(branch_name="b", token_id="t2"), "merge")

        assert outcome.merged_token is not None
        outcome_calls = data_flow.record_token_outcome.call_args_list
        assert len(outcome_calls) == 2
        for c in outcome_calls:
            assert c.kwargs["outcome"] == TerminalOutcome.SUCCESS
            assert c.kwargs["path"] == TerminalPath.COALESCED
            assert c.kwargs["sink_name"] is None
            assert c.kwargs["join_group_id"] == outcome.merged_token.join_group_id

    def test_token_manager_coalesce_tokens_called(self):
        executor, _, _, tm, _ = self._setup()
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
        executor, _, _, _, _ = _make_executor()
        s = _settings(policy="first")
        executor.register_coalesce(s, "node_1")
        t = _make_token(branch_name="a", token_id="t1")
        o = executor.accept(t, "merge")
        assert o.held is False
        assert o.merged_token is not None

    def test_only_one_consumed_token(self):
        executor, _, _, _, _ = _make_executor()
        s = _settings(policy="first")
        executor.register_coalesce(s, "node_1")
        t = _make_token(branch_name="a", token_id="t1")
        o = executor.accept(t, "merge")
        assert len(o.consumed_tokens) == 1
        assert o.consumed_tokens[0].token_id == "t1"

    def test_second_arrival_is_late(self):
        executor, _, _, _, _ = _make_executor()
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
        executor, _, _, _, _ = _make_executor()
        s = _settings(branches=["a", "b", "c"], policy="quorum", quorum_count=2)
        executor.register_coalesce(s, "node_1")
        o1 = executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        o2 = executor.accept(_make_token(branch_name="b", token_id="t2"), "merge")
        assert o1.held is True
        assert o2.held is False
        assert o2.merged_token is not None

    def test_third_arrival_is_late(self):
        executor, _, _, _, _ = _make_executor()
        s = _settings(branches=["a", "b", "c"], policy="quorum", quorum_count=2)
        executor.register_coalesce(s, "node_1")
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        executor.accept(_make_token(branch_name="b", token_id="t2"), "merge")
        o = executor.accept(_make_token(branch_name="c", token_id="t3"), "merge")
        assert o.failure_reason == "late_arrival_after_merge"

    def test_quorum_of_one_triggers_like_first(self):
        executor, _, _, _, _ = _make_executor()
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
        executor, _, _, _, _ = _make_executor()
        s = _settings(policy="best_effort", timeout_seconds=60.0)
        executor.register_coalesce(s, "node_1")
        o = executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        assert o.held is True

    def test_merges_when_all_accounted_for(self):
        """best_effort merges when arrived + lost >= expected."""
        executor, _, _, _, _ = _make_executor()
        s = _settings(branches=["a", "b"], policy="best_effort", timeout_seconds=60.0)
        executor.register_coalesce(s, "node_1")
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        # Notify branch b lost
        result = executor.notify_branch_lost("merge", "row_1", "b", "error_routed")
        assert result is not None
        assert result.merged_token is not None

    def test_all_branches_arrived_triggers_merge(self):
        """best_effort merges immediately when all branches arrive."""
        executor, _, _, _, _ = _make_executor()
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
        executor, _, _, _, _ = _make_executor()
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
        executor, execution, data_flow, _, _ = _make_executor()
        executor.register_coalesce(_settings(), "node_1")
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        executor.accept(_make_token(branch_name="b", token_id="t2"), "merge")
        execution.reset_mock()
        data_flow.reset_mock()
        late = _make_token(branch_name="a", token_id="t_late", row_id="row_1")
        executor.accept(late, "merge")

        # Should begin + complete with FAILED
        execution.begin_node_state.assert_called_once()
        execution.complete_node_state.assert_called_once()
        fail_call = execution.complete_node_state.call_args
        assert fail_call.kwargs["status"] == NodeStateStatus.FAILED

        # Should record a terminal FAILED token outcome immediately
        data_flow.record_token_outcome.assert_called_once()
        outcome_call = data_flow.record_token_outcome.call_args
        assert outcome_call.kwargs["ref"].token_id == "t_late"
        assert outcome_call.kwargs["outcome"] == TerminalOutcome.FAILURE
        assert outcome_call.kwargs["path"] == TerminalPath.UNROUTED
        assert isinstance(outcome_call.kwargs["error_hash"], str)
        assert len(outcome_call.kwargs["error_hash"]) == 16

    def test_late_arrival_consumed_tokens(self):
        executor, _, _, _, _ = _make_executor()
        executor.register_coalesce(_settings(), "node_1")
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        executor.accept(_make_token(branch_name="b", token_id="t2"), "merge")
        late = _make_token(branch_name="a", token_id="t_late", row_id="row_1")
        o = executor.accept(late, "merge")
        assert len(o.consumed_tokens) == 1
        assert o.consumed_tokens[0].token_id == "t_late"

    def test_late_arrival_metadata_has_policy(self):
        executor, _, _, _, _ = _make_executor()
        executor.register_coalesce(_settings(), "node_1")
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        executor.accept(_make_token(branch_name="b", token_id="t2"), "merge")
        late = _make_token(branch_name="a", token_id="t_late", row_id="row_1")
        o = executor.accept(late, "merge")
        assert o.coalesce_metadata.policy == CoalescePolicy.REQUIRE_ALL
        assert o.coalesce_metadata.reason is not None


# ===========================================================================
# union merge
# ===========================================================================


class TestUnionMerge:
    def test_fields_from_both_branches(self):
        executor, _, _, tm, _ = _make_executor()
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
        executor, _, _, tm, _ = _make_executor()
        executor.register_coalesce(_settings(branches=["a", "b"], merge="union"), "node_1")
        t1 = _make_token(branch_name="a", token_id="t1", data={"shared": "from_a"})
        t2 = _make_token(branch_name="b", token_id="t2", data={"shared": "from_b"})
        executor.accept(t1, "merge")
        executor.accept(t2, "merge")
        merged_data = tm.coalesce_tokens.call_args.kwargs["merged_data"]
        assert merged_data.to_dict()["shared"] == "from_b"

    def test_collision_metadata_recorded(self):
        executor, _, _, _, _ = _make_executor()
        executor.register_coalesce(_settings(branches=["a", "b"], merge="union"), "node_1")
        t1 = _make_token(branch_name="a", token_id="t1", data={"shared": "from_a"})
        t2 = _make_token(branch_name="b", token_id="t2", data={"shared": "from_b"})
        executor.accept(t1, "merge")
        o = executor.accept(t2, "merge")
        assert o.coalesce_metadata.union_field_collisions is not None
        assert "shared" in o.coalesce_metadata.union_field_collisions

    def test_no_collisions_no_collision_metadata(self):
        """When there are no field collisions, collision metadata should be absent."""
        executor, _, _, _, _ = _make_executor()
        executor.register_coalesce(_settings(merge="union"), "node_1")
        t1 = _make_token(branch_name="a", token_id="t1", data={"x": 1})
        t2 = _make_token(branch_name="b", token_id="t2", data={"y": 2})
        executor.accept(t1, "merge")
        o = executor.accept(t2, "merge")
        assert o.coalesce_metadata.union_field_collisions is None

    def test_collision_tracks_all_contributing_branches(self):
        """Collision metadata lists all branches that contributed the same field."""
        executor, _, _, _, _ = _make_executor()
        s = _settings(branches=["a", "b", "c"], merge="union", policy="require_all")
        executor.register_coalesce(s, "node_1")
        t1 = _make_token(branch_name="a", token_id="t1", data={"f": 1})
        t2 = _make_token(branch_name="b", token_id="t2", data={"f": 2})
        t3 = _make_token(branch_name="c", token_id="t3", data={"f": 3})
        executor.accept(t1, "merge")
        executor.accept(t2, "merge")
        o = executor.accept(t3, "merge")
        collision_branches = o.coalesce_metadata.union_field_collisions["f"]
        assert "a" in collision_branches
        assert "b" in collision_branches
        assert "c" in collision_branches

    # ------------------------------------------------------------------
    # Field-level provenance: field_origins + collision_values
    # ------------------------------------------------------------------

    def test_union_merge_records_field_origins_for_all_fields(self):
        """Every field produced by a union merge must be tagged with its origin branch."""
        executor, _, _, _, _ = _make_executor()
        s = _settings(branches=["a", "b", "c"], merge="union", policy="require_all")
        executor.register_coalesce(s, "node_1")
        t1 = _make_token(branch_name="a", token_id="t1", data={"x": 1})
        t2 = _make_token(branch_name="b", token_id="t2", data={"y": 2})
        t3 = _make_token(branch_name="c", token_id="t3", data={"z": 3})
        executor.accept(t1, "merge")
        executor.accept(t2, "merge")
        o = executor.accept(t3, "merge")
        origins = o.coalesce_metadata.union_field_origins
        assert origins is not None
        assert origins["x"] == "a"
        assert origins["y"] == "b"
        assert origins["z"] == "c"
        # No collisions -> collision_values should be absent (None).
        assert o.coalesce_metadata.union_field_collision_values is None

    def test_union_merge_collision_preserves_overwritten_value(self):
        """When two branches collide, both (branch, value) entries must be recorded."""
        executor, _, _, _, _ = _make_executor()
        executor.register_coalesce(_settings(branches=["a", "b"], merge="union"), "node_1")
        t1 = _make_token(branch_name="a", token_id="t1", data={"shared": "from_a"})
        t2 = _make_token(branch_name="b", token_id="t2", data={"shared": "from_b"})
        executor.accept(t1, "merge")
        o = executor.accept(t2, "merge")
        collision_values = o.coalesce_metadata.union_field_collision_values
        assert collision_values is not None
        assert list(collision_values["shared"]) == [("a", "from_a"), ("b", "from_b")]
        # Default last_wins: winner in merged data is the last branch.
        assert o.coalesce_metadata.union_field_origins["shared"] == "b"

    def test_union_merge_three_way_collision_preserves_all_values(self):
        """Three-way collisions preserve every (branch, value) entry in declaration order."""
        executor, _, _, _, _ = _make_executor()
        s = _settings(branches=["a", "b", "c"], merge="union", policy="require_all")
        executor.register_coalesce(s, "node_1")
        t1 = _make_token(branch_name="a", token_id="t1", data={"f": "va"})
        t2 = _make_token(branch_name="b", token_id="t2", data={"f": "vb"})
        t3 = _make_token(branch_name="c", token_id="t3", data={"f": "vc"})
        executor.accept(t1, "merge")
        executor.accept(t2, "merge")
        o = executor.accept(t3, "merge")
        entries = list(o.coalesce_metadata.union_field_collision_values["f"])
        assert entries == [("a", "va"), ("b", "vb"), ("c", "vc")]

    def test_union_merge_field_origins_flow_to_metadata(self):
        """field_origins returned from _merge_data must be reflected in CoalesceMetadata."""
        executor, _, _, _, _ = _make_executor()
        executor.register_coalesce(_settings(branches=["a", "b"], merge="union"), "node_1")
        t1 = _make_token(branch_name="a", token_id="t1", data={"x": 1, "y": 2})
        t2 = _make_token(branch_name="b", token_id="t2", data={"z": 3})
        executor.accept(t1, "merge")
        o = executor.accept(t2, "merge")
        origins = o.coalesce_metadata.union_field_origins
        assert origins == {"x": "a", "y": "a", "z": "b"}

    # ------------------------------------------------------------------
    # union_collision_policy=last_wins (default)
    # ------------------------------------------------------------------

    def test_union_collision_policy_last_wins_is_default(self):
        """CoalesceSettings default union_collision_policy is last_wins."""
        s = _settings(branches=["a", "b"], merge="union")
        assert s.union_collision_policy == "last_wins"

    def test_union_collision_policy_last_wins_explicit(self):
        """Explicit last_wins matches default behavior."""
        executor, _, _, tm, _ = _make_executor()
        s = _settings(
            branches=["a", "b"],
            merge="union",
            union_collision_policy="last_wins",
        )
        executor.register_coalesce(s, "node_1")
        t1 = _make_token(branch_name="a", token_id="t1", data={"shared": "from_a"})
        t2 = _make_token(branch_name="b", token_id="t2", data={"shared": "from_b"})
        executor.accept(t1, "merge")
        o = executor.accept(t2, "merge")
        merged = tm.coalesce_tokens.call_args.kwargs["merged_data"].to_dict()
        assert merged["shared"] == "from_b"
        assert o.coalesce_metadata.union_field_origins["shared"] == "b"

    # ------------------------------------------------------------------
    # union_collision_policy=first_wins
    # ------------------------------------------------------------------

    def test_union_collision_policy_first_wins_two_way(self):
        """first_wins: the merged row takes the first branch's value."""
        executor, _, _, tm, _ = _make_executor()
        s = _settings(
            branches=["a", "b"],
            merge="union",
            union_collision_policy="first_wins",
        )
        executor.register_coalesce(s, "node_1")
        t1 = _make_token(branch_name="a", token_id="t1", data={"shared": "from_a"})
        t2 = _make_token(branch_name="b", token_id="t2", data={"shared": "from_b"})
        executor.accept(t1, "merge")
        o = executor.accept(t2, "merge")
        merged = tm.coalesce_tokens.call_args.kwargs["merged_data"].to_dict()
        assert merged["shared"] == "from_a"
        # Origins reflect the winner.
        assert o.coalesce_metadata.union_field_origins["shared"] == "a"
        # Collision values still record every contributing branch in order.
        entries = list(o.coalesce_metadata.union_field_collision_values["shared"])
        assert entries == [("a", "from_a"), ("b", "from_b")]

    def test_union_collision_policy_first_wins_three_way(self):
        """first_wins: first branch in settings.branches order wins for 3-way collisions."""
        executor, _, _, tm, _ = _make_executor()
        s = _settings(
            branches=["a", "b", "c"],
            merge="union",
            policy="require_all",
            union_collision_policy="first_wins",
        )
        executor.register_coalesce(s, "node_1")
        t1 = _make_token(branch_name="a", token_id="t1", data={"f": "va"})
        t2 = _make_token(branch_name="b", token_id="t2", data={"f": "vb"})
        t3 = _make_token(branch_name="c", token_id="t3", data={"f": "vc"})
        executor.accept(t1, "merge")
        executor.accept(t2, "merge")
        o = executor.accept(t3, "merge")
        merged = tm.coalesce_tokens.call_args.kwargs["merged_data"].to_dict()
        assert merged["f"] == "va"
        assert o.coalesce_metadata.union_field_origins["f"] == "a"

    # ------------------------------------------------------------------
    # union_collision_policy=fail
    # ------------------------------------------------------------------

    def test_union_collision_policy_fail_raises_on_collision(self):
        """fail: CoalesceCollisionError raised with full metadata attached."""
        executor, _, _, _, _ = _make_executor()
        s = _settings(
            branches=["a", "b"],
            merge="union",
            union_collision_policy="fail",
        )
        executor.register_coalesce(s, "node_1")
        t1 = _make_token(branch_name="a", token_id="t1", data={"shared": "from_a"})
        t2 = _make_token(branch_name="b", token_id="t2", data={"shared": "from_b"})
        executor.accept(t1, "merge")
        with pytest.raises(CoalesceCollisionError) as exc_info:
            executor.accept(t2, "merge")
        # Metadata must be attached so the orchestrator's failure path
        # can persist the full collision record to the audit trail.
        md = exc_info.value.metadata
        assert md.union_field_origins is not None
        assert md.union_field_collision_values is not None
        entries = list(md.union_field_collision_values["shared"])
        assert entries == [("a", "from_a"), ("b", "from_b")]

    def test_union_collision_policy_fail_no_collisions_is_noop(self):
        """fail: non-overlapping branches merge successfully without raising."""
        executor, _, _, tm, _ = _make_executor()
        s = _settings(
            branches=["a", "b"],
            merge="union",
            union_collision_policy="fail",
        )
        executor.register_coalesce(s, "node_1")
        t1 = _make_token(branch_name="a", token_id="t1", data={"x": 1})
        t2 = _make_token(branch_name="b", token_id="t2", data={"y": 2})
        executor.accept(t1, "merge")
        o = executor.accept(t2, "merge")
        merged = tm.coalesce_tokens.call_args.kwargs["merged_data"].to_dict()
        assert merged == {"x": 1, "y": 2}
        # No collisions: collision_values absent.
        assert o.coalesce_metadata.union_field_collision_values is None
        # field_origins always populated.
        assert o.coalesce_metadata.union_field_origins == {"x": "a", "y": "b"}

    def test_union_collision_policy_fail_records_metadata_to_audit(self):
        """fail policy must propagate collision metadata to complete_node_state(context_after=...).

        Without this propagation, the audit trail loses the field-level provenance
        that union_collision_policy=fail exists to capture. The stringified exception
        message preserves only the field name — every (branch, value) pair would be
        lost, defeating the whole point of opting into hard-fail enforcement.
        """
        executor, execution, _, _, _ = _make_executor()
        s = _settings(
            branches=["a", "b"],
            merge="union",
            union_collision_policy="fail",
        )
        executor.register_coalesce(s, "node_1")
        t1 = _make_token(branch_name="a", token_id="t1", data={"shared": "from_a"})
        t2 = _make_token(branch_name="b", token_id="t2", data={"shared": "from_b"})
        executor.accept(t1, "merge")
        with pytest.raises(CoalesceCollisionError):
            executor.accept(t2, "merge")

        # Inspect complete_node_state calls: the failure cleanup handler must have
        # recorded at least one FAILED node state carrying the collision metadata.
        fail_calls = [call for call in execution.complete_node_state.call_args_list if call.kwargs.get("status") == NodeStateStatus.FAILED]
        assert fail_calls, "expected at least one FAILED node state on union_collision_policy=fail"

        metadata_calls = [
            call
            for call in fail_calls
            if call.kwargs.get("context_after") is not None and call.kwargs["context_after"].union_field_collision_values is not None
        ]
        assert metadata_calls, (
            "expected fail-path audit record to carry union_field_collision_values; "
            "without this, the Landscape audit trail loses the (branch, value) pairs "
            "that union_collision_policy=fail is specifically designed to preserve"
        )

        md = metadata_calls[0].kwargs["context_after"]
        # Every branch's contributing value must survive into the audit record.
        entries = list(md.union_field_collision_values["shared"])
        assert entries == [("a", "from_a"), ("b", "from_b")]
        # field_origins must also be present (last_wins default before the raise).
        assert md.union_field_origins is not None
        assert md.union_field_origins["shared"] == "b"

    def test_union_collision_policy_fail_records_terminal_failed_outcomes(self):
        """union_collision_policy=fail must record FAILURE/UNROUTED for consumed tokens.

        Bug: The exception handler in _execute_merge only calls complete_node_state(FAILED)
        but never calls record_token_outcome(FAILED). Without terminal outcomes:
        - Recovery treats the row as incomplete (key remains in _pending)
        - Lineage resolution can't find a terminal token
        """
        executor, _, data_flow, _, _ = _make_executor()
        s = _settings(
            branches=["a", "b"],
            merge="union",
            union_collision_policy="fail",
        )
        executor.register_coalesce(s, "node_1")
        t1 = _make_token(branch_name="a", token_id="t1", data={"shared": "from_a"})
        t2 = _make_token(branch_name="b", token_id="t2", data={"shared": "from_b"})
        executor.accept(t1, "merge")
        with pytest.raises(CoalesceCollisionError):
            executor.accept(t2, "merge")

        # All consumed tokens must have terminal FAILED outcomes recorded
        outcome_calls = data_flow.record_token_outcome.call_args_list
        assert len(outcome_calls) == 2, f"expected record_token_outcome(FAILED) for both consumed tokens; got {len(outcome_calls)} calls"
        for c in outcome_calls:
            assert c.kwargs["outcome"] == TerminalOutcome.FAILURE
            assert c.kwargs["path"] == TerminalPath.UNROUTED
            assert "error_hash" in c.kwargs

        token_ids = {c.kwargs["ref"].token_id for c in outcome_calls}
        assert token_ids == {"t1", "t2"}

    def test_union_collision_policy_fail_cleans_up_pending(self):
        """union_collision_policy=fail must remove key from _pending after failure.

        Bug: The exception handler doesn't call del self._pending[key] or
        _mark_completed(key). Without cleanup:
        - Recovery treats the row as incomplete
        - Late arrivals aren't rejected
        """
        executor, _, _, _, _ = _make_executor()
        s = _settings(
            branches=["a", "b"],
            merge="union",
            union_collision_policy="fail",
        )
        executor.register_coalesce(s, "node_1")
        t1 = _make_token(branch_name="a", token_id="t1", data={"shared": "from_a"})
        t2 = _make_token(branch_name="b", token_id="t2", data={"shared": "from_b"})
        executor.accept(t1, "merge")

        # Capture the key before the failure
        assert len(executor._pending) == 1
        key = next(iter(executor._pending.keys()))

        with pytest.raises(CoalesceCollisionError):
            executor.accept(t2, "merge")

        # After failure, _pending should be empty
        assert key not in executor._pending, (
            "_pending should be cleaned up after union_collision_policy=fail; leaving the key breaks recovery (treats row as incomplete)"
        )

        # Key should be in completed set (rejects late arrivals)
        assert key in executor._completed_keys, "key should be marked completed to reject late arrivals"

    # ------------------------------------------------------------------
    # Orthogonality: union_collision_policy vs arrival policy
    # ------------------------------------------------------------------

    def test_union_collision_policy_independent_of_require_all(self):
        """union_collision_policy=fail with require_all still raises on collisions.

        The two policy axes (arrival policy and collision policy) are independent:
        require_all governs branch arrival; union_collision_policy governs
        field-level conflict resolution within the merged row.
        """
        executor, _, _, _, _ = _make_executor()
        s = _settings(
            branches=["a", "b"],
            merge="union",
            policy="require_all",
            union_collision_policy="fail",
        )
        executor.register_coalesce(s, "node_1")
        t1 = _make_token(branch_name="a", token_id="t1", data={"shared": 1})
        t2 = _make_token(branch_name="b", token_id="t2", data={"shared": 2})
        executor.accept(t1, "merge")
        with pytest.raises(CoalesceCollisionError):
            executor.accept(t2, "merge")

    def test_union_collision_policy_with_best_effort_records_arrived_only(self):
        """best_effort with one lost branch: field_origins reflects arrived branches only.

        Lost-branch handling under best_effort merges whatever arrived by the
        deadline. field_origins and collision_values are still populated —
        just scoped to the arrived subset. (Lost-branch provenance handling
        is a separate issue out of scope here.)
        """
        clock = MockClock(start=100.0)
        executor, _, _, _, _ = _make_executor(clock=clock)
        s = _settings(
            branches=["a", "b"],
            merge="union",
            policy="best_effort",
            timeout_seconds=5.0,
        )
        executor.register_coalesce(s, "node_1")
        t1 = _make_token(branch_name="a", token_id="t1", data={"x": 1})
        executor.accept(t1, "merge")
        # Advance clock past timeout so best_effort flushes with only `a` arrived.
        clock.advance(10.0)
        outcomes = executor.check_timeouts("merge")
        assert len(outcomes) == 1
        outcome = outcomes[0]
        assert outcome.merged_token is not None
        origins = outcome.coalesce_metadata.union_field_origins
        assert origins == {"x": "a"}
        # No collisions (only one branch contributed).
        assert outcome.coalesce_metadata.union_field_collision_values is None

    def test_union_collision_policy_first_wins_with_timeout(self):
        """first_wins with timeout: collision resolution uses settings.branches order, not arrival.

        When best_effort triggers via timeout and multiple branches have arrived
        with colliding fields, first_wins must resolve to the branch that appears
        first in the configured `settings.branches` list — regardless of which
        branch's token arrived first. This test verifies that branch-order, not
        arrival-order, determines the winner.
        """
        clock = MockClock(start=100.0)
        executor, _, _, tm, _ = _make_executor(clock=clock)
        # Branches ordered ["a", "b", "c"] — under first_wins, "a" wins collisions.
        s = _settings(
            branches=["a", "b", "c"],
            merge="union",
            policy="best_effort",
            timeout_seconds=5.0,
            union_collision_policy="first_wins",
        )
        executor.register_coalesce(s, "node_1")

        # Submit b first, then a — intentionally reversed from branch order.
        # Branch c never arrives (simulates lost/slow branch).
        t_b = _make_token(branch_name="b", token_id="t_b", data={"shared": "from_b", "only_b": 10})
        t_a = _make_token(branch_name="a", token_id="t_a", data={"shared": "from_a", "only_a": 20})
        executor.accept(t_b, "merge")
        executor.accept(t_a, "merge")

        # Advance past timeout — triggers best_effort flush with a and b arrived.
        clock.advance(10.0)
        outcomes = executor.check_timeouts("merge")
        assert len(outcomes) == 1
        outcome = outcomes[0]
        assert outcome.merged_token is not None

        # Merged data: first_wins means "a" wins the collision on "shared".
        merged = tm.coalesce_tokens.call_args.kwargs["merged_data"].to_dict()
        assert merged["shared"] == "from_a"  # first_wins: "a" wins over "b"
        assert merged["only_a"] == 20
        assert merged["only_b"] == 10

        # field_origins: "shared" attributed to "a" (first in branches order).
        origins = outcome.coalesce_metadata.union_field_origins
        assert origins["shared"] == "a"
        assert origins["only_a"] == "a"
        assert origins["only_b"] == "b"

        # collision_values: records both contributing branches for "shared".
        collision_values = outcome.coalesce_metadata.union_field_collision_values
        assert collision_values is not None
        entries = list(collision_values["shared"])
        # Order in collision_values is branch order, not arrival order.
        assert entries == [("a", "from_a"), ("b", "from_b")]


# ===========================================================================
# nested merge
# ===========================================================================


class TestNestedMerge:
    def test_each_branch_nested(self):
        executor, _, _, tm, _ = _make_executor()
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
        executor, _, _, tm, _ = _make_executor()
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
        executor, _, _, tm, _ = _make_executor()
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
        executor, _, _, tm, _ = _make_executor()
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
        executor, _, _, _, _ = _make_executor()
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
        executor, _, _, tm, _ = _make_executor()
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
        executor, _, _, _, _ = _make_executor()
        executor.register_coalesce(_settings(policy="require_all"), "node_1")
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        results = executor.check_timeouts("merge")
        assert results == []

    def test_not_expired_returns_empty(self):
        executor, _, _, _, clock = _make_executor()
        s = _settings(policy="best_effort", timeout_seconds=10.0)
        executor.register_coalesce(s, "node_1")
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        clock.advance(5.0)  # Only 5s of 10s timeout
        results = executor.check_timeouts("merge")
        assert results == []

    def test_best_effort_expired_merges(self):
        executor, _, _, _, clock = _make_executor()
        s = _settings(policy="best_effort", timeout_seconds=10.0)
        executor.register_coalesce(s, "node_1")
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        clock.advance(11.0)
        results = executor.check_timeouts("merge")
        assert len(results) == 1
        assert results[0].merged_token is not None

    def test_best_effort_expired_cleans_pending(self):
        """Timeout-triggered merge removes the pending entry."""
        executor, _, _, _, clock = _make_executor()
        s = _settings(policy="best_effort", timeout_seconds=10.0)
        executor.register_coalesce(s, "node_1")
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        assert ("merge", "row_1") in executor._pending
        clock.advance(11.0)
        executor.check_timeouts("merge")
        assert ("merge", "row_1") not in executor._pending

    def test_quorum_expired_quorum_not_met_fails(self):
        executor, _, _, _, clock = _make_executor()
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
        executor, _, _, _, clock = _make_executor()
        s = _settings(policy="require_all", timeout_seconds=5.0)
        executor.register_coalesce(s, "node_1")
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        clock.advance(6.0)
        results = executor.check_timeouts("merge")
        assert len(results) == 1
        assert results[0].failure_reason == "incomplete_branches"
        assert results[0].outcomes_recorded is True

    def test_multiple_pending_some_expired(self):
        executor, _, _, _, clock = _make_executor()
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
        with pytest.raises(OrchestrationInvariantError, match="not registered"):
            executor.check_timeouts("ghost")

    def test_exact_timeout_boundary_triggers(self):
        """Timeout check fires when elapsed == timeout_seconds."""
        executor, _, _, _, clock = _make_executor()
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
        executor, _, _, _, _ = _make_executor()
        s = _settings(policy="best_effort", timeout_seconds=60.0)
        executor.register_coalesce(s, "node_1")
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        results = executor.flush_pending()
        assert len(results) == 1
        assert results[0].merged_token is not None

    def test_best_effort_one_lost_one_arrived_flush(self):
        """Flush merges arrived tokens even when some branches are lost."""
        executor, _, _, _, _ = _make_executor()
        s = _settings(branches=["a", "b", "c"], policy="best_effort", timeout_seconds=60.0)
        executor.register_coalesce(s, "node_1")
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        # Don't report any losses; flush should merge what's there
        results = executor.flush_pending()
        assert len(results) == 1
        assert results[0].merged_token is not None

    def test_quorum_not_met_at_flush_fails(self):
        executor, _, _, _, _ = _make_executor()
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
        executor, _, _, _, _ = _make_executor()
        s = _settings(policy="require_all", timeout_seconds=5.0)
        executor.register_coalesce(s, "node_1")
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        results = executor.flush_pending()
        assert len(results) == 1
        assert results[0].failure_reason == "incomplete_branches"

    def test_first_policy_with_pending_raises(self):
        executor, _, _, _, _ = _make_executor()
        s = _settings(policy="first")
        executor.register_coalesce(s, "node_1")
        # Normally impossible since first merges immediately.
        # Force a pending entry for the test.
        key = ("merge", "row_1")
        token = _make_token(branch_name="a")
        executor._pending[key] = _PendingCoalesce(
            branches={"a": _BranchEntry(token=token, arrival_time=100.0, state_id="state_fake")},
            first_arrival=100.0,
        )
        with pytest.raises(RuntimeError, match="Invariant violation"):
            executor.flush_pending()

    def test_flush_clears_completed_keys(self):
        executor, _, _, _, _ = _make_executor()
        s = _settings(policy="require_all", timeout_seconds=5.0)
        executor.register_coalesce(s, "node_1")
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        executor.accept(_make_token(branch_name="b", token_id="t2"), "merge")
        assert len(executor._completed_keys) == 1
        executor.flush_pending()
        assert len(executor._completed_keys) == 0

    def test_flush_no_pending_returns_empty(self):
        """Flush with no pending entries returns an empty list."""
        executor, _, _, _, _ = _make_executor()
        s = _settings(policy="require_all")
        executor.register_coalesce(s, "node_1")
        results = executor.flush_pending()
        assert results == []

    def test_flush_multiple_pending_rows(self):
        """Flush processes all pending entries across different rows."""
        executor, _, _, _, _ = _make_executor()
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
        with pytest.raises(OrchestrationInvariantError, match="not registered"):
            executor.notify_branch_lost("ghost", "row_1", "a", "reason")

    def test_unknown_branch_raises(self):
        executor, *_ = _make_executor()
        executor.register_coalesce(_settings(branches=["a", "b"]), "node_1")
        with pytest.raises(OrchestrationInvariantError, match="not in expected branches"):
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

    @pytest.mark.filterwarnings("ignore:Coalesce.*quorum_count.*equals branch count:UserWarning")
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
        with pytest.raises(OrchestrationInvariantError, match="already arrived"):
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
        executor, _, _, _, _ = _make_executor()
        executor.register_coalesce(_settings(), "node_1")
        contract = _make_contract(mode="FLEXIBLE")
        t1 = _make_token(branch_name="a", token_id="t1", data={"amount": 1}, contract=contract)
        # Simulate a bug: token with None contract
        bad_row = MagicMock(spec=PipelineRow)
        bad_row.contract = None
        bad_row.to_dict.return_value = {"amount": 2}
        t2 = TokenInfo(row_id="row_1", token_id="t2", row_data=bad_row, branch_name="b")
        executor.accept(t1, "merge")
        with pytest.raises(OrchestrationInvariantError, match="has no contract"):
            executor.accept(t2, "merge")

    def test_union_contracts_merged(self):
        """Union merge should merge contracts from all branches."""
        executor, _, _, tm, _ = _make_executor()
        executor.register_coalesce(_settings(merge="union"), "node_1")
        c_a = _make_contract(
            fields=[
                make_field("x", python_type=int, required=True, source="declared"),
            ],
            mode="FLEXIBLE",
        )
        c_b = _make_contract(
            fields=[
                make_field("y", python_type=str, required=True, source="declared"),
            ],
            mode="FLEXIBLE",
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
        executor, _, _, tm, _ = _make_executor()
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
        executor, _, _, tm, _ = _make_executor()
        s = _settings(merge="select", select_branch="a")
        executor.register_coalesce(s, "node_1")
        c_a = _make_contract(
            fields=[
                make_field("chosen", python_type=str, required=True, source="declared"),
            ],
            mode="FLEXIBLE",
        )
        c_b = _make_contract(
            fields=[
                make_field("ignored", python_type=int, required=True, source="declared"),
            ],
            mode="FLEXIBLE",
        )
        t1 = _make_token(branch_name="a", token_id="t1", data={"chosen": "yes"}, contract=c_a)
        t2 = _make_token(branch_name="b", token_id="t2", data={"ignored": 0}, contract=c_b)
        executor.accept(t1, "merge")
        executor.accept(t2, "merge")
        merged_data = tm.coalesce_tokens.call_args.kwargs["merged_data"]
        assert merged_data.contract is c_a

    def test_conflicting_contracts_fail_gracefully(self):
        """Union merge with conflicting field types fails row gracefully.

        Type conflicts are detected at merge time when contracts from different
        branches have incompatible types for the same field. For observed schemas,
        this can't be caught at build time, so we fail gracefully instead of
        crashing — the audit trail must remain complete.
        (See: elspeth-c75ac86e35)
        """
        executor, _, _, _, _ = _make_executor()
        executor.register_coalesce(_settings(merge="union"), "node_1")
        c_a = _make_contract(
            fields=[
                make_field("value", python_type=int, required=True, source="declared"),
            ],
            mode="FLEXIBLE",
        )
        c_b = _make_contract(
            fields=[
                make_field("value", python_type=str, required=True, source="declared"),
            ],
            mode="FLEXIBLE",
        )
        t1 = _make_token(branch_name="a", token_id="t1", data={"value": 1}, contract=c_a)
        t2 = _make_token(branch_name="b", token_id="t2", data={"value": "x"}, contract=c_b)
        executor.accept(t1, "merge")
        # Second accept triggers merge, which fails due to type conflict
        outcome = executor.accept(t2, "merge")

        # Outcome indicates failure, not held or merged
        assert outcome.failure_reason is not None
        assert "contract_type_conflict" in outcome.failure_reason
        assert outcome.held is False
        assert outcome.merged_token is None
        assert outcome.outcomes_recorded is True  # Tokens properly terminated

    def test_observed_schema_type_conflict_fails_gracefully(self):
        """Observed schemas with runtime type conflicts fail gracefully.

        This is the primary scenario for elspeth-c75ac86e35: branches with
        OBSERVED schemas (no declared fields) can produce incompatible types
        at runtime. Build-time validation can't catch this because observed
        schemas have no fields until data flows. The fix ensures graceful
        failure at merge time instead of crashing.
        """
        executor, _, _, _, _ = _make_executor()
        executor.register_coalesce(_settings(merge="union"), "node_1")

        # Both contracts are OBSERVED (runtime-inferred via _make_contract),
        # simulating the bug scenario where build-time validation passes but
        # runtime types conflict
        c_a = _make_contract(
            fields=[
                make_field("count", python_type=int, required=True, source="inferred"),
            ],
            mode="OBSERVED",
        )
        c_b = _make_contract(
            fields=[
                make_field("count", python_type=str, required=True, source="inferred"),
            ],
            mode="OBSERVED",
        )

        t1 = _make_token(branch_name="a", token_id="t1", data={"count": 42}, contract=c_a)
        t2 = _make_token(branch_name="b", token_id="t2", data={"count": "forty-two"}, contract=c_b)

        executor.accept(t1, "merge")
        outcome = executor.accept(t2, "merge")

        # Graceful failure, not a crash
        assert outcome.failure_reason is not None
        assert "contract_type_conflict" in outcome.failure_reason
        # Error message should include helpful details about the conflict
        assert "count" in outcome.failure_reason  # Field name
        assert "int" in outcome.failure_reason  # Type info
        assert "str" in outcome.failure_reason  # Type info
        assert outcome.outcomes_recorded is True


# ===========================================================================
# Audit trail details
# ===========================================================================


class TestAuditTrailDetails:
    def test_begin_node_state_captures_input_data(self):
        """begin_node_state should pass the token's row data as input_data."""
        executor, execution, _, _, _ = _make_executor()
        executor.register_coalesce(_settings(), "node_1")
        t = _make_token(branch_name="a", token_id="t1", data={"amount": 42})
        executor.accept(t, "merge")
        kw = execution.begin_node_state.call_args.kwargs
        assert kw["token_id"] == "t1"
        assert kw["run_id"] == "run_1"
        assert kw["step_index"] == 5
        assert kw["input_data"]["amount"] == 42

    def test_begin_node_state_uses_correct_node_id(self):
        """begin_node_state should use the node_id from register_coalesce."""
        executor, execution, _, _, _ = _make_executor()
        executor.register_coalesce(_settings(), "coalesce_node_42")
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        kw = execution.begin_node_state.call_args.kwargs
        assert kw["node_id"] == "coalesce_node_42"

    def test_complete_node_state_duration_ms(self):
        """Completed node states should have a non-negative duration_ms."""
        executor, execution, _, _, clock = _make_executor()
        executor.register_coalesce(_settings(), "node_1")
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        clock.advance(0.5)
        executor.accept(_make_token(branch_name="b", token_id="t2"), "merge")
        calls = execution.complete_node_state.call_args_list
        durations = [c.kwargs["duration_ms"] for c in calls if c.kwargs.get("status") == NodeStateStatus.COMPLETED]
        assert len(durations) == 2
        assert all(d >= 0 for d in durations)
        # At least one should have waited ~500ms
        assert any(d >= 400 for d in durations)

    def test_complete_node_state_includes_coalesce_metadata(self):
        """Completed node states should include CoalesceMetadata in context_after."""
        from elspeth.contracts.coalesce_metadata import CoalesceMetadata

        executor, execution, _, _, _ = _make_executor()
        executor.register_coalesce(_settings(), "node_1")
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        executor.accept(_make_token(branch_name="b", token_id="t2"), "merge")
        for c in execution.complete_node_state.call_args_list:
            if c.kwargs.get("status") == NodeStateStatus.COMPLETED:
                ctx = c.kwargs.get("context_after")
                assert isinstance(ctx, CoalesceMetadata)
                assert "policy" in ctx.to_dict()

    def test_complete_node_state_output_data_merged_into(self):
        """Completed node states have output_data with merged_into token ID."""
        executor, execution, _, _, _ = _make_executor()
        executor.register_coalesce(_settings(), "node_1")
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        executor.accept(_make_token(branch_name="b", token_id="t2"), "merge")
        for c in execution.complete_node_state.call_args_list:
            if c.kwargs.get("status") == NodeStateStatus.COMPLETED:
                output = c.kwargs.get("output_data", {})
                assert "merged_into" in output
                assert output["merged_into"].startswith("merged_")

    def test_record_token_outcome_has_join_group_id(self):
        """Token outcomes should include join_group_id from merged token."""
        executor, _, data_flow, _, _ = _make_executor()
        executor.register_coalesce(_settings(), "node_1")
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        executor.accept(_make_token(branch_name="b", token_id="t2"), "merge")
        for c in data_flow.record_token_outcome.call_args_list:
            assert c.kwargs["join_group_id"] is not None

    def test_record_token_outcome_has_correct_token_ids(self):
        """Token outcomes should reference the original consumed token IDs."""
        executor, _, data_flow, _, _ = _make_executor()
        executor.register_coalesce(_settings(), "node_1")
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        executor.accept(_make_token(branch_name="b", token_id="t2"), "merge")
        token_ids = {c.kwargs["ref"].token_id for c in data_flow.record_token_outcome.call_args_list}
        assert token_ids == {"t1", "t2"}

    def test_merge_metadata_arrival_order(self):
        """Coalesce metadata should include arrival_order with offset_ms."""
        executor, _, _, _, clock = _make_executor()
        executor.register_coalesce(_settings(), "node_1")
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        clock.advance(0.2)
        o = executor.accept(_make_token(branch_name="b", token_id="t2"), "merge")
        arrival_order = o.coalesce_metadata.arrival_order
        assert len(arrival_order) == 2
        assert arrival_order[0].branch == "a"
        assert arrival_order[0].arrival_offset_ms == pytest.approx(0.0)
        assert arrival_order[1].branch == "b"
        assert arrival_order[1].arrival_offset_ms == pytest.approx(200.0)

    def test_merge_metadata_wait_duration(self):
        """Coalesce metadata should include total wait_duration_ms."""
        executor, _, _, _, clock = _make_executor()
        executor.register_coalesce(_settings(), "node_1")
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        clock.advance(1.5)
        o = executor.accept(_make_token(branch_name="b", token_id="t2"), "merge")
        assert o.coalesce_metadata.wait_duration_ms == pytest.approx(1500.0)

    def test_merge_metadata_branches_lost_empty_when_none_lost(self):
        """Branches_lost in metadata should be empty MappingProxy when all arrived."""
        executor, _, _, _, _ = _make_executor()
        executor.register_coalesce(_settings(), "node_1")
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        o = executor.accept(_make_token(branch_name="b", token_id="t2"), "merge")
        assert dict(o.coalesce_metadata.branches_lost) == {}


# ===========================================================================
# Default clock usage
# ===========================================================================


class TestDefaultClock:
    def test_uses_default_clock_when_none(self):
        """Constructor should use DEFAULT_CLOCK when clock=None."""
        from elspeth.engine.clock import DEFAULT_CLOCK

        execution = MagicMock(spec=ExecutionRepository)
        execution.begin_node_state.side_effect = lambda **kw: Mock(state_id="s1")
        executor = CoalesceExecutor(
            execution,
            MagicMock(),
            MagicMock(),
            "run_1",
            step_resolver=lambda n: 0,
            clock=None,
            data_flow=MagicMock(spec=DataFlowRepository),
        )
        assert executor._clock is DEFAULT_CLOCK

    def test_uses_injected_clock(self):
        clock = MockClock(start=42.0)
        execution = MagicMock(spec=ExecutionRepository)
        execution.begin_node_state.side_effect = lambda **kw: Mock(state_id="s1")
        executor = CoalesceExecutor(
            execution,
            MagicMock(),
            MagicMock(),
            "run_1",
            step_resolver=lambda n: 0,
            clock=clock,
            data_flow=MagicMock(spec=DataFlowRepository),
        )
        assert executor._clock is clock


# ===========================================================================
# Multi-row isolation
# ===========================================================================


class TestMultiRowIsolation:
    def test_different_rows_independent(self):
        """Tokens for different row_ids are tracked independently."""
        executor, _, _, _, _ = _make_executor()
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
        executor, _, _, _, _ = _make_executor()
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
        executor, execution, _, _, clock = _make_executor()
        s = _settings(policy="require_all", timeout_seconds=5.0)
        executor.register_coalesce(s, "node_1")
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        clock.advance(6.0)
        executor.check_timeouts("merge")
        # Check that complete_node_state was called with FAILED
        fail_calls = [c for c in execution.complete_node_state.call_args_list if c.kwargs.get("status") == NodeStateStatus.FAILED]
        assert len(fail_calls) == 1

    def test_failure_records_token_outcomes_failed(self):
        executor, _, data_flow, _, clock = _make_executor()
        s = _settings(policy="require_all", timeout_seconds=5.0)
        executor.register_coalesce(s, "node_1")
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        clock.advance(6.0)
        executor.check_timeouts("merge")
        outcome_calls = data_flow.record_token_outcome.call_args_list
        assert len(outcome_calls) == 1
        assert outcome_calls[0].kwargs["outcome"] == TerminalOutcome.FAILURE
        assert outcome_calls[0].kwargs["path"] == TerminalPath.UNROUTED

    def test_failure_metadata_includes_policy(self):
        executor, _, _, _, clock = _make_executor()
        s = _settings(policy="require_all", timeout_seconds=5.0)
        executor.register_coalesce(s, "node_1")
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        clock.advance(6.0)
        results = executor.check_timeouts("merge")
        md = results[0].coalesce_metadata
        assert md.policy == CoalescePolicy.REQUIRE_ALL
        assert set(md.expected_branches) == {"a", "b"}

    def test_failure_removes_pending_entry(self):
        executor, _, _, _, clock = _make_executor()
        s = _settings(policy="require_all", timeout_seconds=5.0)
        executor.register_coalesce(s, "node_1")
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        assert ("merge", "row_1") in executor._pending
        clock.advance(6.0)
        executor.check_timeouts("merge")
        assert ("merge", "row_1") not in executor._pending

    def test_failure_marks_key_completed(self):
        executor, _, _, _, clock = _make_executor()
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
        assert result.coalesce_metadata.branches_lost is not None
        assert "b" in result.coalesce_metadata.branches_lost

    @pytest.mark.filterwarnings("ignore:Coalesce.*quorum_count.*equals branch count:UserWarning")
    def test_failure_metadata_includes_quorum_required(self):
        executor, *_ = _make_executor()
        s = _settings(branches=["a", "b", "c"], policy="quorum", quorum_count=3)
        executor.register_coalesce(s, "node_1")
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        # Loss of b -> max_possible=2 < quorum=3 -> fail
        result = executor.notify_branch_lost("merge", "row_1", "b", "error")
        assert result.coalesce_metadata.quorum_required == 3

    def test_require_all_timeout_metadata_has_timeout_seconds(self):
        executor, _, _, _, clock = _make_executor()
        s = _settings(policy="require_all", timeout_seconds=8.0)
        executor.register_coalesce(s, "node_1")
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        clock.advance(9.0)
        results = executor.check_timeouts("merge")
        assert results[0].coalesce_metadata.timeout_seconds == 8.0

    def test_failure_error_hash_is_deterministic(self):
        """The error_hash recorded for failed tokens should be consistent."""
        executor, _, data_flow, _, clock = _make_executor()
        s = _settings(policy="require_all", timeout_seconds=5.0)
        executor.register_coalesce(s, "node_1")
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        clock.advance(6.0)
        executor.check_timeouts("merge")
        # record_token_outcome should have been called with an error_hash
        kw = data_flow.record_token_outcome.call_args.kwargs
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
        assert set(result.coalesce_metadata.branches_arrived) == {"a", "b"}

    def test_require_all_timeout_error_includes_timeout_ms(self):
        """Bug de4781: require_all timeout path must include timeout_ms in error payload.

        Previously, timeout_ms was only set when 'timeout' appeared in the
        failure_reason string. The require_all timeout path uses
        failure_reason='incomplete_branches', so timeout_ms was omitted.
        """
        executor, execution, _, _, clock = _make_executor()
        s = _settings(policy="require_all", timeout_seconds=5.0)
        executor.register_coalesce(s, "node_1")
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        clock.advance(6.0)
        executor.check_timeouts("merge")
        # The error payload recorded via complete_node_state must have timeout_ms
        fail_call = next(c for c in execution.complete_node_state.call_args_list if c.kwargs.get("status") == NodeStateStatus.FAILED)
        error = fail_call.kwargs["error"]
        assert error.timeout_ms == 5000
        assert error.failure_reason == "incomplete_branches"

    def test_flush_pending_require_all_does_not_set_timeout_ms(self):
        """flush_pending is NOT a timeout — timeout_ms should be None."""
        executor, execution, *_ = _make_executor()
        s = _settings(policy="require_all", timeout_seconds=5.0)
        executor.register_coalesce(s, "node_1")
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        executor.flush_pending()
        fail_call = next(c for c in execution.complete_node_state.call_args_list if c.kwargs.get("status") == NodeStateStatus.FAILED)
        error = fail_call.kwargs["error"]
        assert error.timeout_ms is None
        assert error.failure_reason == "incomplete_branches"


# ===========================================================================
# Lost branch expected fields — audit trail for diverted branch field impact
# ===========================================================================


class TestLostBranchExpectedFields:
    """Tests for lost_branch_expected_fields in CoalesceMetadata.

    When a branch is diverted to an error sink (lost), the coalesce metadata
    should record which fields that branch would have contributed. This enables
    audit queries like "what fields were expected from lost branch X?" without
    requiring DAG traversal at query time.
    """

    def test_failure_metadata_includes_lost_branch_expected_fields_when_registered(self):
        """When branch schemas are registered, lost_branch_expected_fields is populated."""
        executor, *_ = _make_executor()
        s = _settings(branches=["a", "b"], policy="require_all")
        branch_schemas = {
            "a": ("field_x", "field_y"),
            "b": ("field_z",),
        }
        executor.register_coalesce(s, "node_1", branch_schemas)

        # Loss of b triggers require_all failure
        result = executor.notify_branch_lost("merge", "row_1", "b", "upstream_fail")

        assert result.coalesce_metadata.lost_branch_expected_fields is not None
        assert result.coalesce_metadata.lost_branch_expected_fields == {"b": ("field_z",)}

    def test_failure_metadata_lost_branch_expected_fields_none_when_not_registered(self):
        """When no branch schemas are registered, lost_branch_expected_fields is None."""
        executor, *_ = _make_executor()
        s = _settings(branches=["a", "b"], policy="require_all")
        # No branch schemas passed to register_coalesce
        executor.register_coalesce(s, "node_1")

        # Loss of b triggers require_all failure
        result = executor.notify_branch_lost("merge", "row_1", "b", "upstream_fail")

        assert result.coalesce_metadata.lost_branch_expected_fields is None

    def test_failure_metadata_multiple_lost_branches_expected_fields(self):
        """When multiple branches are lost, all their expected fields are recorded."""
        executor, *_ = _make_executor()
        # Use best_effort with 3 branches so we can lose 2 and still merge
        s = _settings(branches=["a", "b", "c"], policy="best_effort", timeout_seconds=1.0)
        branch_schemas = {
            "a": ("field_1",),
            "b": ("field_2", "field_3"),
            "c": ("field_4",),
        }
        executor.register_coalesce(s, "node_1", branch_schemas)

        # Let c arrive first
        executor.accept(_make_token(branch_name="c", token_id="t1"), "merge")

        # Lose branch a (no merge yet — still waiting)
        result = executor.notify_branch_lost("merge", "row_1", "a", "error_a")
        assert result is None  # best_effort waits for all branches to be accounted for

        # Lose branch b — now all branches accounted for (1 arrived, 2 lost) -> merge
        result = executor.notify_branch_lost("merge", "row_1", "b", "error_b")
        assert result is not None
        assert result.merged_token is not None

        # Both lost branches should appear in lost_branch_expected_fields
        expected_fields = result.coalesce_metadata.lost_branch_expected_fields
        assert expected_fields is not None
        assert set(expected_fields.keys()) == {"a", "b"}
        assert expected_fields["a"] == ("field_1",)
        assert expected_fields["b"] == ("field_2", "field_3")

    def test_merge_metadata_includes_lost_branch_expected_fields_for_best_effort(self):
        """When best_effort merge completes with lost branches, expected fields are recorded."""
        executor, *_ = _make_executor()
        s = _settings(branches=["a", "b"], policy="best_effort", timeout_seconds=1.0)
        branch_schemas = {
            "a": ("field_x",),
            "b": ("field_y", "field_z"),
        }
        executor.register_coalesce(s, "node_1", branch_schemas)

        # Branch a arrives
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        # Branch b is lost
        result = executor.notify_branch_lost("merge", "row_1", "b", "diverted")

        # best_effort should merge after loss notification
        assert result is not None
        assert result.merged_token is not None
        assert result.coalesce_metadata.lost_branch_expected_fields == {"b": ("field_y", "field_z")}

    def test_merge_metadata_no_lost_branches_has_none_expected_fields(self):
        """When no branches are lost, lost_branch_expected_fields is None."""
        executor, *_ = _make_executor()
        s = _settings(branches=["a", "b"], policy="require_all")
        branch_schemas = {
            "a": ("field_x",),
            "b": ("field_y",),
        }
        executor.register_coalesce(s, "node_1", branch_schemas)

        # Both branches arrive
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        result = executor.accept(_make_token(branch_name="b", token_id="t2"), "merge")

        # Successful merge with no lost branches
        assert result.merged_token is not None
        assert result.coalesce_metadata.lost_branch_expected_fields is None

    def test_failure_serialization_includes_lost_branch_expected_fields(self):
        """to_dict() serializes lost_branch_expected_fields correctly."""
        executor, *_ = _make_executor()
        s = _settings(branches=["a", "b"], policy="require_all")
        branch_schemas = {
            "a": ("field_x", "field_y"),
            "b": ("field_z",),
        }
        executor.register_coalesce(s, "node_1", branch_schemas)

        result = executor.notify_branch_lost("merge", "row_1", "b", "upstream_fail")

        serialized = result.coalesce_metadata.to_dict()
        assert "lost_branch_expected_fields" in serialized
        assert serialized["lost_branch_expected_fields"] == {"b": ["field_z"]}

    def test_check_timeouts_includes_lost_branch_expected_fields(self):
        """check_timeouts path records lost_branch_expected_fields in merge metadata."""
        executor, _, _, _, clock = _make_executor()
        # Use 3 branches so losing one doesn't immediately trigger merge
        # (best_effort merges when all branches accounted for OR on timeout)
        s = _settings(branches=["a", "b", "c"], policy="best_effort", timeout_seconds=5.0)
        branch_schemas = {
            "a": ("field_x",),
            "b": ("field_y", "field_z"),
            "c": ("field_w",),
        }
        executor.register_coalesce(s, "node_1", branch_schemas)

        # Branch a arrives
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        # Branch b is lost (accounted = 2, expected = 3, so no immediate merge)
        result = executor.notify_branch_lost("merge", "row_1", "b", "diverted")
        assert result is None  # Not all accounted yet

        # Advance past timeout — this triggers merge via check_timeouts
        # (c never arrived, but best_effort merges with whatever we have)
        clock.advance(6.0)
        results = executor.check_timeouts("merge")

        # Should have one result with lost_branch_expected_fields
        assert len(results) == 1
        result = results[0]
        assert result.merged_token is not None
        assert result.coalesce_metadata.lost_branch_expected_fields == {"b": ("field_y", "field_z")}

    def test_flush_pending_includes_lost_branch_expected_fields(self):
        """flush_pending path records lost_branch_expected_fields in merge metadata."""
        executor, *_ = _make_executor()
        # Use best_effort so loss doesn't trigger immediate fail, and flush will merge
        # Long timeout ensures it won't fire before flush_pending
        s = _settings(branches=["a", "b", "c"], policy="best_effort", timeout_seconds=3600.0)
        branch_schemas = {
            "a": ("field_1",),
            "b": ("field_2",),
            "c": ("field_3",),
        }
        executor.register_coalesce(s, "node_1", branch_schemas)

        # Only branch a arrives
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        # Branch b is lost (accounted = 2/3, not all yet)
        result = executor.notify_branch_lost("merge", "row_1", "b", "error")
        assert result is None  # Not all accounted yet

        # Flush at end-of-source — best_effort merges with whatever we have
        results = executor.flush_pending()

        # Should have one merge result with lost_branch_expected_fields for b
        assert len(results) == 1
        result = results[0]
        assert result.merged_token is not None
        assert result.coalesce_metadata.lost_branch_expected_fields == {"b": ("field_2",)}


# ===========================================================================
# Bug D3-2: best_effort timeout with zero arrivals
# ===========================================================================


class TestBestEffortTimeoutZeroArrivals:
    """Regression tests for best_effort coalesce timeout with zero arrivals.

    When a pending coalesce has best_effort policy and all branches are lost
    via notify_branch_lost (len(pending.arrived) == 0), the timeout check
    must fail and clean up the entry rather than leaving it in _pending forever.
    """

    def test_best_effort_timeout_with_zero_arrivals_fails_and_cleans_up(self):
        """notify_branch_lost only, clock advances past timeout, check_timeouts
        returns a failure outcome and removes from _pending.
        """
        executor, _execution, _data_flow, _, clock = _make_executor()
        s = _settings(
            branches=["a", "b"],
            policy="best_effort",
            merge="union",
            timeout_seconds=10.0,
        )
        executor.register_coalesce(s, "node_1")

        # Lose branch "a" — this creates a pending entry with no arrivals
        result_a = executor.notify_branch_lost("merge", "row_1", "a", "error_a")
        # best_effort with 1 lost + 0 arrived < 2 total branches — still waiting
        assert result_a is None

        # Advance clock past timeout
        clock.advance(11.0)

        # check_timeouts should now detect the timed-out entry and fail it
        results = executor.check_timeouts("merge")
        assert len(results) == 1

        outcome = results[0]
        assert outcome.held is False
        assert outcome.merged_token is None
        assert outcome.failure_reason == "best_effort_timeout_no_arrivals"
        assert outcome.outcomes_recorded is True

        # The key should be removed from _pending
        assert ("merge", "row_1") not in executor._pending

        # The key should be marked as completed (for late-arrival detection)
        assert ("merge", "row_1") in executor._completed_keys

    def test_best_effort_timeout_with_arrivals_still_merges(self):
        """Confirm the existing behavior: best_effort with arrivals merges on timeout."""
        executor, _execution, _data_flow, _, clock = _make_executor()
        s = _settings(
            branches=["a", "b"],
            policy="best_effort",
            merge="union",
            timeout_seconds=10.0,
        )
        executor.register_coalesce(s, "node_1")

        # Accept token for branch "a"
        outcome_a = executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        assert outcome_a.held is True

        # Advance clock past timeout (branch "b" never arrives)
        clock.advance(11.0)

        results = executor.check_timeouts("merge")
        assert len(results) == 1
        assert results[0].merged_token is not None
        assert results[0].failure_reason is None

        # _pending should be cleaned up
        assert ("merge", "row_1") not in executor._pending

    def test_best_effort_timeout_zero_arrivals_with_all_branches_lost(self):
        """When all branches are lost and timeout fires, fail cleanly."""
        executor, _execution, _data_flow, _, clock = _make_executor()
        s = _settings(
            branches=["a", "b", "c"],
            policy="best_effort",
            merge="union",
            timeout_seconds=5.0,
        )
        executor.register_coalesce(s, "node_1")

        # Lose branch "a" — creates pending, but _evaluate_after_loss returns None
        # because only 1/3 branches accounted for
        result = executor.notify_branch_lost("merge", "row_1", "a", "error_a")
        assert result is None

        # Lose branch "b" — 2/3 accounted for, still waiting
        result = executor.notify_branch_lost("merge", "row_1", "b", "error_b")
        assert result is None

        # Advance past timeout before branch "c" is lost or arrives
        clock.advance(6.0)

        results = executor.check_timeouts("merge")
        assert len(results) == 1
        outcome = results[0]
        assert outcome.failure_reason == "best_effort_timeout_no_arrivals"
        assert outcome.outcomes_recorded is True
        assert ("merge", "row_1") not in executor._pending

    def test_best_effort_timeout_zero_arrivals_does_not_leave_entry_in_pending(self):
        """The primary regression: ensure the entry is actually removed from _pending."""
        executor, _, _, _, clock = _make_executor()
        s = _settings(
            branches=["a", "b"],
            policy="best_effort",
            merge="union",
            timeout_seconds=3.0,
        )
        executor.register_coalesce(s, "node_1")

        # Lose one branch (creates pending with 0 arrivals, 1 lost)
        executor.notify_branch_lost("merge", "row_1", "a", "error")

        # Verify it's in _pending before timeout
        assert ("merge", "row_1") in executor._pending

        clock.advance(4.0)
        executor.check_timeouts("merge")

        # After timeout, it MUST be gone from _pending (this was the bug)
        assert ("merge", "row_1") not in executor._pending


# ===========================================================================
# Bug: completed_keys lost on checkpoint restore
# ===========================================================================


class TestCheckpointCompletedKeys:
    """Regression tests for _completed_keys surviving checkpoint/restore.

    Bug: get_checkpoint_state() did not serialize _completed_keys, and
    restore_from_checkpoint() called _completed_keys.clear(). After a
    checkpoint/restore cycle, late arrivals were not detected — creating
    contradictory audit outcomes (both COALESCED and FAILED/timeout for
    the same fork group).
    """

    def test_completed_keys_restored_from_landscape_on_resume(self):
        """On restore, completed keys are reconstructed from Landscape, not checkpoint.

        The Landscape is the source of truth. The checkpoint's completed_keys
        field is ignored — the Landscape query returns ALL completed coalesces
        (no FIFO eviction limit).
        """
        executor, _execution, _data_flow, _, _clock = _make_executor()
        executor.register_coalesce(_settings(branches=["a", "b"]), "node_1")

        # Complete a coalesce: both branches arrive → merge
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        executor.accept(_make_token(branch_name="b", token_id="t2"), "merge")

        # Verify the key is marked completed
        assert ("merge", "row_1") in executor._completed_keys

        # Checkpoint (completed_keys is now written as empty)
        checkpoint = executor.get_checkpoint_state()
        assert checkpoint.completed_keys == ()  # No longer serialized

        # Build a new executor and restore, with Landscape mock returning the completed key
        executor2, _execution2, _data_flow2, _, _clock2 = _make_executor()
        executor2.register_coalesce(_settings(branches=["a", "b"]), "node_1")
        _execution2.get_completed_row_ids_for_nodes.return_value = {("node_1", "row_1")}
        executor2.restore_from_checkpoint(checkpoint)

        # Late arrival on the restored executor must be detected (via Landscape)
        assert ("merge", "row_1") in executor2._completed_keys
        late = _make_token(branch_name="a", token_id="t_late", row_id="row_1")
        outcome = executor2.accept(late, "merge")

        assert outcome.held is False
        assert outcome.failure_reason == "late_arrival_after_merge"
        assert outcome.outcomes_recorded is True

    def test_get_checkpoint_state_does_not_serialize_for_size_check(self, monkeypatch: pytest.MonkeyPatch):
        """Checkpoint state construction must not pre-serialize the DTO.

        The CheckpointManager is the persistence boundary that serializes the
        state.  Serializing here purely for size validation doubles checkpoint
        cost on every interrupted coalesce run.
        """
        import elspeth.engine.coalesce_executor as coalesce_module

        def fail_if_called(_obj: object) -> str:
            raise AssertionError("CoalesceExecutor.get_checkpoint_state must not serialize")

        monkeypatch.setattr(coalesce_module, "checkpoint_dumps", fail_if_called, raising=False)

        executor, _execution, _data_flow, _, _clock = _make_executor()
        executor.register_coalesce(_settings(branches=["a", "b"]), "node_1")
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")

        checkpoint = executor.get_checkpoint_state()

        assert checkpoint.pending

    def test_landscape_reconstruction_restores_all_completed_keys(self):
        """Multiple completed keys restored from Landscape, not checkpoint."""
        executor, _, _, _, _clock = _make_executor()
        s = _settings(branches=["a", "b"])
        executor.register_coalesce(s, "node_1")

        # Complete 3 different row coalesces
        for i in range(3):
            row_id = f"row_{i}"
            executor.accept(_make_token(row_id=row_id, branch_name="a", token_id=f"t{i}_a"), "merge")
            executor.accept(_make_token(row_id=row_id, branch_name="b", token_id=f"t{i}_b"), "merge")

        assert len(executor._completed_keys) == 3

        # Checkpoint → restore with Landscape mock
        checkpoint = executor.get_checkpoint_state()
        executor2, execution2, _, _, _ = _make_executor()
        executor2.register_coalesce(s, "node_1")
        execution2.get_completed_row_ids_for_nodes.return_value = {("node_1", f"row_{i}") for i in range(3)}
        executor2.restore_from_checkpoint(checkpoint)

        # All 3 completed keys must be present (from Landscape)
        for i in range(3):
            assert ("merge", f"row_{i}") in executor2._completed_keys

    def test_landscape_fallback_catches_evicted_keys(self):
        """After FIFO eviction, the Landscape fallback still detects late arrivals."""
        executor, execution, _, _, _ = _make_executor(max_completed_keys=2)
        s = _settings(branches=["a", "b"])
        executor.register_coalesce(s, "node_1")

        # Complete 3 coalesces — row_0 will be evicted from FIFO (max=2)
        for i in range(3):
            row_id = f"row_{i}"
            executor.accept(_make_token(row_id=row_id, branch_name="a", token_id=f"t{i}_a"), "merge")
            executor.accept(_make_token(row_id=row_id, branch_name="b", token_id=f"t{i}_b"), "merge")

        # row_0 was evicted from FIFO
        assert ("merge", "row_0") not in executor._completed_keys
        # row_1 and row_2 remain in FIFO
        assert ("merge", "row_1") in executor._completed_keys
        assert ("merge", "row_2") in executor._completed_keys

        # Late arrival for evicted row_0 — Landscape fallback should catch it
        execution.get_completed_row_ids_for_nodes.return_value = {("node_1", "row_0")}
        late = _make_token(branch_name="a", token_id="t_late", row_id="row_0")
        outcome = executor.accept(late, "merge")

        assert outcome.held is False
        assert outcome.failure_reason == "late_arrival_after_merge"
        # Key should now be in the FIFO cache (backfilled from Landscape)
        assert ("merge", "row_0") in executor._completed_keys

    def test_pending_and_completed_both_restored(self):
        """Checkpoint with both pending and completed entries restores both."""
        executor, _, _, _, _clock = _make_executor()
        s = _settings(branches=["a", "b"])
        executor.register_coalesce(s, "node_1")

        # Complete row_1
        executor.accept(_make_token(row_id="row_1", branch_name="a", token_id="t1a"), "merge")
        executor.accept(_make_token(row_id="row_1", branch_name="b", token_id="t1b"), "merge")

        # Leave row_2 pending (only branch a arrived)
        executor.accept(_make_token(row_id="row_2", branch_name="a", token_id="t2a"), "merge")

        assert ("merge", "row_1") in executor._completed_keys
        assert ("merge", "row_2") in executor._pending

        checkpoint = executor.get_checkpoint_state()

        executor2, execution2, _, _, _ = _make_executor()
        executor2.register_coalesce(s, "node_1")
        # Mock Landscape to return completed key for row_1
        execution2.get_completed_row_ids_for_nodes.return_value = {("node_1", "row_1")}
        executor2.restore_from_checkpoint(checkpoint)

        # Completed key restored (from Landscape)
        assert ("merge", "row_1") in executor2._completed_keys
        # Pending entry restored (from checkpoint)
        assert ("merge", "row_2") in executor2._pending

    def test_checkpoint_dto_writes_empty_completed_keys(self):
        """Checkpoint no longer serializes completed_keys (Landscape is source of truth)."""
        executor, _, _, _, _ = _make_executor()
        s = _settings(branches=["a", "b"])
        executor.register_coalesce(s, "node_1")

        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        executor.accept(_make_token(branch_name="b", token_id="t2"), "merge")

        checkpoint = executor.get_checkpoint_state()
        wire = checkpoint.to_dict()

        # Wire format still includes the field (schema compat) but it's empty
        assert "completed_keys" in wire
        assert wire["completed_keys"] == []

    def test_checkpoint_dto_from_dict_restores_completed_keys(self):
        """CoalesceCheckpointState.from_dict should parse completed_keys."""
        from elspeth.contracts.coalesce_checkpoint import CoalesceCheckpointState

        wire = {
            "_version": "1.0",
            "pending": [],
            "completed_keys": [["merge", "row_1"], ["merge", "row_2"]],
        }
        state = CoalesceCheckpointState.from_dict(wire)
        assert state.completed_keys == (("merge", "row_1"), ("merge", "row_2"))


class TestRestoreFromCheckpoint:
    """Tests for restore_from_checkpoint version and coalesce name validation."""

    def test_rejects_incompatible_checkpoint_version(self):
        """restore_from_checkpoint raises AuditIntegrityError on version mismatch."""
        from elspeth.contracts.coalesce_checkpoint import CoalesceCheckpointState

        executor, *_ = _make_executor()
        executor.register_coalesce(_settings(name="merge"), "node_1")

        state = CoalesceCheckpointState(
            version="0.0-bogus",
            pending=(),
            completed_keys=(),
        )
        with pytest.raises(AuditIntegrityError, match="Incompatible coalesce checkpoint version"):
            executor.restore_from_checkpoint(state)

    def test_rejects_unknown_coalesce_name_in_pending(self):
        """restore_from_checkpoint raises AuditIntegrityError for unregistered coalesce names."""
        from elspeth.contracts.coalesce_checkpoint import (
            CoalesceCheckpointState,
            CoalescePendingCheckpoint,
            CoalesceTokenCheckpoint,
        )
        from elspeth.engine.coalesce_executor import COALESCE_CHECKPOINT_VERSION

        executor, *_ = _make_executor()
        executor.register_coalesce(_settings(name="merge"), "node_1")

        contract = _make_contract(mode="FLEXIBLE")
        token_cp = CoalesceTokenCheckpoint(
            token_id="tok_1",
            row_id="row_1",
            branch_name="a",
            fork_group_id=None,
            join_group_id=None,
            expand_group_id=None,
            row_data={"amount": 100},
            contract=contract.to_checkpoint_format(),
            state_id="state_001",
            arrival_offset_seconds=0.5,
        )
        pending = CoalescePendingCheckpoint(
            coalesce_name="nonexistent_merge",
            row_id="row_1",
            elapsed_age_seconds=1.0,
            branches={"a": token_cp},
            lost_branches={},
        )
        state = CoalesceCheckpointState(
            version=COALESCE_CHECKPOINT_VERSION,
            pending=(pending,),
            completed_keys=(),
        )
        with pytest.raises(AuditIntegrityError, match="unknown coalesce 'nonexistent_merge'"):
            executor.restore_from_checkpoint(state)


# ===========================================================================
# _should_merge mutation survivors — direct policy boundary tests
# ===========================================================================


class TestShouldMergeMutationGaps:
    """Kill mutants in CoalesceExecutor._should_merge().

    These test _should_merge directly to verify exact boundary conditions
    that can't be tested through accept() (which always adds a branch
    before calling _should_merge, so arrived_count is always >= 1).
    """

    def test_first_policy_does_not_fire_at_zero_arrivals(self) -> None:
        """Kill mutant: ``arrived_count >= 1`` → ``>= 0``.

        With >= 0, the "first" policy would trigger before any branch
        arrives, producing a merged token with no data (silent data loss).
        """
        executor, _, _, _, _ = _make_executor()
        s = _settings(branches=["a", "b", "c"], policy="first")
        executor.register_coalesce(s, NodeID("node_1"))

        # Create pending with 0 arrivals
        pending = _PendingCoalesce(branches={}, first_arrival=100.0)
        assert executor._should_merge(s, pending) is False

    def test_first_policy_fires_at_one_arrival(self) -> None:
        """Confirm "first" policy fires at exactly 1 arrival."""
        executor, _, _, _, _ = _make_executor()
        s = _settings(branches=["a", "b", "c"], policy="first")

        pending = _PendingCoalesce(
            branches={"a": _BranchEntry(token=_make_token(branch_name="a"), arrival_time=100.0, state_id="s1")},
            first_arrival=100.0,
        )
        assert executor._should_merge(s, pending) is True

    def test_best_effort_lost_branches_add_not_subtract(self) -> None:
        """Kill mutant: ``arrived_count + len(lost_branches)`` → ``- len(lost_branches)``.

        With subtraction, a 3-branch fork where 1 is lost and 2 arrive
        computes 2 - 1 = 1 instead of 2 + 1 = 3 — merge never triggers,
        tokens stuck in barrier forever (silent row drop).
        """
        executor, _, _, _, _ = _make_executor()
        s = _settings(branches=["a", "b", "c"], policy="best_effort", timeout_seconds=60.0)

        pending = _PendingCoalesce(
            branches={
                "a": _BranchEntry(token=_make_token(branch_name="a"), arrival_time=100.0, state_id="s1"),
                "b": _BranchEntry(token=_make_token(branch_name="b", token_id="t2"), arrival_time=100.1, state_id="s2"),
            },
            first_arrival=100.0,
            lost_branches={"c": "error_routed"},
        )
        # Correct: 2 + 1 = 3 >= 3 → True
        # Mutant:  2 - 1 = 1 >= 3 → False
        assert executor._should_merge(s, pending) is True

    def test_best_effort_does_not_merge_when_unaccounted(self) -> None:
        """Confirm best_effort does NOT merge when branches are still unaccounted."""
        executor, _, _, _, _ = _make_executor()
        s = _settings(branches=["a", "b", "c"], policy="best_effort", timeout_seconds=60.0)

        pending = _PendingCoalesce(
            branches={"a": _BranchEntry(token=_make_token(branch_name="a"), arrival_time=100.0, state_id="s1")},
            first_arrival=100.0,
        )
        # 1 arrived + 0 lost = 1 < 3 expected
        assert executor._should_merge(s, pending) is False

    def test_quorum_fires_when_arrivals_exceed_quorum_count(self) -> None:
        """Kill mutant: ``arrived_count >= settings.quorum_count`` → ``==``.

        With ``==``, 3 arrivals with quorum_count=2 would compute
        3 == 2 → False, and the merge would never trigger despite
        exceeding the quorum threshold.
        """
        executor, _, _, _, _ = _make_executor()
        s = _settings(branches=["a", "b", "c"], policy="quorum", quorum_count=2)

        pending = _PendingCoalesce(
            branches={
                "a": _BranchEntry(token=_make_token(branch_name="a"), arrival_time=100.0, state_id="s1"),
                "b": _BranchEntry(token=_make_token(branch_name="b", token_id="t2"), arrival_time=100.1, state_id="s2"),
                "c": _BranchEntry(token=_make_token(branch_name="c", token_id="t3"), arrival_time=100.2, state_id="s3"),
            },
            first_arrival=100.0,
        )
        # Correct: 3 >= 2 → True
        # Mutant:  3 == 2 → False
        assert executor._should_merge(s, pending) is True


# ===========================================================================
# Nested merge produces locked contract
# ===========================================================================


class TestNestedMergeContractLocked:
    """Kill mutant: ``locked=True`` → ``locked=False`` in nested merge contract.

    An unlocked contract would allow downstream transforms to mutate the
    schema after the merge — violating the FIXED contract invariant.
    """

    def test_nested_merge_produces_locked_contract(self) -> None:
        executor, _, _, _tm, _ = _make_executor()
        s = _settings(policy="first", merge="nested")
        executor.register_coalesce(s, NodeID("node_1"))

        t = _make_token(branch_name="a", token_id="t1", data={"x": 1})
        o = executor.accept(t, "merge")

        assert o.held is False
        assert o.merged_token is not None
        assert o.merged_token.row_data.contract.locked is True


# ===========================================================================
# select_branch not arrived returns held=False
# ===========================================================================


class TestSelectBranchNotArrivedFailure:
    """Kill mutant: ``held=False`` → ``held=True`` in select_branch_not_arrived path.

    With ``held=True``, the caller would treat the failure as still-pending,
    leaving tokens stuck in the barrier forever.
    """

    def test_select_branch_not_arrived_returns_held_false(self) -> None:
        executor, _, _, _, _ = _make_executor()
        s = _settings(
            branches=["a", "b", "c"],
            policy="best_effort",
            merge="select",
            select_branch="c",
            timeout_seconds=60.0,
        )
        executor.register_coalesce(s, NodeID("node_1"))

        # Accept tokens for branches "a" and "b"
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        executor.accept(_make_token(branch_name="b", token_id="t2"), "merge")

        # Notify branch "c" (the select_branch) as lost — makes all accounted for
        outcome = executor.notify_branch_lost("merge", "row_1", "c", "error_routed")

        # Merge triggers (3 accounted = 3 expected), but select_branch="c" not arrived
        assert outcome is not None
        assert outcome.held is False
        assert outcome.failure_reason == "select_branch_not_arrived"
        assert outcome.outcomes_recorded is True


# ===========================================================================
# Checkpoint/restore round-trip (elspeth-c44cdc5ffe)
# ===========================================================================


class TestCheckpointRestoreRoundTrip:
    """Verify that partial coalesce state survives checkpoint/restore and
    the restored executor can complete the merge successfully.

    Bug: elspeth-c44cdc5ffe — Coalesce checkpoint/restore round-trip not tested.
    """

    def test_partial_accept_checkpoint_restore_then_merge(self):
        """Accept branch a → checkpoint → restore to new executor → accept branch b → merge succeeds."""
        executor, _execution, _data_flow, _tm, _clock = _make_executor()
        s = _settings(branches=["a", "b"], policy="require_all")
        executor.register_coalesce(s, "node_1")

        # Accept first branch
        outcome_a = executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        assert outcome_a.held is True

        # Checkpoint
        checkpoint = executor.get_checkpoint_state()
        assert len(checkpoint.pending) == 1
        assert checkpoint.pending[0].coalesce_name == "merge"
        assert checkpoint.pending[0].row_id == "row_1"
        assert "a" in checkpoint.pending[0].branches

        # Restore to a fresh executor
        executor2, _execution2, _data_flow2, _tm2, _clock2 = _make_executor()
        executor2.register_coalesce(s, "node_1")
        executor2.restore_from_checkpoint(checkpoint)

        # Verify pending state was restored
        assert ("merge", "row_1") in executor2._pending
        pending = executor2._pending[("merge", "row_1")]
        assert "a" in pending.branches
        assert pending.branches["a"].token.token_id == "t1"
        assert pending.branches["a"].token.row_data.to_dict() == {"amount": 100}

        # Accept second branch on restored executor — should trigger merge
        outcome_b = executor2.accept(
            _make_token(branch_name="b", token_id="t2", data={"amount": 200}),
            "merge",
        )
        assert outcome_b.held is False
        assert outcome_b.merged_token is not None
        assert outcome_b.failure_reason is None

    def test_lost_branches_survive_checkpoint_restore(self):
        """lost_branches recorded before checkpoint must be present after restore."""
        executor, *_ = _make_executor()
        s = _settings(branches=["a", "b", "c"], policy="best_effort", timeout_seconds=60.0)
        executor.register_coalesce(s, "node_1")

        # Accept one branch and lose another
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        executor.notify_branch_lost("merge", "row_1", "b", "upstream_error")

        # Verify lost_branches in pending
        pending = executor._pending[("merge", "row_1")]
        assert pending.lost_branches == {"b": "upstream_error"}

        # Checkpoint
        checkpoint = executor.get_checkpoint_state()
        assert checkpoint.pending[0].lost_branches == {"b": "upstream_error"}

        # Restore to fresh executor
        executor2, *_ = _make_executor()
        executor2.register_coalesce(s, "node_1")
        executor2.restore_from_checkpoint(checkpoint)

        # lost_branches must survive
        restored_pending = executor2._pending[("merge", "row_1")]
        assert restored_pending.lost_branches == {"b": "upstream_error"}

        # Losing the last remaining branch should trigger merge (all accounted for)
        outcome = executor2.notify_branch_lost("merge", "row_1", "c", "timeout")
        assert outcome is not None
        # arrived=1 + lost=2 = 3 = total_branches → merge with the one arrived branch
        assert outcome.merged_token is not None

    def test_lost_branches_checkpoint_then_accept_remaining_merges(self):
        """After restoring lost_branches, accepting the remaining branch completes the merge."""
        executor, *_ = _make_executor()
        s = _settings(branches=["a", "b", "c"], policy="best_effort", timeout_seconds=60.0)
        executor.register_coalesce(s, "node_1")

        # Lose branch a, accept branch b
        executor.notify_branch_lost("merge", "row_1", "a", "error_routed")
        executor.accept(_make_token(branch_name="b", token_id="t1"), "merge")

        # Checkpoint with 1 lost, 1 arrived, 1 pending
        checkpoint = executor.get_checkpoint_state()

        # Restore
        executor2, *_ = _make_executor()
        executor2.register_coalesce(s, "node_1")
        executor2.restore_from_checkpoint(checkpoint)

        # Accept remaining branch c — all 3 accounted for (1 lost + 2 arrived)
        outcome = executor2.accept(
            _make_token(branch_name="c", token_id="t2", data={"amount": 300}),
            "merge",
        )
        assert outcome.held is False
        assert outcome.merged_token is not None

    def test_checkpoint_preserves_arrival_time_offsets(self):
        """Branch arrival time offsets survive round-trip for correct timeout evaluation."""
        clock = MockClock(start=100.0)
        executor, *_ = _make_executor(clock=clock)
        s = _settings(branches=["a", "b", "c"], policy="require_all", timeout_seconds=30.0)
        executor.register_coalesce(s, "node_1")

        # Accept branch a at t=100
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")

        # Advance 10s, accept branch b at t=110
        clock.advance(10.0)
        executor.accept(_make_token(branch_name="b", token_id="t2"), "merge")

        # Checkpoint at t=110
        checkpoint = executor.get_checkpoint_state()

        # Verify offsets in checkpoint
        branches = checkpoint.pending[0].branches
        assert branches["a"].arrival_offset_seconds == pytest.approx(0.0)
        assert branches["b"].arrival_offset_seconds == pytest.approx(10.0)

        # Restore at t=200 (different clock)
        clock2 = MockClock(start=200.0)
        executor2, *_ = _make_executor(clock=clock2)
        executor2.register_coalesce(s, "node_1")
        executor2.restore_from_checkpoint(checkpoint)

        # The restored pending should maintain the relative offset between branches
        restored = executor2._pending[("merge", "row_1")]
        arrival_a = restored.branches["a"].arrival_time
        arrival_b = restored.branches["b"].arrival_time
        assert (arrival_b - arrival_a) == pytest.approx(10.0)

    def test_version_mismatch_raises_audit_integrity_error(self):
        """Restoring from a checkpoint with wrong version must raise AuditIntegrityError."""
        from elspeth.contracts.coalesce_checkpoint import CoalesceCheckpointState

        executor, *_ = _make_executor()
        executor.register_coalesce(_settings(), "node_1")

        state = CoalesceCheckpointState(
            version="99.99",
            pending=(),
            completed_keys=(),
        )
        with pytest.raises(AuditIntegrityError, match="Incompatible coalesce checkpoint version"):
            executor.restore_from_checkpoint(state)

    def test_checkpoint_row_data_survives_round_trip(self):
        """Row data and schema contract must be identical after checkpoint/restore."""
        contract = _make_contract(
            [
                make_field("amount", original_name="amount", python_type=int, required=True, source="declared"),
                make_field("label", original_name="label", python_type=str, required=False, source="declared"),
            ],
            mode="FLEXIBLE",
        )
        executor, *_ = _make_executor()
        s = _settings(branches=["a", "b"], policy="require_all")
        executor.register_coalesce(s, "node_1")

        token = _make_token(
            branch_name="a",
            token_id="t1",
            data={"amount": 42, "label": "test"},
            contract=contract,
        )
        executor.accept(token, "merge")

        checkpoint = executor.get_checkpoint_state()
        executor2, *_ = _make_executor()
        executor2.register_coalesce(s, "node_1")
        executor2.restore_from_checkpoint(checkpoint)

        restored = executor2._pending[("merge", "row_1")]
        restored_token = restored.branches["a"].token
        assert restored_token.row_data.to_dict() == {"amount": 42, "label": "test"}
        assert restored_token.token_id == "t1"
        assert restored_token.row_id == "row_1"
        assert restored_token.branch_name == "a"


# ===========================================================================
# notify_branch_lost / _evaluate_after_loss (elspeth-bc0461a50e)
# ===========================================================================


class TestNotifyBranchLostEvaluateAfterLoss:
    """Targeted tests for notify_branch_lost and _evaluate_after_loss logic.

    Bug: elspeth-bc0461a50e — notify_branch_lost and _evaluate_after_loss untested.

    Tests are grouped by merge policy to verify each policy's loss-handling semantics.
    """

    # --- require_all policy ---

    def test_require_all_single_loss_immediate_failure(self):
        """require_all: ANY lost branch causes immediate failure, even with no arrivals."""
        executor, *_ = _make_executor()
        s = _settings(branches=["a", "b", "c"], policy="require_all")
        executor.register_coalesce(s, "node_1")

        result = executor.notify_branch_lost("merge", "row_1", "a", "upstream_crash")
        assert result is not None
        assert result.held is False
        assert result.failure_reason is not None
        assert "branch_lost" in result.failure_reason
        assert "a" in result.failure_reason
        assert result.outcomes_recorded is True

    def test_require_all_loss_after_partial_arrivals_fails(self):
        """require_all: loss after some branches arrived still fails immediately."""
        executor, *_ = _make_executor()
        s = _settings(branches=["a", "b", "c"], policy="require_all")
        executor.register_coalesce(s, "node_1")

        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        executor.accept(_make_token(branch_name="b", token_id="t2"), "merge")

        result = executor.notify_branch_lost("merge", "row_1", "c", "error_routed")
        assert result is not None
        assert "branch_lost" in result.failure_reason
        assert "c" in result.failure_reason
        assert result.outcomes_recorded is True
        # Consumed tokens should include the arrived branches
        assert len(result.consumed_tokens) == 2

    def test_require_all_failure_metadata_includes_lost_branches(self):
        """require_all failure metadata should record which branches were lost."""
        executor, *_ = _make_executor()
        s = _settings(branches=["a", "b", "c"], policy="require_all")
        executor.register_coalesce(s, "node_1")

        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        result = executor.notify_branch_lost("merge", "row_1", "b", "error_routed")
        assert result.coalesce_metadata is not None
        assert result.coalesce_metadata.branches_lost == {"b": "error_routed"}

    # --- quorum policy ---

    @pytest.mark.filterwarnings("ignore:Coalesce.*quorum_count.*equals branch count:UserWarning")
    def test_quorum_loss_makes_quorum_impossible(self):
        """quorum: when remaining live branches < quorum_count, fail immediately."""
        executor, *_ = _make_executor()
        s = _settings(branches=["a", "b", "c"], policy="quorum", quorum_count=3)
        executor.register_coalesce(s, "node_1")

        # Lose one branch: max_possible = 3-1 = 2 < quorum_count=3
        result = executor.notify_branch_lost("merge", "row_1", "a", "error_routed")
        assert result is not None
        assert "quorum_impossible" in result.failure_reason
        assert "need=3" in result.failure_reason
        assert "max_possible=2" in result.failure_reason

    def test_quorum_loss_still_achievable_returns_none(self):
        """quorum: if quorum is still achievable after loss, keep waiting."""
        executor, *_ = _make_executor()
        s = _settings(branches=["a", "b", "c", "d"], policy="quorum", quorum_count=2)
        executor.register_coalesce(s, "node_1")

        # Lose one: max_possible = 4-1 = 3 >= 2, arrived=0 < 2 → wait
        result = executor.notify_branch_lost("merge", "row_1", "d", "error_routed")
        assert result is None

    def test_quorum_one_arrival_one_loss_still_meets_quorum(self):
        """quorum: 1 arrived + 1 lost with quorum=1 — quorum met, merge triggers."""
        executor, *_ = _make_executor()
        # 3 branches, quorum=1: one arrival should meet quorum on accept.
        # Instead: 4 branches, quorum=2, 1 arrived, 1 lost.
        # max_possible = 4-1 = 3 >= 2 (still possible), arrived=1 < 2 → wait on first loss.
        # Then accept 2nd → quorum met on accept. That path is through _should_merge.
        # Test the path in _evaluate_after_loss where arrived >= quorum after loss:
        # Use _evaluate_after_loss directly since accept() eagerly merges on quorum.
        s = _settings(branches=["a", "b", "c"], policy="quorum", quorum_count=2)
        executor.register_coalesce(s, "node_1")

        # Accept one branch — quorum not met (1 < 2), held
        outcome = executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        assert outcome.held is True

        # Lose one branch — max_possible=2 >= 2 (still possible), arrived=1 < 2 → wait
        result = executor.notify_branch_lost("merge", "row_1", "b", "error_routed")
        assert result is None

        # Accept second branch — quorum met (2 >= 2), merge triggers via _should_merge
        outcome2 = executor.accept(_make_token(branch_name="c", token_id="t2"), "merge")
        assert outcome2.held is False
        assert outcome2.merged_token is not None

    # --- best_effort policy ---

    def test_best_effort_all_accounted_after_loss_triggers_merge(self):
        """best_effort: merge fires when arrived + lost == total branches."""
        executor, *_ = _make_executor()
        s = _settings(branches=["a", "b", "c"], policy="best_effort", timeout_seconds=60.0)
        executor.register_coalesce(s, "node_1")

        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        executor.accept(_make_token(branch_name="b", token_id="t2"), "merge")

        # Losing c means all 3 accounted for (2 arrived + 1 lost)
        result = executor.notify_branch_lost("merge", "row_1", "c", "error_routed")
        assert result is not None
        assert result.merged_token is not None
        assert result.failure_reason is None

    def test_best_effort_not_all_accounted_returns_none(self):
        """best_effort: if some branches remain unaccounted, keep waiting."""
        executor, *_ = _make_executor()
        s = _settings(branches=["a", "b", "c", "d"], policy="best_effort", timeout_seconds=60.0)
        executor.register_coalesce(s, "node_1")

        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        # Lose b: arrived=1 + lost=1 = 2 < 4 total → wait
        result = executor.notify_branch_lost("merge", "row_1", "b", "error_routed")
        assert result is None

    def test_best_effort_all_lost_no_arrivals_fails(self):
        """best_effort: if all branches are lost with zero arrivals, fail."""
        executor, *_ = _make_executor()
        s = _settings(branches=["a", "b"], policy="best_effort", timeout_seconds=60.0)
        executor.register_coalesce(s, "node_1")

        executor.notify_branch_lost("merge", "row_1", "a", "error1")
        result = executor.notify_branch_lost("merge", "row_1", "b", "error2")

        assert result is not None
        assert result.failure_reason == "all_branches_lost"
        assert result.merged_token is None
        assert result.outcomes_recorded is True

    # --- first policy ---

    def test_first_policy_loss_returns_none(self):
        """first: branch loss has no effect — merge should have happened on first arrival."""
        executor, *_ = _make_executor()
        s = _settings(branches=["a", "b", "c"], policy="first")
        executor.register_coalesce(s, "node_1")

        result = executor.notify_branch_lost("merge", "row_1", "b", "error_routed")
        assert result is None

    # --- Edge cases ---

    def test_branch_lost_before_any_branch_arrives_creates_pending(self):
        """Edge case (lines 1148-1156): branch lost with no pending entry creates one."""
        executor, *_ = _make_executor()
        s = _settings(branches=["a", "b", "c"], policy="best_effort", timeout_seconds=60.0)
        executor.register_coalesce(s, "node_1")

        # No accept() calls — no pending entry exists yet
        assert ("merge", "row_1") not in executor._pending

        result = executor.notify_branch_lost("merge", "row_1", "a", "upstream_crash")

        # Pending entry must have been created
        assert ("merge", "row_1") in executor._pending
        pending = executor._pending[("merge", "row_1")]
        assert pending.branches == {}  # No arrived branches
        assert pending.lost_branches == {"a": "upstream_crash"}
        # 1 lost + 0 arrived = 1 < 3 total → still waiting
        assert result is None

    def test_branch_lost_before_any_arrival_require_all_fails_immediately(self):
        """require_all: even with no pending entry, branch loss should fail."""
        executor, *_ = _make_executor()
        s = _settings(branches=["a", "b"], policy="require_all")
        executor.register_coalesce(s, "node_1")

        result = executor.notify_branch_lost("merge", "row_1", "a", "upstream_crash")
        assert result is not None
        assert "branch_lost" in result.failure_reason
        # Key should be completed after failure
        assert ("merge", "row_1") in executor._completed_keys
        assert ("merge", "row_1") not in executor._pending

    def test_duplicate_loss_same_branch_raises_invariant_error(self):
        """Reporting the same branch as lost twice raises OrchestrationInvariantError."""
        executor, *_ = _make_executor()
        s = _settings(branches=["a", "b", "c"], policy="best_effort", timeout_seconds=60.0)
        executor.register_coalesce(s, "node_1")

        executor.notify_branch_lost("merge", "row_1", "a", "first_reason")
        with pytest.raises(OrchestrationInvariantError, match="already marked lost"):
            executor.notify_branch_lost("merge", "row_1", "a", "second_reason")

    def test_lost_branch_that_already_arrived_raises_invariant_error(self):
        """A branch that already arrived cannot be reported as lost."""
        executor, *_ = _make_executor()
        s = _settings(branches=["a", "b"], policy="require_all")
        executor.register_coalesce(s, "node_1")

        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        with pytest.raises(OrchestrationInvariantError, match="already arrived"):
            executor.notify_branch_lost("merge", "row_1", "a", "error_routed")

    def test_loss_for_already_completed_coalesce_returns_none(self):
        """If the coalesce already completed (merged), loss notification is a no-op."""
        executor, *_ = _make_executor()
        s = _settings(branches=["a", "b"], policy="require_all")
        executor.register_coalesce(s, "node_1")

        # Complete the coalesce
        executor.accept(_make_token(branch_name="a", token_id="t1"), "merge")
        executor.accept(_make_token(branch_name="b", token_id="t2"), "merge")
        assert ("merge", "row_1") in executor._completed_keys

        # Loss notification after completion is a no-op
        result = executor.notify_branch_lost("merge", "row_1", "a", "late_error")
        assert result is None

    def test_loss_for_unregistered_coalesce_raises(self):
        """notify_branch_lost on unregistered coalesce raises OrchestrationInvariantError."""
        executor, *_ = _make_executor()
        with pytest.raises(OrchestrationInvariantError, match="not registered"):
            executor.notify_branch_lost("ghost", "row_1", "a", "reason")

    def test_loss_for_unknown_branch_raises(self):
        """notify_branch_lost with unknown branch name raises OrchestrationInvariantError."""
        executor, *_ = _make_executor()
        s = _settings(branches=["a", "b"])
        executor.register_coalesce(s, "node_1")

        with pytest.raises(OrchestrationInvariantError, match="not in expected branches"):
            executor.notify_branch_lost("merge", "row_1", "z", "reason")

    def test_multiple_losses_then_final_arrival_merges(self):
        """best_effort: multiple losses followed by last arrival triggers merge."""
        executor, *_ = _make_executor()
        s = _settings(branches=["a", "b", "c", "d"], policy="best_effort", timeout_seconds=60.0)
        executor.register_coalesce(s, "node_1")

        # Lose 3 branches
        executor.notify_branch_lost("merge", "row_1", "a", "err1")
        executor.notify_branch_lost("merge", "row_1", "b", "err2")
        executor.notify_branch_lost("merge", "row_1", "c", "err3")

        # Accept the last branch — all 4 accounted for (3 lost + 1 arrived)
        outcome = executor.accept(
            _make_token(branch_name="d", token_id="t1", data={"amount": 99}),
            "merge",
        )
        assert outcome.held is False
        assert outcome.merged_token is not None

    @pytest.mark.filterwarnings("ignore:Coalesce.*quorum_count.*equals branch count:UserWarning")
    def test_quorum_loss_before_any_arrival_impossible(self):
        """quorum: loss before any arrival making quorum impossible triggers failure."""
        executor, *_ = _make_executor()
        # 2 branches, quorum=2 — losing either one makes quorum impossible
        s = _settings(branches=["a", "b"], policy="quorum", quorum_count=2)
        executor.register_coalesce(s, "node_1")

        result = executor.notify_branch_lost("merge", "row_1", "a", "error_routed")
        assert result is not None
        assert "quorum_impossible" in result.failure_reason
        assert result.outcomes_recorded is True


class TestPrecomputedOutputSchema:
    """Tests for P2 fix: using pre-computed DAG schema for build/runtime alignment.

    When output_schema is passed to register_coalesce(), the executor uses it
    directly for union merge instead of calling SchemaContract.merge(). This
    ensures runtime contracts match the DAG-computed schema, preserving the
    nullable semantics from the P1 fix.
    """

    def test_union_merge_uses_precomputed_schema_when_provided(self):
        """P2 fix: Runtime contract matches DAG schema when pre-computed schema provided."""
        executor, *_ = _make_executor()

        # Create a pre-computed schema with require_all OR semantics:
        # x is required (OR: required in ANY branch) but nullable (ANY branch allows None)
        precomputed_schema = SchemaContract(
            mode="FLEXIBLE",
            fields=(
                make_field(
                    "x",
                    original_name="x",
                    python_type=int,
                    required=True,  # OR semantics: required if required in ANY branch
                    source="declared",
                    nullable=True,  # P1 fix: nullable because one branch allows None
                ),
            ),
            locked=True,
        )

        # Register coalesce with pre-computed schema
        s = _settings(branches=["a", "b"], policy="require_all", merge="union")
        executor.register_coalesce(s, "node_1", output_schema=precomputed_schema)

        # Create tokens with different runtime contracts (would normally merge differently)
        contract_a = SchemaContract(
            mode="FLEXIBLE",
            fields=(make_field("x", original_name="x", python_type=int, required=True, source="declared", nullable=False),),
            locked=True,
        )
        contract_b = SchemaContract(
            mode="FLEXIBLE",
            fields=(make_field("x", original_name="x", python_type=int, required=False, source="declared", nullable=True),),
            locked=True,
        )

        token_a = _make_token(branch_name="a", token_id="t1", data={"x": 42}, contract=contract_a)
        token_b = _make_token(branch_name="b", token_id="t2", data={"x": None}, contract=contract_b)

        # Accept both tokens
        outcome_a = executor.accept(token_a, "merge")
        assert outcome_a.held is True  # Waiting for branch b

        outcome_b = executor.accept(token_b, "merge")
        assert outcome_b.held is False  # Merge triggered

        # Verify contract matches DAG schema, not runtime merge
        merged_contract = outcome_b.merged_token.row_data.contract
        assert merged_contract == precomputed_schema, "Runtime contract should match pre-computed DAG schema, not runtime merge"

    def test_union_merge_falls_back_to_runtime_merge_when_no_schema(self):
        """Without pre-computed schema, runtime merge() is used (backward compat)."""
        executor, *_ = _make_executor()

        # Register coalesce WITHOUT pre-computed schema
        s = _settings(branches=["a", "b"], policy="require_all", merge="union")
        executor.register_coalesce(s, "node_1")  # No output_schema

        # Create tokens with same contract
        contract = _make_contract(mode="FLEXIBLE")
        token_a = _make_token(branch_name="a", token_id="t1", data={"amount": 100}, contract=contract)
        token_b = _make_token(branch_name="b", token_id="t2", data={"amount": 200}, contract=contract)

        executor.accept(token_a, "merge")
        outcome = executor.accept(token_b, "merge")

        # Should still work (runtime merge fallback)
        assert outcome.held is False
        assert outcome.merged_token is not None
        # Contract should be result of runtime merge (same as input since identical contracts)
        assert outcome.merged_token.row_data.contract.mode == contract.mode


class TestOriginalNamePreservation:
    """Regression tests for preserving original_name in coalesce contracts.

    Bug: P2-RC5-original-name-loss

    The orchestrator rebuilds coalesce contracts from output_schema_config,
    which only has normalized names. Branch contracts carry the actual
    original→normalized mapping from the source. The merged contract must
    preserve original names for sinks using `headers: original`.

    Prior bug: create_contract_from_config() was called without field_resolution,
    so all field contracts got original_name == normalized_name, breaking
    header resolution for downstream sinks.
    """

    def test_union_merge_preserves_branch_original_names(self):
        """Union merge should preserve original_name from branch contracts."""
        executor, *_ = _make_executor()

        # Pre-computed schema with normalized names only (simulates DAG builder)
        precomputed = SchemaContract(
            mode="FLEXIBLE",
            fields=(
                make_field("customer_id", original_name="customer_id", python_type=str, required=True, source="declared"),
                make_field("amount", original_name="amount", python_type=float, required=True, source="declared"),
            ),
            locked=True,
        )

        s = _settings(branches=["a", "b"], policy="require_all", merge="union")
        executor.register_coalesce(s, "node_1", output_schema=precomputed)

        # Branch contracts have DIFFERENT original names (from source headers)
        contract_a = SchemaContract(
            mode="FLEXIBLE",
            fields=(
                make_field("customer_id", original_name="Customer ID", python_type=str, required=True, source="declared"),
                make_field("amount", original_name="Transaction Amount", python_type=float, required=True, source="declared"),
            ),
            locked=True,
        )
        contract_b = SchemaContract(
            mode="FLEXIBLE",
            fields=(
                make_field("customer_id", original_name="Customer ID", python_type=str, required=True, source="declared"),
                make_field("amount", original_name="Transaction Amount", python_type=float, required=True, source="declared"),
            ),
            locked=True,
        )

        token_a = _make_token(branch_name="a", token_id="t1", data={"customer_id": "C1", "amount": 100.0}, contract=contract_a)
        token_b = _make_token(branch_name="b", token_id="t2", data={"customer_id": "C1", "amount": 200.0}, contract=contract_b)

        executor.accept(token_a, "merge")
        outcome = executor.accept(token_b, "merge")

        # Merged contract should preserve original names from branches
        merged = outcome.merged_token.row_data.contract
        customer_id_field = next(f for f in merged.fields if f.normalized_name == "customer_id")
        amount_field = next(f for f in merged.fields if f.normalized_name == "amount")

        assert customer_id_field.original_name == "Customer ID", (
            f"original_name was lost: expected 'Customer ID', got '{customer_id_field.original_name}'"
        )
        assert amount_field.original_name == "Transaction Amount", (
            f"original_name was lost: expected 'Transaction Amount', got '{amount_field.original_name}'"
        )

    def test_union_merge_original_name_follows_policy_not_arrival(self):
        """original_name must match the winning branch per coalesce policy, not arrival order.

        Bug: P2-RC5-original-name-arrival-order

        When two branches normalize different source headers to the same field:
        - Branch A: "Customer ID" -> customer_id
        - Branch B: "customer-ID" -> customer_id (different original)

        With last_wins policy, the merged contract should use B's original_name
        when B's value wins, even if A arrived first. Otherwise, a downstream
        sink using `headers: original` labels B's value with A's header.
        """
        executor, *_ = _make_executor()

        # Pre-computed schema with normalized names only
        precomputed = SchemaContract(
            mode="FLEXIBLE",
            fields=(make_field("customer_id", original_name="customer_id", python_type=str, required=True, source="declared"),),
            locked=True,
        )

        # settings.branches = ["a", "b"] means B's value wins under last_wins
        s = _settings(branches=["a", "b"], policy="require_all", merge="union")
        executor.register_coalesce(s, "node_1", output_schema=precomputed)

        # Branch contracts have DIFFERENT original names for the same normalized field
        contract_a = SchemaContract(
            mode="FLEXIBLE",
            fields=(make_field("customer_id", original_name="Customer ID", python_type=str, required=True, source="declared"),),
            locked=True,
        )
        contract_b = SchemaContract(
            mode="FLEXIBLE",
            fields=(make_field("customer_id", original_name="customer-ID", python_type=str, required=True, source="declared"),),
            locked=True,
        )

        # A arrives FIRST, but B's value should win (last_wins default)
        token_a = _make_token(branch_name="a", token_id="t1", data={"customer_id": "A_VALUE"}, contract=contract_a)
        token_b = _make_token(branch_name="b", token_id="t2", data={"customer_id": "B_VALUE"}, contract=contract_b)

        executor.accept(token_a, "merge")
        outcome = executor.accept(token_b, "merge")

        # Value should be B's (last_wins)
        assert outcome.merged_token.row_data["customer_id"] == "B_VALUE"

        # original_name must be B's header, not A's (the bug: A arrives first -> A's header)
        merged = outcome.merged_token.row_data.contract
        customer_id_field = next(f for f in merged.fields if f.normalized_name == "customer_id")

        assert customer_id_field.original_name == "customer-ID", (
            f"original_name should match winning branch B's header 'customer-ID', "
            f"but got '{customer_id_field.original_name}' (likely from first-arrived branch A)"
        )


class TestObservedUnionCoalesce:
    """Regression tests for OBSERVED union coalesce handling.

    Bug: P1-RC5-observed-union-contracts

    For OBSERVED schemas, the precomputed contract is empty (fields=()) since
    types are inferred at runtime. The executor should skip precomputed and
    merge branch contracts directly.

    Prior bug: The guard required output_schema for ALL union merges, but
    OBSERVED unions don't need precomputed — they should use branch contracts.
    """

    def test_observed_union_merges_branch_contracts_directly(self):
        """OBSERVED union should merge branch contracts, not use empty precomputed."""
        executor, *_ = _make_executor()

        # Pre-computed OBSERVED schema (empty fields, as DAG builder would produce)
        precomputed = SchemaContract(
            mode="OBSERVED",
            fields=(),
            locked=False,
        )

        s = _settings(branches=["a", "b"], policy="require_all", merge="union")
        executor.register_coalesce(s, "node_1", output_schema=precomputed)

        # Branch contracts have fields (observed from data)
        contract_a = SchemaContract(
            mode="OBSERVED",
            fields=(make_field("x", original_name="x", python_type=int, required=True, source="inferred"),),
            locked=True,
        )
        contract_b = SchemaContract(
            mode="OBSERVED",
            fields=(make_field("x", original_name="x", python_type=int, required=True, source="inferred"),),
            locked=True,
        )

        token_a = _make_token(branch_name="a", token_id="t1", data={"x": 1}, contract=contract_a)
        token_b = _make_token(branch_name="b", token_id="t2", data={"x": 2}, contract=contract_b)

        executor.accept(token_a, "merge")
        outcome = executor.accept(token_b, "merge")

        # Merged contract should have fields from branches, not empty precomputed
        merged = outcome.merged_token.row_data.contract
        assert len(merged.fields) > 0, "OBSERVED union dropped all fields (using empty precomputed)"
        assert any(f.normalized_name == "x" for f in merged.fields), "Field 'x' missing from merged contract"

    def test_observed_union_without_precomputed_falls_back_gracefully(self):
        """OBSERVED union without precomputed should fall back to runtime merge."""
        executor, *_ = _make_executor()

        # Register WITHOUT pre-computed schema (all branches OBSERVED)
        s = _settings(branches=["a", "b"], policy="require_all", merge="union")
        executor.register_coalesce(s, "node_1")  # No output_schema

        # Both branch contracts are OBSERVED
        contract = SchemaContract(
            mode="OBSERVED",
            fields=(make_field("y", original_name="y", python_type=str, required=True, source="inferred"),),
            locked=True,
        )

        token_a = _make_token(branch_name="a", token_id="t1", data={"y": "hello"}, contract=contract)
        token_b = _make_token(branch_name="b", token_id="t2", data={"y": "world"}, contract=contract)

        executor.accept(token_a, "merge")
        outcome = executor.accept(token_b, "merge")

        # Should work (OBSERVED branches don't require precomputed)
        assert outcome.held is False
        assert outcome.merged_token is not None
        assert any(f.normalized_name == "y" for f in outcome.merged_token.row_data.contract.fields)
