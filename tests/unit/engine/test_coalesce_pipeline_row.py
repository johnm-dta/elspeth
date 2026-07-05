# tests/unit/engine/test_coalesce_pipeline_row.py
"""Tests for CoalesceExecutor with PipelineRow support (Task 6)."""

from types import SimpleNamespace
from typing import Any

import pytest

from elspeth.contracts import TokenInfo
from elspeth.contracts.errors import OrchestrationInvariantError
from elspeth.contracts.schema_contract import PipelineRow, SchemaContract
from elspeth.contracts.types import NodeID
from elspeth.core.config import CoalesceSettings
from elspeth.testing import make_field, make_row
from tests.unit.engine.conftest import MockCoalesceExecutor


class _CallRecord:
    def __init__(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        self.args = args
        self.kwargs = kwargs


class _CallRecorder:
    def __init__(self, return_value: Any = None, side_effect: Any = None) -> None:
        self.return_value = return_value
        self.side_effect = side_effect
        self.call_args: _CallRecord | None = None
        self.call_args_list: list[_CallRecord] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        record = _CallRecord(args, kwargs)
        self.call_args = record
        self.call_args_list.append(record)
        if self.side_effect is not None:
            if isinstance(self.side_effect, BaseException):
                raise self.side_effect
            return self.side_effect(*args, **kwargs)
        return self.return_value

    @property
    def call_count(self) -> int:
        return len(self.call_args_list)

    def assert_called_once(self) -> None:
        assert self.call_count == 1


class _RecorderDouble:
    def __init__(self) -> None:
        self.create_row = _CallRecorder(SimpleNamespace(row_id="row_001"))
        self.create_token = _CallRecorder(SimpleNamespace(token_id="token_001"))
        self.coalesce_tokens = _CallRecorder(SimpleNamespace(token_id="merged_001", join_group_id="join_001"))
        self.has_completed_row_for_node = _CallRecorder(False)
        self.get_completed_row_ids_for_nodes = _CallRecorder([])
        self.begin_node_state = _CallRecorder(SimpleNamespace(state_id="state_001"))
        self.complete_node_state = _CallRecorder()


class _DataFlowDouble:
    def __init__(self) -> None:
        self.record_token_outcome = _CallRecorder()


class _SpanFactorySentinel:
    pass


class _TokenManagerDouble:
    def __init__(self) -> None:
        self.coalesce_tokens = _CallRecorder(side_effect=_coalesce_tokens_impl)


class _BadPipelineRow:
    contract = None

    def to_dict(self) -> dict[str, int]:
        return {"amount": 100}


def _coalesce_tokens_impl(parents: list[TokenInfo], merged_data: PipelineRow, node_id: NodeID, run_id: str) -> TokenInfo:
    return TokenInfo(
        row_id=parents[0].row_id,
        token_id="merged_001",
        row_data=merged_data,
        join_group_id="join_001",
    )


def _make_contract(fields: list[Any] | None = None) -> SchemaContract:
    """Create a schema contract for testing."""
    if fields is None:
        fields = [
            make_field(
                "amount",
                original_name="'Amount'",
                python_type=int,
                required=True,
                source="declared",
            ),
        ]
    return SchemaContract(fields=tuple(fields), mode="OBSERVED", locked=True)


def _make_recorder() -> _RecorderDouble:
    """Create an ExecutionRepository double."""
    return _RecorderDouble()


def _make_span_factory() -> _SpanFactorySentinel:
    """Create a SpanFactory placeholder."""
    return _SpanFactorySentinel()


def _make_token_manager() -> _TokenManagerDouble:
    """Create a TokenManager double."""
    return _TokenManagerDouble()


class TestCoalesceExecutorPipelineRow:
    """Tests for CoalesceExecutor with PipelineRow and contract merging."""

    def test_coalesce_merges_contracts(self) -> None:
        """Coalesce should merge contracts from all branches."""
        # Create contracts for each branch
        contract_a = _make_contract(
            fields=[
                make_field(
                    "amount",
                    original_name="'Amount'",
                    python_type=int,
                    required=True,
                    source="declared",
                ),
                make_field(
                    "branch_a_field",
                    original_name="branch_a_field",
                    python_type=str,
                    required=False,
                    source="inferred",
                ),
            ]
        )
        contract_b = _make_contract(
            fields=[
                make_field(
                    "amount",
                    original_name="'Amount'",
                    python_type=int,
                    required=True,
                    source="declared",
                ),
                make_field(
                    "branch_b_field",
                    original_name="branch_b_field",
                    python_type=str,
                    required=False,
                    source="inferred",
                ),
            ]
        )

        execution = _make_recorder()
        data_flow = _DataFlowDouble()
        span_factory = _make_span_factory()
        token_manager = _make_token_manager()

        executor = MockCoalesceExecutor(
            execution=execution,
            span_factory=span_factory,
            token_manager=token_manager,
            run_id="run_001",
            step_resolver=lambda node_id: 3,
            data_flow=data_flow,
        )

        # Register coalesce point
        settings = CoalesceSettings(
            name="merge_point",
            branches=["branch_a", "branch_b"],
            policy="require_all",
            merge="union",
        )
        executor.register_coalesce(settings, NodeID("node_coalesce_001"))

        # Create tokens with PipelineRow for each branch
        token_a = TokenInfo(
            row_id="row_001",
            token_id="token_a",
            row_data=make_row({"amount": 100, "branch_a_field": "a"}, contract=contract_a),
            branch_name="branch_a",
            fork_group_id="fork_001",
        )
        token_b = TokenInfo(
            row_id="row_001",
            token_id="token_b",
            row_data=make_row({"amount": 100, "branch_b_field": "b"}, contract=contract_b),
            branch_name="branch_b",
            fork_group_id="fork_001",
        )

        # Accept both tokens
        outcome_a = executor.accept(token_a, "merge_point")
        assert outcome_a.held is True  # Waiting for branch_b

        outcome_b = executor.accept(token_b, "merge_point")
        assert outcome_b.held is False  # Merge triggered

        # Should have called coalesce_tokens with PipelineRow containing merged contract
        token_manager.coalesce_tokens.assert_called_once()
        call_kwargs = token_manager.coalesce_tokens.call_args.kwargs
        merged_data = call_kwargs["merged_data"]

        # Verify merged data is PipelineRow with merged contract
        assert isinstance(merged_data, PipelineRow)
        merged_contract = merged_data.contract

        # Merged contract should have fields from both branches
        assert merged_contract.get_field("amount") is not None
        assert merged_contract.get_field("branch_a_field") is not None
        assert merged_contract.get_field("branch_b_field") is not None

    def test_coalesce_crashes_if_contract_none(self) -> None:
        """Coalesce should crash if any token has None contract.

        Per CLAUDE.md: "Bad data in the audit trail = crash immediately"
        A token with None contract is a bug in upstream code.
        """
        contract = _make_contract()
        execution = _make_recorder()
        data_flow = _DataFlowDouble()
        span_factory = _make_span_factory()
        token_manager = _make_token_manager()

        executor = MockCoalesceExecutor(
            execution=execution,
            span_factory=span_factory,
            token_manager=token_manager,
            run_id="run_001",
            step_resolver=lambda node_id: 3,
            data_flow=data_flow,
        )

        settings = CoalesceSettings(
            name="merge_point",
            branches=["branch_a", "branch_b"],
            policy="require_all",
            merge="union",
        )
        executor.register_coalesce(settings, NodeID("node_coalesce_001"))

        # Token A has contract, Token B has None contract (bug scenario)
        token_a = TokenInfo(
            row_id="row_001",
            token_id="token_a",
            row_data=make_row({"amount": 100}, contract=contract),
            branch_name="branch_a",
            fork_group_id="fork_001",
        )

        token_b = TokenInfo(
            row_id="row_001",
            token_id="token_b",
            row_data=_BadPipelineRow(),
            branch_name="branch_b",
            fork_group_id="fork_001",
        )

        # Accept first token
        executor.accept(token_a, "merge_point")

        # Accept second token should crash
        with pytest.raises(OrchestrationInvariantError, match="has no contract"):
            executor.accept(token_b, "merge_point")

    def test_coalesce_merge_failure_returns_graceful_failure(self) -> None:
        """Contract merge failure should return graceful failure outcome.

        When contracts have conflicting types for the same field,
        merge() raises ContractMergeError. For observed schemas, this
        can't be caught at build time, so we fail gracefully to preserve
        audit trail integrity. (See: elspeth-c75ac86e35)
        """
        # Create contracts with conflicting types for same field
        contract_a = _make_contract(
            fields=[
                make_field(
                    "value",
                    original_name="value",
                    python_type=int,  # int
                    required=True,
                    source="declared",
                ),
            ]
        )
        contract_b = _make_contract(
            fields=[
                make_field(
                    "value",
                    original_name="value",
                    python_type=str,  # str - conflicts with int!
                    required=True,
                    source="declared",
                ),
            ]
        )

        execution = _make_recorder()
        data_flow = _DataFlowDouble()
        span_factory = _make_span_factory()
        token_manager = _make_token_manager()

        executor = MockCoalesceExecutor(
            execution=execution,
            span_factory=span_factory,
            token_manager=token_manager,
            run_id="run_001",
            step_resolver=lambda node_id: 3,
            data_flow=data_flow,
        )

        settings = CoalesceSettings(
            name="merge_point",
            branches=["branch_a", "branch_b"],
            policy="require_all",
            merge="union",
        )
        executor.register_coalesce(settings, NodeID("node_coalesce_001"))

        token_a = TokenInfo(
            row_id="row_001",
            token_id="token_a",
            row_data=make_row({"value": 100}, contract=contract_a),
            branch_name="branch_a",
            fork_group_id="fork_001",
        )
        token_b = TokenInfo(
            row_id="row_001",
            token_id="token_b",
            row_data=make_row({"value": "text"}, contract=contract_b),
            branch_name="branch_b",
            fork_group_id="fork_001",
        )

        executor.accept(token_a, "merge_point")

        # Second accept triggers merge, which fails due to type conflict
        outcome = executor.accept(token_b, "merge_point")

        # Outcome indicates failure, not held or merged
        assert outcome.failure_reason is not None
        assert "contract_type_conflict" in outcome.failure_reason
        assert outcome.held is False
        assert outcome.merged_token is None
        assert outcome.outcomes_recorded is True  # Tokens properly terminated

    def test_first_policy_merges_immediately(self) -> None:
        """Coalesce with "first" policy should merge on first arrival.

        "first" policy is used when any branch completing is sufficient.
        CoalesceSettings requires at least 2 branches, but "first" policy
        allows merge as soon as any single branch arrives.
        """
        contract = _make_contract()
        execution = _make_recorder()
        data_flow = _DataFlowDouble()
        span_factory = _make_span_factory()
        token_manager = _make_token_manager()

        executor = MockCoalesceExecutor(
            execution=execution,
            span_factory=span_factory,
            token_manager=token_manager,
            run_id="run_001",
            step_resolver=lambda node_id: 3,
            data_flow=data_flow,
        )

        # Two branches with "first" policy - merge on first arrival
        settings = CoalesceSettings(
            name="merge_point",
            branches=["branch_a", "branch_b"],
            policy="first",
            merge="union",
        )
        executor.register_coalesce(settings, NodeID("node_coalesce_001"))

        token_a = TokenInfo(
            row_id="row_001",
            token_id="token_a",
            row_data=make_row({"amount": 100}, contract=contract),
            branch_name="branch_a",
            fork_group_id="fork_001",
        )

        # Accept first token - should merge immediately with "first" policy
        outcome = executor.accept(token_a, "merge_point")

        assert outcome.held is False
        assert outcome.merged_token is not None

        # Merged token should have PipelineRow with same contract
        assert isinstance(outcome.merged_token.row_data, PipelineRow)
        assert outcome.merged_token.row_data.contract is contract

    def test_coalesce_preserves_row_data_correctly(self) -> None:
        """Coalesce should preserve row data according to merge strategy."""
        contract = _make_contract()
        execution = _make_recorder()
        data_flow = _DataFlowDouble()
        span_factory = _make_span_factory()
        token_manager = _make_token_manager()

        executor = MockCoalesceExecutor(
            execution=execution,
            span_factory=span_factory,
            token_manager=token_manager,
            run_id="run_001",
            step_resolver=lambda node_id: 3,
            data_flow=data_flow,
        )

        settings = CoalesceSettings(
            name="merge_point",
            branches=["branch_a", "branch_b"],
            policy="require_all",
            merge="union",  # Union merge combines all fields
        )
        executor.register_coalesce(settings, NodeID("node_coalesce_001"))

        token_a = TokenInfo(
            row_id="row_001",
            token_id="token_a",
            row_data=make_row({"amount": 100, "a_only": "a"}, contract=contract),
            branch_name="branch_a",
            fork_group_id="fork_001",
        )
        token_b = TokenInfo(
            row_id="row_001",
            token_id="token_b",
            row_data=make_row({"amount": 200, "b_only": "b"}, contract=contract),
            branch_name="branch_b",
            fork_group_id="fork_001",
        )

        executor.accept(token_a, "merge_point")
        executor.accept(token_b, "merge_point")

        # Union merge: later branches override, all fields present
        call_kwargs = token_manager.coalesce_tokens.call_args.kwargs
        merged_data = call_kwargs["merged_data"]

        # Use to_dict() to access all fields
        data_dict = merged_data.to_dict()
        assert data_dict["a_only"] == "a"  # From branch_a
        assert data_dict["b_only"] == "b"  # From branch_b
        assert data_dict["amount"] == 200  # Overridden by branch_b (later in list)
