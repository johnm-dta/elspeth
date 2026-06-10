"""Hypothesis property tests for SinkExecutor failsink routing.

These verify invariants that hold across ALL possible batch sizes and
diversion patterns --- not just the hand-crafted fixtures in unit tests.

NOTE: These tests verify the single-run invariant only. On resume,
diverted tokens may produce duplicate outcomes (see resume caveat
in the spec). That is a known P1 follow-up, not a property violation.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from hypothesis import given, settings
from hypothesis import strategies as st

from elspeth.contracts import PendingOutcome, TerminalOutcome, TerminalPath, TokenInfo
from elspeth.contracts.diversion import RowDiversion, SinkWriteResult
from elspeth.contracts.results import ArtifactDescriptor
from elspeth.engine.executors.sink import SinkExecutor


def _make_token(token_id: str, row_data: dict[str, object] | None = None) -> MagicMock:
    token = MagicMock(spec=TokenInfo)
    token.token_id = token_id
    token.row_id = f"row-{token_id}"
    mock_row = MagicMock()
    mock_row.to_dict.return_value = row_data or {"field": "value"}
    mock_row.contract = MagicMock()
    mock_row.contract.merge_for_batch.return_value = mock_row.contract
    token.row_data = mock_row
    return token


def _make_executor() -> tuple[SinkExecutor, MagicMock, MagicMock]:
    execution = MagicMock()
    data_flow = MagicMock()
    state_counter = [0]

    def _begin_state(**kwargs: Any) -> MagicMock:
        state_counter[0] += 1
        state = MagicMock()
        state.state_id = f"state-{state_counter[0]}"
        return state

    execution.begin_node_state.side_effect = _begin_state
    spans = MagicMock()
    spans.sink_span.return_value.__enter__ = MagicMock(return_value=None)
    spans.sink_span.return_value.__exit__ = MagicMock(return_value=False)
    return SinkExecutor(execution, data_flow, spans, run_id="run-1"), execution, data_flow


def _build_scenario(batch_size: int, diverted_indices: set[int]) -> tuple[list[MagicMock], MagicMock]:
    """Build tokens and a sink mock for a given batch/diversion scenario."""
    tokens = [_make_token(f"t{i}") for i in range(batch_size)]
    diversions = tuple(RowDiversion(row_index=i, reason=f"reason-{i}", row_data={"i": i}) for i in sorted(diverted_indices))
    artifact = ArtifactDescriptor.for_file(path="/tmp/p", content_hash="a" * 64, size_bytes=0)
    sink = MagicMock()
    sink.name = "primary"
    sink.node_id = "node-primary"
    sink.declared_guaranteed_fields = frozenset()
    sink.declared_required_fields = frozenset()
    sink._on_write_failure = "discard"
    sink._reset_diversion_log = MagicMock()
    sink.write.return_value = SinkWriteResult(artifact=artifact, diversions=diversions)
    return tokens, sink


@given(
    batch_size=st.integers(min_value=1, max_value=30),
    diverted_indices_raw=st.lists(st.integers(min_value=0, max_value=29), max_size=30),
)
@settings(max_examples=200)
def test_partition_completeness(batch_size: int, diverted_indices_raw: list[int]) -> None:
    """Every token gets exactly one outcome: COMPLETED + DIVERTED == total batch."""
    diverted_indices = {i for i in diverted_indices_raw if i < batch_size}
    tokens, sink = _build_scenario(batch_size, diverted_indices)
    executor, _execution, data_flow = _make_executor()

    executor.write(
        sink=sink,
        tokens=tokens,  # type: ignore[arg-type]
        ctx=MagicMock(run_id="run-1"),
        step_in_pipeline=5,
        sink_name="primary",
        pending_outcome=PendingOutcome(outcome=TerminalOutcome.SUCCESS, path=TerminalPath.DEFAULT_FLOW),
    )

    outcome_calls = data_flow.record_token_outcome.call_args_list
    completed_ids = {
        c.kwargs["ref"].token_id
        for c in outcome_calls
        if (c.kwargs["outcome"], c.kwargs["path"]) == (TerminalOutcome.SUCCESS, TerminalPath.DEFAULT_FLOW)
    }
    diverted_ids = {
        c.kwargs["ref"].token_id
        for c in outcome_calls
        if (c.kwargs["outcome"], c.kwargs["path"]) == (TerminalOutcome.FAILURE, TerminalPath.SINK_DISCARDED)
    }

    # Partition completeness: every token accounted for
    assert len(completed_ids) + len(diverted_ids) == batch_size
    # Disjoint: no token in both sets
    assert completed_ids & diverted_ids == set()
    # All tokens present
    all_token_ids = {t.token_id for t in tokens}
    assert completed_ids | diverted_ids == all_token_ids


@given(
    batch_size=st.integers(min_value=1, max_value=30),
    diverted_indices_raw=st.lists(st.integers(min_value=0, max_value=29), max_size=30),
)
@settings(max_examples=200)
def test_exactly_once_terminal_state(batch_size: int, diverted_indices_raw: list[int]) -> None:
    """Each token_id appears in exactly one record_token_outcome call."""
    diverted_indices = {i for i in diverted_indices_raw if i < batch_size}
    tokens, sink = _build_scenario(batch_size, diverted_indices)
    executor, _execution, data_flow = _make_executor()

    executor.write(
        sink=sink,
        tokens=tokens,  # type: ignore[arg-type]
        ctx=MagicMock(run_id="run-1"),
        step_in_pipeline=5,
        sink_name="primary",
        pending_outcome=PendingOutcome(outcome=TerminalOutcome.SUCCESS, path=TerminalPath.DEFAULT_FLOW),
    )

    outcome_calls = data_flow.record_token_outcome.call_args_list
    recorded_token_ids = [c.kwargs["ref"].token_id for c in outcome_calls]
    # No duplicates
    assert len(recorded_token_ids) == len(set(recorded_token_ids))
    # All input tokens present
    assert set(recorded_token_ids) == {t.token_id for t in tokens}


# =============================================================================
# Failsink mode property tests
# =============================================================================


def _build_failsink_scenario(batch_size: int, diverted_indices: set[int]) -> tuple[list[MagicMock], MagicMock, MagicMock]:
    """Build tokens, primary sink, and failsink for a failsink-mode scenario."""
    tokens = [_make_token(f"t{i}") for i in range(batch_size)]
    diversions = tuple(RowDiversion(row_index=i, reason=f"reason-{i}", row_data={"i": i}) for i in sorted(diverted_indices))
    artifact = ArtifactDescriptor.for_file(path="/tmp/p", content_hash="a" * 64, size_bytes=0)
    failsink_artifact = ArtifactDescriptor.for_file(path="/tmp/f", content_hash="b" * 64, size_bytes=0)
    sink = MagicMock()
    sink.name = "primary"
    sink.node_id = "node-primary"
    sink.declared_guaranteed_fields = frozenset()
    sink.declared_required_fields = frozenset()
    sink._on_write_failure = "csv_failsink"
    sink._reset_diversion_log = MagicMock()
    sink.write.return_value = SinkWriteResult(artifact=artifact, diversions=diversions)
    failsink = MagicMock()
    failsink.name = "csv_failsink"
    failsink.node_id = "node-failsink"
    failsink.declared_guaranteed_fields = frozenset()
    failsink.declared_required_fields = frozenset()
    failsink.write.return_value = SinkWriteResult(artifact=failsink_artifact)
    failsink._reset_diversion_log = MagicMock()
    return tokens, sink, failsink


@given(
    batch_size=st.integers(min_value=1, max_value=30),
    diverted_indices_raw=st.lists(st.integers(min_value=0, max_value=29), max_size=30),
)
@settings(max_examples=200)
def test_failsink_partition_completeness(batch_size: int, diverted_indices_raw: list[int]) -> None:
    """Failsink mode: every token gets exactly one outcome (COMPLETED or DIVERTED)."""
    diverted_indices = {i for i in diverted_indices_raw if i < batch_size}
    tokens, sink, failsink = _build_failsink_scenario(batch_size, diverted_indices)
    executor, _execution, data_flow = _make_executor()

    executor.write(
        sink=sink,
        tokens=tokens,  # type: ignore[arg-type]
        ctx=MagicMock(run_id="run-1"),
        step_in_pipeline=5,
        sink_name="primary",
        pending_outcome=PendingOutcome(outcome=TerminalOutcome.SUCCESS, path=TerminalPath.DEFAULT_FLOW),
        failsink=failsink,
        failsink_name="csv_failsink",
        failsink_edge_id="edge-failsink-1",
    )

    outcome_calls = data_flow.record_token_outcome.call_args_list
    completed_ids = {
        c.kwargs["ref"].token_id
        for c in outcome_calls
        if (c.kwargs["outcome"], c.kwargs["path"]) == (TerminalOutcome.SUCCESS, TerminalPath.DEFAULT_FLOW)
    }
    diverted_ids = {
        c.kwargs["ref"].token_id
        for c in outcome_calls
        if (c.kwargs["outcome"], c.kwargs["path"]) == (TerminalOutcome.TRANSIENT, TerminalPath.SINK_FALLBACK_TO_FAILSINK)
    }

    assert len(completed_ids) + len(diverted_ids) == batch_size
    assert completed_ids & diverted_ids == set()
    assert completed_ids | diverted_ids == {t.token_id for t in tokens}


@given(
    batch_size=st.integers(min_value=1, max_value=30),
    diverted_indices_raw=st.lists(st.integers(min_value=0, max_value=29), max_size=30),
)
@settings(max_examples=200)
def test_failsink_exactly_once_terminal_state(batch_size: int, diverted_indices_raw: list[int]) -> None:
    """Failsink mode: each token_id appears in exactly one record_token_outcome call."""
    diverted_indices = {i for i in diverted_indices_raw if i < batch_size}
    tokens, sink, failsink = _build_failsink_scenario(batch_size, diverted_indices)
    executor, _execution, data_flow = _make_executor()

    executor.write(
        sink=sink,
        tokens=tokens,  # type: ignore[arg-type]
        ctx=MagicMock(run_id="run-1"),
        step_in_pipeline=5,
        sink_name="primary",
        pending_outcome=PendingOutcome(outcome=TerminalOutcome.SUCCESS, path=TerminalPath.DEFAULT_FLOW),
        failsink=failsink,
        failsink_name="csv_failsink",
        failsink_edge_id="edge-failsink-1",
    )

    outcome_calls = data_flow.record_token_outcome.call_args_list
    recorded_token_ids = [c.kwargs["ref"].token_id for c in outcome_calls]
    assert len(recorded_token_ids) == len(set(recorded_token_ids))
    assert set(recorded_token_ids) == {t.token_id for t in tokens}
