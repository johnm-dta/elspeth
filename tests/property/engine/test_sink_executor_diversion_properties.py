"""Hypothesis property tests for SinkExecutor failsink routing.

These verify invariants that hold across ALL possible batch sizes and
diversion patterns --- not just the hand-crafted fixtures in unit tests.

NOTE: These tests verify the single-run invariant only. On resume,
diverted tokens may produce duplicate outcomes (see resume caveat
in the spec). That is a known P1 follow-up, not a property violation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from elspeth.contracts import Artifact, PendingOutcome, PipelineRow, PluginSchema, SchemaContract, TerminalOutcome, TerminalPath, TokenInfo
from elspeth.contracts.diversion import RowDiversion, SinkWriteResult
from elspeth.contracts.results import ArtifactDescriptor
from elspeth.engine.executors.sink import SinkExecutor


class _PermissiveSchema(PluginSchema):
    """Accept arbitrary sink rows for executor plumbing properties."""


@dataclass(frozen=True)
class _RecordedCall:
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


class _CallRecorder:
    def __init__(self, *, return_value: Any = None, side_effect: Any = None) -> None:
        self.return_value = return_value
        self.side_effect = side_effect
        self.call_args_list: list[_RecordedCall] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.call_args_list.append(_RecordedCall(args=args, kwargs=dict(kwargs)))
        if self.side_effect is None:
            return self.return_value
        effect = self.side_effect
        if isinstance(effect, list):
            effect = effect.pop(0)
        if isinstance(effect, BaseException):
            raise effect
        if isinstance(effect, type) and issubclass(effect, BaseException):
            raise effect()
        if callable(effect):
            return effect(*args, **kwargs)
        return effect


class _SpanContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return False


class _SpanFactoryDouble:
    def __init__(self) -> None:
        self.sink_span = _CallRecorder(return_value=_SpanContext())


class _SinkDouble:
    def __init__(
        self,
        *,
        name: str,
        node_id: str,
        artifact: ArtifactDescriptor,
        diversions: tuple[RowDiversion, ...] = (),
        on_write_failure: str = "discard",
    ) -> None:
        self.name = name
        self.node_id = node_id
        self.input_schema = _PermissiveSchema
        self.declared_guaranteed_fields = frozenset()
        self.declared_required_fields = frozenset()
        self._on_write_failure = on_write_failure
        self._reset_diversion_log = _CallRecorder()
        self.write = _CallRecorder(return_value=SinkWriteResult(artifact=artifact, diversions=diversions))
        self.flush = _CallRecorder()


_TEST_CONTRACT = SchemaContract(mode="OBSERVED", fields=(), locked=True)


def _make_token(token_id: str, row_data: dict[str, object] | None = None) -> TokenInfo:
    pipeline_row = PipelineRow(row_data or {"field": "value"}, _TEST_CONTRACT)
    return TokenInfo(row_id=f"row-{token_id}", token_id=token_id, row_data=pipeline_row)


def _make_context() -> SimpleNamespace:
    def _for_contract(contract: SchemaContract | None) -> SimpleNamespace:
        scoped = SimpleNamespace(run_id="run-1", operation_id=None, contract=contract)
        scoped.for_contract = _for_contract
        return scoped

    ctx = SimpleNamespace(run_id="run-1", operation_id=None)
    ctx.for_contract = _for_contract
    return ctx


def _make_executor() -> tuple[SinkExecutor, SimpleNamespace, SimpleNamespace]:
    state_counter = [0]
    artifact_counter = [0]

    def _begin_state(**kwargs: Any) -> SimpleNamespace:
        del kwargs
        state_counter[0] += 1
        return SimpleNamespace(state_id=f"state-{state_counter[0]}")

    def _register_artifact(**kwargs: Any) -> Artifact:
        artifact_counter[0] += 1
        return Artifact(
            artifact_id=f"artifact-{artifact_counter[0]}",
            run_id=kwargs["run_id"],
            produced_by_state_id=kwargs["state_id"],
            sink_node_id=kwargs["sink_node_id"],
            artifact_type=kwargs["artifact_type"],
            path_or_uri=kwargs["path"],
            content_hash=kwargs["content_hash"],
            size_bytes=kwargs["size_bytes"],
            created_at=datetime.now(UTC),
            idempotency_key=kwargs.get("idempotency_key"),
            publication_evidence_kind="legacy_returned",
        )

    execution = SimpleNamespace(
        begin_node_state=_CallRecorder(side_effect=_begin_state),
        complete_node_state=_CallRecorder(),
        register_artifact=_CallRecorder(side_effect=_register_artifact),
        record_routing_event=_CallRecorder(),
        begin_operation=_CallRecorder(return_value=SimpleNamespace(operation_id="op-1")),
        complete_operation=_CallRecorder(),
    )
    data_flow = SimpleNamespace(record_token_outcome=_CallRecorder())
    spans = _SpanFactoryDouble()
    return SinkExecutor(execution, data_flow, spans, run_id="run-1"), execution, data_flow


def _build_scenario(batch_size: int, diverted_indices: set[int]) -> tuple[list[TokenInfo], _SinkDouble]:
    """Build tokens and a sink mock for a given batch/diversion scenario."""
    tokens = [_make_token(f"t{i}") for i in range(batch_size)]
    diversions = tuple(RowDiversion(row_index=i, reason=f"reason-{i}", row_data={"i": i}) for i in sorted(diverted_indices))
    artifact = ArtifactDescriptor.for_file(path="/tmp/p", content_hash="a" * 64, size_bytes=0)
    sink = _SinkDouble(name="primary", node_id="node-primary", artifact=artifact, diversions=diversions)
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
        tokens=tokens,
        ctx=_make_context(),
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
        tokens=tokens,
        ctx=_make_context(),
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


def _build_failsink_scenario(batch_size: int, diverted_indices: set[int]) -> tuple[list[TokenInfo], _SinkDouble, _SinkDouble]:
    """Build tokens, primary sink, and failsink for a failsink-mode scenario."""
    tokens = [_make_token(f"t{i}") for i in range(batch_size)]
    diversions = tuple(RowDiversion(row_index=i, reason=f"reason-{i}", row_data={"i": i}) for i in sorted(diverted_indices))
    artifact = ArtifactDescriptor.for_file(path="/tmp/p", content_hash="a" * 64, size_bytes=0)
    failsink_artifact = ArtifactDescriptor.for_file(path="/tmp/f", content_hash="b" * 64, size_bytes=0)
    sink = _SinkDouble(
        name="primary",
        node_id="node-primary",
        artifact=artifact,
        diversions=diversions,
        on_write_failure="csv_failsink",
    )
    failsink = _SinkDouble(name="csv_failsink", node_id="node-failsink", artifact=failsink_artifact)
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
        tokens=tokens,
        ctx=_make_context(),
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
        tokens=tokens,
        ctx=_make_context(),
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
