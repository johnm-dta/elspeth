"""Tests for SinkExecutor failsink routing.

Tests the critical path: after sink.write() returns a SinkWriteResult with
diversions, the executor must record correct per-token outcomes and write
diverted rows to the failsink (or record discard).
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from elspeth.contracts import PendingOutcome, PluginSchema, TokenInfo
from elspeth.contracts.audit import Artifact
from elspeth.contracts.declaration_contracts import _attach_contract_name_from_dispatcher
from elspeth.contracts.diversion import RowDiversion, SinkWriteResult
from elspeth.contracts.enums import NodeStateStatus, RoutingMode, TerminalOutcome, TerminalPath
from elspeth.contracts.errors import (
    AuditIntegrityError,
    FrameworkBugError,
    PluginContractViolation,
    SinkRequiredFieldsViolation,
)
from elspeth.contracts.results import ArtifactDescriptor
from elspeth.engine.executors.sink import SinkExecutor


class _PermissiveSchema(PluginSchema):
    """Accept arbitrary sink rows for executor plumbing tests."""


class _RecordedCall:
    def __init__(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        self.args = args
        self.kwargs = kwargs

    def __getitem__(self, index: int) -> Any:
        if index == 0:
            return self.args
        if index == 1:
            return self.kwargs
        raise IndexError(index)


class _CallRecorder:
    def __init__(self, *, return_value: Any = None, side_effect: Any = None) -> None:
        self.return_value = return_value
        self.side_effect = side_effect
        self.call_args_list: list[_RecordedCall] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.call_args_list.append(_RecordedCall(args, dict(kwargs)))
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

    @property
    def call_count(self) -> int:
        return len(self.call_args_list)

    @property
    def call_args(self) -> _RecordedCall:
        if not self.call_args_list:
            raise AssertionError("Recorder was not called")
        return self.call_args_list[-1]

    def assert_called_once(self) -> None:
        assert self.call_count == 1

    def assert_not_called(self) -> None:
        assert self.call_count == 0


class _ContractDouble:
    def merge_for_batch(self, other: object) -> _ContractDouble:
        del other
        return self


class _RowDouble:
    def __init__(self, data: dict[str, object]) -> None:
        self._data = data
        self.contract = _ContractDouble()

    def to_dict(self) -> dict[str, object]:
        return dict(self._data)


class _TokenDouble:
    def __init__(self, token_id: str, row_data: dict[str, object] | None = None) -> None:
        self.token_id = token_id
        self.row_id = f"row-{token_id}"
        self.row_data = _RowDouble(row_data or {"field": "value"})
        self.resume_attempt_offset = 0
        self.resume_checkpoint_id = None


class _SinkDouble:
    def __init__(
        self,
        *,
        name: str,
        node_id: str,
        diversions: tuple[RowDiversion, ...] = (),
        on_write_failure: str = "discard",
        artifact_path: str = "/tmp/test",
    ) -> None:
        self.name = name
        self.node_id = node_id
        self.input_schema = _PermissiveSchema
        self.declared_guaranteed_fields = frozenset()
        self.declared_required_fields = frozenset()
        self.write = _CallRecorder(
            return_value=SinkWriteResult(
                artifact=_make_artifact(artifact_path),
                diversions=diversions,
            )
        )
        self.flush = _CallRecorder()
        self._on_write_failure = on_write_failure
        self._reset_diversion_log = _CallRecorder()


class _SpanContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return False


class _SpanFactoryDouble:
    def __init__(self) -> None:
        self.sink_span = _CallRecorder(return_value=_SpanContext())


def _make_context(run_id: str = "run-1") -> SimpleNamespace:
    return SimpleNamespace(run_id=run_id, operation_id=None)


def _make_token(token_id: str = "tok-1", row_data: dict[str, object] | None = None) -> TokenInfo:
    """Create a minimal token double."""
    return _TokenDouble(token_id, row_data)  # type: ignore[return-value]


def _make_artifact(path: str = "/tmp/test") -> ArtifactDescriptor:
    return ArtifactDescriptor.for_file(path=path, content_hash="a" * 64, size_bytes=100)


def _default_pending() -> PendingOutcome:
    return PendingOutcome(outcome=TerminalOutcome.SUCCESS, path=TerminalPath.DEFAULT_FLOW)


def _make_sink(
    name: str = "primary",
    node_id: str = "node-primary",
    diversions: tuple[RowDiversion, ...] = (),
    on_write_failure: str = "discard",
) -> _SinkDouble:
    return _SinkDouble(name=name, node_id=node_id, diversions=diversions, on_write_failure=on_write_failure)


def _make_failsink(name: str = "csv_failsink", node_id: str = "node-failsink") -> _SinkDouble:
    return _SinkDouble(name=name, node_id=node_id, artifact_path="/tmp/failsink")


def _make_executor() -> tuple[SinkExecutor, SimpleNamespace, SimpleNamespace]:
    execution = SimpleNamespace()
    execution.begin_node_state = _CallRecorder()
    execution.complete_node_state = _CallRecorder()
    execution.register_artifact = _CallRecorder()
    execution.record_routing_event = _CallRecorder()
    execution.begin_operation = _CallRecorder(return_value=SimpleNamespace(operation_id="op-1"))
    execution.complete_operation = _CallRecorder()
    data_flow = SimpleNamespace()
    data_flow.record_token_outcome = _CallRecorder()
    state_counter = [0]
    artifact_counter = [0]

    def _begin_state(**kwargs: Any) -> SimpleNamespace:
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
        )

    execution.begin_node_state.side_effect = _begin_state
    execution.register_artifact.side_effect = _register_artifact
    spans = _SpanFactoryDouble()
    executor = SinkExecutor(execution, data_flow, spans, "run-1")
    return executor, execution, data_flow  # type: ignore[return-value]


def _unique_completion_kwargs_by_state(completions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_state: dict[str, dict[str, Any]] = {}
    for kwargs in completions:
        state_id = kwargs["state_id"]
        assert state_id not in by_state, f"duplicate successful completion for {state_id}"
        by_state[state_id] = kwargs
    return by_state


def _complete_node_state_kwargs_by_state(execution: SimpleNamespace) -> dict[str, dict[str, Any]]:
    return _unique_completion_kwargs_by_state([call.kwargs for call in execution.complete_node_state.call_args_list])


def _assert_single_primary_divert_cleanup(
    execution: SimpleNamespace,
    *,
    phase: str,
    exception_type: str,
) -> None:
    # Failsink states now open BEFORE failsink I/O (elspeth-adaca19c75), so a
    # single diverted token has TWO open states (primary anchor + failsink
    # destination) and cleanup must terminalize both.
    begin_calls = execution.begin_node_state.call_args_list
    assert len(begin_calls) == 2
    assert begin_calls[0].kwargs["token_id"] == "t0"
    assert begin_calls[0].kwargs["node_id"] == "node-primary"
    assert begin_calls[1].kwargs["token_id"] == "t0"
    assert begin_calls[1].kwargs["node_id"] == "node-failsink"

    completion_by_state = _complete_node_state_kwargs_by_state(execution)
    assert set(completion_by_state) == {"state-1", "state-2"}
    for state_id in ("state-1", "state-2"):
        failed_kwargs = completion_by_state[state_id]
        assert failed_kwargs["status"] == NodeStateStatus.FAILED
        assert failed_kwargs["error"].phase == phase
        assert failed_kwargs["error"].exception_type == exception_type


class TestSinkWriteResultValidation:
    """Shared SinkWriteResult validation used by primary sink and failsink paths."""

    def test_primary_result_returns_artifact_and_diversions(self) -> None:
        diversions = (RowDiversion(row_index=0, reason="bad metadata", row_data={"field": "bad"}),)
        artifact = _make_artifact("/tmp/primary")
        result = SinkWriteResult(artifact=artifact, diversions=diversions)

        returned_artifact, returned_diversions = SinkExecutor._require_sink_write_result(
            label="primary",
            result=result,
            allow_diversions=True,
        )

        assert returned_artifact is artifact
        assert returned_diversions == diversions

    def test_primary_result_rejects_wrong_return_type(self) -> None:
        with pytest.raises(PluginContractViolation, match="Sink 'primary' returned dict, expected SinkWriteResult"):
            SinkExecutor._require_sink_write_result(
                label="primary",
                result={},
                allow_diversions=True,
            )

    def test_primary_result_rejects_wrong_artifact_type(self) -> None:
        result = object.__new__(SinkWriteResult)
        object.__setattr__(result, "artifact", "not-an-artifact")
        object.__setattr__(result, "diversions", ())

        with pytest.raises(
            PluginContractViolation,
            match="Sink 'primary' returned SinkWriteResult with artifact of type str, expected ArtifactDescriptor",
        ):
            SinkExecutor._require_sink_write_result(
                label="primary",
                result=result,
                allow_diversions=True,
            )

    def test_failsink_result_returns_artifact_when_diversions_absent(self) -> None:
        artifact = _make_artifact("/tmp/failsink")
        result = SinkWriteResult(artifact=artifact)

        returned_artifact, returned_diversions = SinkExecutor._require_sink_write_result(
            label="csv_failsink",
            result=result,
            allow_diversions=False,
        )

        assert returned_artifact is artifact
        assert returned_diversions == ()

    def test_failsink_result_rejects_diversions(self) -> None:
        diversions = (RowDiversion(row_index=0, reason="bad metadata", row_data={"field": "bad"}),)
        result = SinkWriteResult(artifact=_make_artifact("/tmp/failsink"), diversions=diversions)

        with pytest.raises(FrameworkBugError, match="Failsink 'csv_failsink' produced 1 diversions"):
            SinkExecutor._require_sink_write_result(
                label="csv_failsink",
                result=result,
                allow_diversions=False,
            )


class TestNoDiversions:
    """Existing behavior preserved when no diversions occur."""

    def test_all_tokens_get_completed_outcome(self) -> None:
        executor, _execution, data_flow = _make_executor()
        sink = _make_sink()
        tokens = [_make_token("t0"), _make_token("t1"), _make_token("t2")]
        executor.write(
            sink=sink,
            tokens=tokens,  # type: ignore[arg-type]
            ctx=_make_context(),
            step_in_pipeline=5,
            sink_name="primary",
            pending_outcome=_default_pending(),
        )
        outcome_calls = data_flow.record_token_outcome.call_args_list
        assert len(outcome_calls) == 3
        for c in outcome_calls:
            assert c.kwargs["outcome"] == TerminalOutcome.SUCCESS
            assert c.kwargs["path"] == TerminalPath.DEFAULT_FLOW
            assert c.kwargs["sink_name"] == "primary"

    def test_no_failsink_write_called(self) -> None:
        executor, _execution, _data_flow = _make_executor()
        sink = _make_sink()
        failsink = _make_failsink()
        tokens = [_make_token("t0")]
        executor.write(
            sink=sink,
            tokens=tokens,  # type: ignore[arg-type]
            ctx=_make_context(),
            step_in_pipeline=5,
            sink_name="primary",
            pending_outcome=_default_pending(),
            failsink=failsink,
        )
        failsink.write.assert_not_called()

    def test_returns_artifact_and_zero_diversions(self) -> None:
        executor, _execution, _data_flow = _make_executor()
        sink = _make_sink()
        tokens = [_make_token("t0")]
        artifact, diversion_counts = executor.write(
            sink=sink,
            tokens=tokens,  # type: ignore[arg-type]
            ctx=_make_context(),
            step_in_pipeline=5,
            sink_name="primary",
            pending_outcome=_default_pending(),
        )
        assert artifact is not None
        assert diversion_counts.total == 0


class TestDiscardMode:
    """on_write_failure='discard' — diverted rows are dropped with audit record."""

    def test_diverted_tokens_get_diverted_outcome(self) -> None:
        executor, _execution, data_flow = _make_executor()
        diversions = (RowDiversion(row_index=1, reason="bad metadata", row_data={"x": 1}),)
        sink = _make_sink(diversions=diversions, on_write_failure="discard")
        tokens = [_make_token("t0"), _make_token("t1")]
        executor.write(
            sink=sink,
            tokens=tokens,  # type: ignore[arg-type]
            ctx=_make_context(),
            step_in_pipeline=5,
            sink_name="primary",
            pending_outcome=_default_pending(),
        )
        outcome_calls = data_flow.record_token_outcome.call_args_list
        assert len(outcome_calls) == 2
        # Build a lookup by token_id for order-independence
        outcomes_by_token = {c.kwargs["ref"].token_id: c.kwargs for c in outcome_calls}
        # t0 (index 0) -> SUCCESS / DEFAULT_FLOW
        assert outcomes_by_token["t0"]["outcome"] == TerminalOutcome.SUCCESS
        assert outcomes_by_token["t0"]["path"] == TerminalPath.DEFAULT_FLOW
        assert outcomes_by_token["t0"]["sink_name"] == "primary"
        # t1 (index 1) -> FAILURE / SINK_DISCARDED
        assert outcomes_by_token["t1"]["outcome"] == TerminalOutcome.FAILURE
        assert outcomes_by_token["t1"]["path"] == TerminalPath.SINK_DISCARDED
        assert outcomes_by_token["t1"]["error_hash"] == hashlib.sha256(b"bad metadata").hexdigest()[:16]
        assert outcomes_by_token["t1"]["sink_name"] == "__discard__"

    def test_discard_mode_opens_primary_state_for_diverted_tokens(self) -> None:
        """Discard-mode diverted tokens get a FAILED node_state at the primary sink."""
        executor, execution, _data_flow = _make_executor()
        diversions = (RowDiversion(row_index=0, reason="bad metadata", row_data={"x": 1}),)
        sink = _make_sink(diversions=diversions, on_write_failure="discard")
        tokens = [_make_token("t0")]
        executor.write(
            sink=sink,
            tokens=tokens,  # type: ignore[arg-type]
            ctx=_make_context(),
            step_in_pipeline=5,
            sink_name="primary",
            pending_outcome=_default_pending(),
        )
        # Diverted token should get a begin_node_state at the primary sink
        begin_calls = execution.begin_node_state.call_args_list
        assert len(begin_calls) == 1
        assert begin_calls[0].kwargs["node_id"] == "node-primary"
        assert begin_calls[0].kwargs["token_id"] == "t0"
        # And a complete_node_state with FAILED status (row didn't reach destination)
        complete_calls = execution.complete_node_state.call_args_list
        assert len(complete_calls) == 1
        assert complete_calls[0].kwargs["status"] == NodeStateStatus.FAILED
        assert complete_calls[0].kwargs["output_data"]["discarded"] is True
        assert "bad metadata" in complete_calls[0].kwargs["output_data"]["reason"]

    def test_all_diverted_all_get_diverted(self) -> None:
        executor, _execution, data_flow = _make_executor()
        diversions = (
            RowDiversion(row_index=0, reason="bad", row_data={"x": 1}),
            RowDiversion(row_index=1, reason="bad", row_data={"x": 2}),
        )
        sink = _make_sink(diversions=diversions, on_write_failure="discard")
        tokens = [_make_token("t0"), _make_token("t1")]
        executor.write(
            sink=sink,
            tokens=tokens,  # type: ignore[arg-type]
            ctx=_make_context(),
            step_in_pipeline=5,
            sink_name="primary",
            pending_outcome=_default_pending(),
        )
        outcome_calls = data_flow.record_token_outcome.call_args_list
        assert all(c.kwargs["outcome"] == TerminalOutcome.FAILURE for c in outcome_calls)
        assert all(c.kwargs["path"] == TerminalPath.SINK_DISCARDED for c in outcome_calls)

    def test_returns_no_artifact_when_all_diverted(self) -> None:
        executor, _execution, _data_flow = _make_executor()
        diversions = (RowDiversion(row_index=0, reason="bad", row_data={"x": 1}),)
        sink = _make_sink(diversions=diversions, on_write_failure="discard")
        tokens = [_make_token("t0")]
        artifact, diversion_counts = executor.write(
            sink=sink,
            tokens=tokens,  # type: ignore[arg-type]
            ctx=_make_context(),
            step_in_pipeline=5,
            sink_name="primary",
            pending_outcome=_default_pending(),
        )
        assert artifact is None
        assert diversion_counts.discard_mode == 1
        assert diversion_counts.total == 1


class TestFailsinkMode:
    """on_write_failure=<sink_name> — diverted rows are written to failsink."""

    def test_failsink_missing_required_field_raises_layer1_violation(self) -> None:
        executor, _execution, data_flow = _make_executor()
        diversions = (RowDiversion(row_index=0, reason="invalid metadata", row_data={"doc": "hello"}),)
        sink = _make_sink(diversions=diversions, on_write_failure="csv_failsink")
        failsink = _make_failsink()
        failsink.declared_required_fields = frozenset({"__diversion_reason", "missing_field"})
        tokens = [_make_token("t0")]

        def _raise_failsink_required_fields(
            *,
            sink: Any,
            rows: list[dict[str, object]],
            tokens: list[TokenInfo],
            run_id: str,
            node_id: str,
            row_contracts: Any,
        ) -> None:
            del rows, row_contracts
            if sink.name != "csv_failsink":
                return
            violation = SinkRequiredFieldsViolation(
                plugin=sink.name,
                node_id=node_id,
                run_id=run_id,
                row_id=tokens[0].row_id,
                token_id=tokens[0].token_id,
                payload={
                    "declared": ["__diversion_reason", "missing_field"],
                    "runtime_observed": ["__diversion_reason"],
                    "missing": ["missing_field"],
                },
                message=(
                    "Sink 'csv_failsink' declared required fields "
                    "['__diversion_reason', 'missing_field'] but row is missing ['missing_field']"
                ),
            )
            _attach_contract_name_from_dispatcher(violation, "sink_required_fields")
            raise violation

        with (
            patch.object(SinkExecutor, "_run_sink_boundary_checks", autospec=True, side_effect=_raise_failsink_required_fields),
            pytest.raises(SinkRequiredFieldsViolation, match=r"declared required fields.*missing.*missing_field"),
        ):
            executor.write(
                sink=sink,
                tokens=tokens,  # type: ignore[arg-type]
                ctx=_make_context(),
                step_in_pipeline=5,
                sink_name="primary",
                pending_outcome=_default_pending(),
                failsink=failsink,
                failsink_name="csv_failsink",
                failsink_edge_id="edge-failsink-1",
            )

        failsink.write.assert_not_called()
        data_flow.record_token_outcome.assert_called_once()
        kwargs = data_flow.record_token_outcome.call_args.kwargs
        assert kwargs["ref"].token_id == "t0"
        assert kwargs["outcome"] == TerminalOutcome.FAILURE
        assert kwargs["path"] == TerminalPath.UNROUTED
        assert kwargs["context"]["exception_type"] == "SinkRequiredFieldsViolation"

    def test_failsink_write_called_with_enriched_rows(self) -> None:
        executor, _execution, _data_flow = _make_executor()
        diversions = (RowDiversion(row_index=1, reason="invalid metadata", row_data={"doc": "hello"}),)
        sink = _make_sink(diversions=diversions, on_write_failure="csv_failsink")
        failsink = _make_failsink()
        tokens = [_make_token("t0"), _make_token("t1")]
        executor.write(
            sink=sink,
            tokens=tokens,  # type: ignore[arg-type]
            ctx=_make_context(),
            step_in_pipeline=5,
            sink_name="primary",
            pending_outcome=_default_pending(),
            failsink=failsink,
            failsink_name="csv_failsink",
            failsink_edge_id="edge-failsink-1",
        )
        # Failsink should have been called with the diverted row
        failsink.write.assert_called_once()
        failsink_rows = failsink.write.call_args[0][0]
        assert len(failsink_rows) == 1
        assert "__diversion_reason" in failsink_rows[0]
        assert failsink_rows[0]["__diversion_reason"] == "invalid metadata"
        assert failsink_rows[0]["__diverted_from"] == "primary"
        assert "__diversion_timestamp" in failsink_rows[0]

    def test_failsink_flush_called(self) -> None:
        executor, _execution, _data_flow = _make_executor()
        diversions = (RowDiversion(row_index=0, reason="bad", row_data={"x": 1}),)
        sink = _make_sink(diversions=diversions, on_write_failure="csv_failsink")
        failsink = _make_failsink()
        tokens = [_make_token("t0")]
        executor.write(
            sink=sink,
            tokens=tokens,  # type: ignore[arg-type]
            ctx=_make_context(),
            step_in_pipeline=5,
            sink_name="primary",
            pending_outcome=_default_pending(),
            failsink=failsink,
            failsink_name="csv_failsink",
            failsink_edge_id="edge-failsink-1",
        )
        failsink.flush.assert_called_once()

    def test_no_diversions_no_failsink_call(self) -> None:
        executor, _execution, _data_flow = _make_executor()
        sink = _make_sink(diversions=(), on_write_failure="csv_failsink")
        failsink = _make_failsink()
        tokens = [_make_token("t0")]
        executor.write(
            sink=sink,
            tokens=tokens,  # type: ignore[arg-type]
            ctx=_make_context(),
            step_in_pipeline=5,
            sink_name="primary",
            pending_outcome=_default_pending(),
            failsink=failsink,
        )
        failsink.write.assert_not_called()

    def test_diverted_tokens_get_failsink_sink_name(self) -> None:
        executor, _execution, data_flow = _make_executor()
        diversions = (RowDiversion(row_index=0, reason="bad", row_data={"x": 1}),)
        sink = _make_sink(diversions=diversions, on_write_failure="csv_failsink")
        failsink = _make_failsink()
        tokens = [_make_token("t0")]
        executor.write(
            sink=sink,
            tokens=tokens,  # type: ignore[arg-type]
            ctx=_make_context(),
            step_in_pipeline=5,
            sink_name="primary",
            pending_outcome=_default_pending(),
            failsink=failsink,
            failsink_name="csv_failsink",
            failsink_edge_id="edge-failsink-1",
        )
        outcome_calls = data_flow.record_token_outcome.call_args_list
        assert outcome_calls[0].kwargs["sink_name"] == "csv_failsink"

    def test_routing_event_recorded_for_diverted_tokens(self) -> None:
        """Failsink mode must record routing_event linking primary -> failsink."""
        executor, execution, _data_flow = _make_executor()
        diversions = (RowDiversion(row_index=0, reason="bad metadata", row_data={"x": 1}),)
        sink = _make_sink(diversions=diversions, on_write_failure="csv_failsink")
        failsink = _make_failsink()
        tokens = [_make_token("t0")]
        executor.write(
            sink=sink,
            tokens=tokens,  # type: ignore[arg-type]
            ctx=_make_context(),
            step_in_pipeline=5,
            sink_name="primary",
            pending_outcome=_default_pending(),
            failsink=failsink,
            failsink_name="csv_failsink",
            failsink_edge_id="edge-failsink-1",
        )
        # routing_event must be anchored to the PRIMARY sink state (the routing
        # decision point), NOT the failsink state. This is the critical invariant:
        # routing events live at the node that made the routing decision.
        execution.record_routing_event.assert_called_once()
        call_kwargs = execution.record_routing_event.call_args.kwargs
        assert call_kwargs["edge_id"] == "edge-failsink-1"
        assert call_kwargs["mode"] == RoutingMode.DIVERT
        assert "bad metadata" in call_kwargs["reason"]["diversion_reason"]
        # state-1 is the primary divert state for t0 (first begin_node_state call).
        # If this were anchored to the failsink state (state-2), the old bug is back.
        assert call_kwargs["state_id"] == "state-1"

    def test_both_artifacts_registered_in_mixed_batch(self) -> None:
        """Mixed batch: primary artifact + failsink artifact both registered."""
        executor, execution, _data_flow = _make_executor()
        diversions = (RowDiversion(row_index=1, reason="bad", row_data={"x": 1}),)
        sink = _make_sink(diversions=diversions, on_write_failure="csv_failsink")
        failsink = _make_failsink()
        primary_artifact = ArtifactDescriptor.for_file(
            path="/tmp/primary.csv",
            content_hash="a" * 64,
            size_bytes=101,
        )
        failsink_artifact = ArtifactDescriptor.for_file(
            path="/tmp/failsink.csv",
            content_hash="b" * 64,
            size_bytes=202,
        )
        sink.write.return_value = SinkWriteResult(
            artifact=primary_artifact,
            diversions=diversions,
        )
        failsink.write.return_value = SinkWriteResult(artifact=failsink_artifact)
        tokens = [_make_token("t0"), _make_token("t1")]
        executor.write(
            sink=sink,
            tokens=tokens,  # type: ignore[arg-type]
            ctx=_make_context(),
            step_in_pipeline=5,
            sink_name="primary",
            pending_outcome=_default_pending(),
            failsink=failsink,
            failsink_name="csv_failsink",
            failsink_edge_id="edge-failsink-1",
        )

        begin_calls = execution.begin_node_state.call_args_list
        assert [(call.kwargs["token_id"], call.kwargs["node_id"]) for call in begin_calls] == [
            ("t0", "node-primary"),
            ("t1", "node-primary"),
            ("t1", "node-failsink"),
        ]

        artifact_calls = [call.kwargs for call in execution.register_artifact.call_args_list]
        assert artifact_calls == [
            {
                "run_id": "run-1",
                "state_id": "state-1",
                "sink_node_id": "node-primary",
                "artifact_type": primary_artifact.artifact_type,
                "path": primary_artifact.path_or_uri,
                "content_hash": primary_artifact.content_hash,
                "size_bytes": primary_artifact.size_bytes,
            },
            {
                "run_id": "run-1",
                "state_id": "state-3",
                "sink_node_id": "node-failsink",
                "artifact_type": failsink_artifact.artifact_type,
                "path": failsink_artifact.path_or_uri,
                "content_hash": failsink_artifact.content_hash,
                "size_bytes": failsink_artifact.size_bytes,
            },
        ]
        assert artifact_calls[0]["state_id"] != artifact_calls[1]["state_id"]
        assert artifact_calls[0]["path"] != artifact_calls[1]["path"]
        assert artifact_calls[0]["content_hash"] != artifact_calls[1]["content_hash"]

    def test_node_states_opened_at_correct_nodes(self) -> None:
        """Primary tokens get states at primary node, diverted at failsink node."""
        executor, execution, _data_flow = _make_executor()
        diversions = (RowDiversion(row_index=1, reason="bad", row_data={"x": 1}),)
        sink = _make_sink(diversions=diversions, on_write_failure="csv_failsink")
        failsink = _make_failsink()
        tokens = [_make_token("t0"), _make_token("t1")]
        executor.write(
            sink=sink,
            tokens=tokens,  # type: ignore[arg-type]
            ctx=_make_context(),
            step_in_pipeline=5,
            sink_name="primary",
            pending_outcome=_default_pending(),
            failsink=failsink,
            failsink_name="csv_failsink",
            failsink_edge_id="edge-failsink-1",
        )
        begin_calls = execution.begin_node_state.call_args_list
        # 3 states: t0 at primary, t1 at primary (divert anchor), t1 at failsink
        assert len(begin_calls) == 3
        primary_calls = [c for c in begin_calls if c.kwargs["node_id"] == "node-primary"]
        failsink_calls = [c for c in begin_calls if c.kwargs["node_id"] == "node-failsink"]
        assert len(primary_calls) == 2  # t0 (written) + t1 (divert anchor)
        assert len(failsink_calls) == 1  # t1 (destination)


class TestFailsinkErrorHandling:
    def test_failsink_write_failure_crashes(self) -> None:
        """If failsink write fails, crash — it's the last resort."""
        executor, _execution, _data_flow = _make_executor()
        diversions = (RowDiversion(row_index=0, reason="bad", row_data={"x": 1}),)
        sink = _make_sink(diversions=diversions, on_write_failure="csv_failsink")
        failsink = _make_failsink()
        failsink.write.side_effect = OSError("disk full")
        tokens = [_make_token("t0")]
        with pytest.raises(OSError, match="disk full"):
            executor.write(
                sink=sink,
                tokens=tokens,  # type: ignore[arg-type]
                ctx=_make_context(),
                step_in_pipeline=5,
                sink_name="primary",
                pending_outcome=_default_pending(),
                failsink=failsink,
                failsink_name="csv_failsink",
                failsink_edge_id="edge-failsink-1",
            )


class TestFailsinkStateOrdering:
    """Failsink node_states open BEFORE external failsink I/O (elspeth-adaca19c75).

    Mirrors the primary path's open-before-I/O invariant: a durable failsink
    write must never exist without a failsink node_state — a crash between
    flush and audit recording would otherwise leave a durable artifact with
    no node_state, routing_event, DIVERTED outcome, or checkpoint.
    """

    def test_failsink_states_opened_before_external_failsink_write(self) -> None:
        executor, execution, _data_flow = _make_executor()
        diversions = (RowDiversion(row_index=0, reason="bad", row_data={"x": 1}),)
        sink = _make_sink(diversions=diversions, on_write_failure="csv_failsink")
        failsink = _make_failsink()
        tokens = [_make_token("t0")]

        call_order: list[str] = []
        orig_begin = execution.begin_node_state.side_effect

        def _begin(**kwargs: Any) -> Any:
            if kwargs.get("node_id") == failsink.node_id:
                call_order.append("failsink_begin_node_state")
            return orig_begin(**kwargs)

        def _write(rows: Any, ctx: Any) -> SinkWriteResult:
            call_order.append("failsink_write")
            return SinkWriteResult(artifact=_make_artifact("/tmp/failsink"))

        execution.begin_node_state.side_effect = _begin
        failsink.write.side_effect = _write

        executor.write(
            sink=sink,
            tokens=tokens,  # type: ignore[arg-type]
            ctx=_make_context(),
            step_in_pipeline=5,
            sink_name="primary",
            pending_outcome=_default_pending(),
            failsink=failsink,
            failsink_name="csv_failsink",
            failsink_edge_id="edge-failsink-1",
        )

        assert call_order == ["failsink_begin_node_state", "failsink_write"], (
            f"failsink node_state must open BEFORE the external failsink write, got: {call_order}"
        )


class TestFailsinkCleanup:
    """Verify node_state recording when failsink write/flush fails."""

    def test_failsink_write_failure_completes_failsink_states_as_failed(self) -> None:
        """When failsink.write() raises, the pre-opened failsink states FAIL.

        Batch: 1 token, 1 diversion. Failsink node_states open BEFORE the
        external failsink write (elspeth-adaca19c75), so a write crash must
        terminalize BOTH the primary divert anchor and the failsink state.
        """
        executor, execution, _data_flow = _make_executor()
        diversions = (RowDiversion(row_index=0, reason="bad", row_data={"x": 1}),)
        sink = _make_sink(diversions=diversions, on_write_failure="csv_failsink")
        failsink = _make_failsink()
        failsink.write.side_effect = OSError("disk full")
        tokens = [_make_token("t0")]
        with pytest.raises(OSError):
            executor.write(
                sink=sink,
                tokens=tokens,  # type: ignore[arg-type]
                ctx=_make_context(),
                step_in_pipeline=5,
                sink_name="primary",
                pending_outcome=_default_pending(),
                failsink=failsink,
                failsink_name="csv_failsink",
                failsink_edge_id="edge-failsink-1",
            )
        # t0's primary divert anchor AND its pre-opened failsink state were
        # open when the failsink write crashed — cleanup fails BOTH.
        complete_calls = execution.complete_node_state.call_args_list
        failed_calls = [c for c in complete_calls if c.kwargs.get("status") == NodeStateStatus.FAILED]
        completed_calls = [c for c in complete_calls if c.kwargs.get("status") == NodeStateStatus.COMPLETED]
        assert len(failed_calls) == 2  # primary divert anchor + failsink state
        assert len(completed_calls) == 0
        # Failsink state opened BEFORE the write (open-before-I/O invariant)
        begin_calls = execution.begin_node_state.call_args_list
        primary_begins = [c for c in begin_calls if c.kwargs.get("node_id") == sink.node_id]
        failsink_begins = [c for c in begin_calls if c.kwargs.get("node_id") == failsink.node_id]
        assert len(primary_begins) == 1  # divert anchor
        assert len(failsink_begins) == 1

    def test_failsink_failure_does_not_affect_primary_states(self) -> None:
        """Primary COMPLETED states remain intact when failsink fails.

        Batch: 2 tokens, 1 diversion at index 1.
        Expect: t0 COMPLETED at primary; t1's divert anchor AND its pre-opened
        failsink state both FAIL when the failsink write crashes.
        """
        executor, execution, _data_flow = _make_executor()
        diversions = (RowDiversion(row_index=1, reason="bad", row_data={"x": 1}),)
        sink = _make_sink(diversions=diversions, on_write_failure="csv_failsink")
        failsink = _make_failsink()
        failsink.write.side_effect = OSError("disk full")
        tokens = [_make_token("t0"), _make_token("t1")]
        with pytest.raises(OSError):
            executor.write(
                sink=sink,
                tokens=tokens,  # type: ignore[arg-type]
                ctx=_make_context(),
                step_in_pipeline=5,
                sink_name="primary",
                pending_outcome=_default_pending(),
                failsink=failsink,
                failsink_name="csv_failsink",
                failsink_edge_id="edge-failsink-1",
            )
        complete_calls = execution.complete_node_state.call_args_list
        # t0: COMPLETED at primary (Phase 2)
        # t1: FAILED at primary (divert anchor) + FAILED at failsink (pre-opened)
        completed_calls = [c for c in complete_calls if c.kwargs.get("status") == NodeStateStatus.COMPLETED]
        failed_calls = [c for c in complete_calls if c.kwargs.get("status") == NodeStateStatus.FAILED]
        assert len(completed_calls) == 1  # t0
        assert len(failed_calls) == 2  # t1 primary divert anchor + t1 failsink state
        # Verify: 2 primary states opened (t0 + t1 divert anchor), 1 failsink
        # state (opened BEFORE the crashing write — open-before-I/O invariant)
        begin_calls = execution.begin_node_state.call_args_list
        primary_begins = [c for c in begin_calls if c.kwargs.get("node_id") == sink.node_id]
        failsink_begins = [c for c in begin_calls if c.kwargs.get("node_id") == failsink.node_id]
        assert len(primary_begins) == 2  # t0 + t1 divert anchor
        assert len(failsink_begins) == 1

    def test_failsink_flush_failure_crashes(self) -> None:
        """If failsink.flush() raises, crash — it's the last resort."""
        executor, _execution, _data_flow = _make_executor()
        diversions = (RowDiversion(row_index=0, reason="bad", row_data={"x": 1}),)
        sink = _make_sink(diversions=diversions, on_write_failure="csv_failsink")
        failsink = _make_failsink()
        failsink.flush.side_effect = OSError("disk full")
        tokens = [_make_token("t0")]
        with pytest.raises(OSError, match="disk full"):
            executor.write(
                sink=sink,
                tokens=tokens,  # type: ignore[arg-type]
                ctx=_make_context(),
                step_in_pipeline=5,
                sink_name="primary",
                pending_outcome=_default_pending(),
                failsink=failsink,
                failsink_name="csv_failsink",
                failsink_edge_id="edge-failsink-1",
            )


class TestFailsinkCleanupEnvelope:
    """Regression: cleanup must cover all failsink-mode raises after states exist."""

    def test_failsink_reset_diversion_log_failure_cleans_primary_divert_states(self) -> None:
        """Failsink setup errors must terminalize pre-opened primary divert states."""
        executor, execution, _data_flow = _make_executor()
        diversions = (RowDiversion(row_index=0, reason="bad", row_data={"x": 1}),)
        sink = _make_sink(diversions=diversions, on_write_failure="csv_failsink")
        failsink = _make_failsink()
        failsink._reset_diversion_log.side_effect = RuntimeError("failsink reset boom")
        tokens = [_make_token("t0")]

        with pytest.raises(RuntimeError, match="failsink reset boom"):
            executor.write(
                sink=sink,
                tokens=tokens,  # type: ignore[arg-type]
                ctx=_make_context(),
                step_in_pipeline=5,
                sink_name="primary",
                pending_outcome=_default_pending(),
                failsink=failsink,
                failsink_name="csv_failsink",
                failsink_edge_id="edge-failsink-1",
            )

        _assert_single_primary_divert_cleanup(
            execution,
            phase="failsink_write",
            exception_type="RuntimeError",
        )

    def test_invalid_failsink_result_type_cleans_primary_divert_states(self) -> None:
        """Malformed failsink return values must fail loudly after cleanup.

        failsink.write() is a first-party system-owned method annotated
        -> SinkWriteResult; a wrong return type is a plugin bug. The executor
        guards the return type and raises PluginContractViolation (an offensive
        crash with a named error), not an AttributeError. Primary-divert
        cleanup still runs because PluginContractViolation is not in
        TIER_1_ERRORS and falls through to the ``except Exception`` cleanup arm.
        """
        executor, execution, _data_flow = _make_executor()
        diversions = (RowDiversion(row_index=0, reason="bad", row_data={"x": 1}),)
        sink = _make_sink(diversions=diversions, on_write_failure="csv_failsink")
        failsink = _make_failsink()
        failsink.write.return_value = object()
        tokens = [_make_token("t0")]

        with pytest.raises(PluginContractViolation, match=r"expected SinkWriteResult"):
            executor.write(
                sink=sink,
                tokens=tokens,  # type: ignore[arg-type]
                ctx=_make_context(),
                step_in_pipeline=5,
                sink_name="primary",
                pending_outcome=_default_pending(),
                failsink=failsink,
                failsink_name="csv_failsink",
                failsink_edge_id="edge-failsink-1",
            )

        _assert_single_primary_divert_cleanup(
            execution,
            phase="failsink_write",
            exception_type="PluginContractViolation",
        )


class TestFailsinkOperationAndSpanRecording:
    """Failsink writes are real sink I/O and need their own operation/span records."""

    def test_failsink_write_records_separate_operation_and_sink_span(self) -> None:
        """Diverted writes should produce primary and failsink operation/span pairs."""
        executor, execution, _data_flow = _make_executor()
        diversions = (RowDiversion(row_index=0, reason="bad", row_data={"x": 1}),)
        sink = _make_sink(diversions=diversions, on_write_failure="csv_failsink")
        failsink = _make_failsink()
        tokens = [_make_token("t0")]
        ctx = _make_context()

        executor.write(
            sink=sink,
            tokens=tokens,  # type: ignore[arg-type]
            ctx=ctx,
            step_in_pipeline=5,
            sink_name="primary",
            pending_outcome=_default_pending(),
            failsink=failsink,
            failsink_name="csv_failsink",
            failsink_edge_id="edge-failsink-1",
        )

        assert execution.begin_operation.call_count == 2
        operation_node_ids = [call.kwargs["node_id"] for call in execution.begin_operation.call_args_list]
        assert operation_node_ids == ["node-primary", "node-failsink"]

        assert executor._spans.sink_span.call_count == 2
        span_names = [call.args[0] for call in executor._spans.sink_span.call_args_list]
        span_node_ids = [call.kwargs["node_id"] for call in executor._spans.sink_span.call_args_list]
        assert span_names == ["primary", "csv_failsink"]
        assert span_node_ids == ["node-primary", "node-failsink"]


class TestNonContiguousDiversions:
    """Verify correct partitioning when diverted rows are non-contiguous."""

    def test_non_contiguous_diversions(self) -> None:
        """Rows 0 and 2 diverted, row 1 primary. Outcomes correctly partitioned.

        Uses token_id keying, not call ordering -- the executor may process
        primary tokens before diverted tokens.
        """
        executor, _execution, data_flow = _make_executor()
        diversions = (
            RowDiversion(row_index=0, reason="bad", row_data={"x": 1}),
            RowDiversion(row_index=2, reason="bad", row_data={"x": 3}),
        )
        sink = _make_sink(diversions=diversions, on_write_failure="discard")
        tokens = [_make_token("t0"), _make_token("t1"), _make_token("t2")]
        executor.write(
            sink=sink,
            tokens=tokens,  # type: ignore[arg-type]
            ctx=_make_context(),
            step_in_pipeline=5,
            sink_name="primary",
            pending_outcome=_default_pending(),
        )
        outcome_calls = data_flow.record_token_outcome.call_args_list
        outcomes_by_token = {c.kwargs["ref"].token_id: c.kwargs for c in outcome_calls}
        assert outcomes_by_token["t0"]["outcome"] == TerminalOutcome.FAILURE
        assert outcomes_by_token["t0"]["path"] == TerminalPath.SINK_DISCARDED
        assert outcomes_by_token["t1"]["outcome"] == TerminalOutcome.SUCCESS
        assert outcomes_by_token["t1"]["path"] == TerminalPath.DEFAULT_FLOW
        assert outcomes_by_token["t2"]["outcome"] == TerminalOutcome.FAILURE
        assert outcomes_by_token["t2"]["path"] == TerminalPath.SINK_DISCARDED


class TestEmptyBatch:
    """Verify behavior when no tokens are provided."""

    def test_empty_batch_with_failsink_configured(self) -> None:
        """Empty token list with failsink configured -- no-op, no crash."""
        executor, _execution, data_flow = _make_executor()
        sink = _make_sink(on_write_failure="csv_failsink")
        failsink = _make_failsink()
        artifact, diversion_counts = executor.write(
            sink=sink,
            tokens=[],
            ctx=_make_context(),
            step_in_pipeline=5,
            sink_name="primary",
            pending_outcome=_default_pending(),
            failsink=failsink,
            failsink_name="csv_failsink",
            failsink_edge_id="edge-failsink-1",
        )
        assert artifact is None
        assert diversion_counts.total == 0
        failsink.write.assert_not_called()
        data_flow.record_token_outcome.assert_not_called()


class TestOnTokenWrittenWithDiversions:
    """Verify on_token_written is called for ALL tokens after their path completes.

    Primary tokens are checkpointed after Phase 2 (sink write durable).
    Diverted tokens are checkpointed after Phase 3 (failsink/discard durable).
    Both must be checkpointed to prevent duplicate writes on resume.
    """

    def test_on_token_written_called_for_all_tokens_discard_mode(self) -> None:
        """Both primary and diverted tokens must be checkpointed (discard mode)."""
        executor, _execution, _data_flow = _make_executor()
        diversions = (RowDiversion(row_index=1, reason="bad", row_data={"x": 1}),)
        sink = _make_sink(diversions=diversions, on_write_failure="discard")
        tokens = [_make_token("t0"), _make_token("t1")]
        callback = _CallRecorder()
        executor.write(
            sink=sink,
            tokens=tokens,  # type: ignore[arg-type]
            ctx=_make_context(),
            step_in_pipeline=5,
            sink_name="primary",
            pending_outcome=_default_pending(),
            on_token_written=callback,
        )
        # Both tokens checkpointed: t0 after primary write, t1 after discard
        assert callback.call_count == 2
        checkpointed_ids = {c[0][0].token_id for c in callback.call_args_list}
        assert checkpointed_ids == {"t0", "t1"}

    def test_on_token_written_called_for_all_tokens_failsink_mode(self) -> None:
        """Both primary and diverted tokens must be checkpointed (failsink mode)."""
        executor, _execution, _data_flow = _make_executor()
        diversions = (RowDiversion(row_index=1, reason="bad", row_data={"x": 1}),)
        sink = _make_sink(diversions=diversions, on_write_failure="csv_failsink")
        failsink = _make_failsink()
        tokens = [_make_token("t0"), _make_token("t1")]
        callback = _CallRecorder()
        executor.write(
            sink=sink,
            tokens=tokens,  # type: ignore[arg-type]
            ctx=_make_context(),
            step_in_pipeline=5,
            sink_name="primary",
            pending_outcome=_default_pending(),
            failsink=failsink,
            failsink_name="csv_failsink",
            failsink_edge_id="edge-failsink-1",
            on_token_written=callback,
        )
        # Both tokens checkpointed: t0 after primary write, t1 after failsink write
        assert callback.call_count == 2
        checkpointed_ids = {c[0][0].token_id for c in callback.call_args_list}
        assert checkpointed_ids == {"t0", "t1"}

    def test_primary_tokens_checkpointed_before_diverted(self) -> None:
        """Primary tokens are checkpointed in Phase 2, diverted in Phase 3."""
        executor, _execution, _data_flow = _make_executor()
        diversions = (RowDiversion(row_index=1, reason="bad", row_data={"x": 1}),)
        sink = _make_sink(diversions=diversions, on_write_failure="discard")
        tokens = [_make_token("t0"), _make_token("t1")]
        callback = _CallRecorder()
        executor.write(
            sink=sink,
            tokens=tokens,  # type: ignore[arg-type]
            ctx=_make_context(),
            step_in_pipeline=5,
            sink_name="primary",
            pending_outcome=_default_pending(),
            on_token_written=callback,
        )
        # t0 (primary) checkpointed first, t1 (diverted) checkpointed second
        assert callback.call_count == 2
        assert callback.call_args_list[0][0][0].token_id == "t0"
        assert callback.call_args_list[1][0][0].token_id == "t1"


class TestMidLoopAuditRecordingCleanup:
    """Tests for the completed_primary_indices/completed_failsink_indices cleanup.

    When recorder calls fail mid-loop during failsink diversion recording,
    remaining OPEN states must be completed as FAILED (not left permanently OPEN).
    """

    def test_recorder_failure_mid_loop_cleans_remaining_states(self) -> None:
        """2 diversions, recorder fails on 2nd routing_event → 2nd token's states cleaned up.

        After successfully recording token 0's routing_event + primary FAILED +
        failsink COMPLETED, the recorder fails on token 1's routing_event.
        Token 1's primary and failsink states must be completed as FAILED.
        """
        executor, execution, _data_flow = _make_executor()

        diversions = (
            RowDiversion(row_index=0, reason="bad0", row_data={"x": 0}),
            RowDiversion(row_index=1, reason="bad1", row_data={"x": 1}),
        )
        sink = _make_sink(diversions=diversions, on_write_failure="csv_failsink")
        failsink = _make_failsink()
        tokens = [_make_token("t0"), _make_token("t1")]

        # record_routing_event succeeds on first call, fails on second
        call_count = [0]

        def routing_event_side_effect(**kwargs: Any) -> None:
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("DB connection lost mid-loop")

        execution.record_routing_event.side_effect = routing_event_side_effect

        with pytest.raises(RuntimeError, match="DB connection lost"):
            executor.write(
                sink=sink,
                tokens=tokens,  # type: ignore[arg-type]
                ctx=_make_context(),
                step_in_pipeline=5,
                sink_name="primary",
                pending_outcome=_default_pending(),
                failsink=failsink,
                failsink_name="csv_failsink",
                failsink_edge_id="edge-failsink-1",
            )

        # Token 0: fully recorded (primary FAILED + failsink COMPLETED)
        # Token 1: cleanup marked both states as FAILED
        complete_calls = execution.complete_node_state.call_args_list
        failed_state_ids = {c.kwargs["state_id"] for c in complete_calls if c.kwargs.get("status") == NodeStateStatus.FAILED}
        completed_state_ids = {c.kwargs["state_id"] for c in complete_calls if c.kwargs.get("status") == NodeStateStatus.COMPLETED}

        # Assert by state_id sets rather than raw counts — resilient to call ordering
        # Token 0 got states 1 (primary-divert), 3 (failsink) — 2 was the primary write state
        # Token 1 got states 4 (primary-divert), 5 (failsink) — opened but never completed normally
        # FAILED: tok0-primary-divert + tok1-primary-divert + tok1-failsink (3 states)
        # COMPLETED: tok0-failsink (1 state)
        assert len(failed_state_ids) == 3
        assert len(completed_state_ids) == 1
        # No overlap between FAILED and COMPLETED
        assert failed_state_ids & completed_state_ids == set()


class TestCompletePrimaryFailureClosesDivertAnchors:
    """Phase-2 (_complete_primary) failures must not leave diverted anchors OPEN.

    Regression tests for elspeth-5a5e83d3e5: a failure while recording the
    primary tokens' completions/artifact/outcomes propagates out of write()
    BEFORE Phase 3 runs, so the diverted tokens' pre-opened primary node_states
    must be closed FAILED by the Phase-2 cleanup envelope — nothing downstream
    would ever terminalize them.

    RuntimeError deliberately exercises the generic arm; AuditIntegrityError
    the TIER_1 best-effort arm (same convention as
    TestUncoveredExceptArmCharacterization).
    """

    def test_outcome_recording_failure_closes_divert_anchor(self) -> None:
        """A generic error from record_token_outcome (primary states already
        COMPLETED) must close the diverted token's primary anchor as FAILED
        with phase='primary_audit_recording' before re-raising."""
        executor, execution, data_flow = _make_executor()
        sink = _make_sink(diversions=(RowDiversion(row_index=1, reason="bad", row_data={"x": 1}),))
        tokens = [_make_token("t0"), _make_token("t1")]
        data_flow.record_token_outcome.side_effect = RuntimeError("DB connection lost")

        with pytest.raises(RuntimeError, match="DB connection lost"):
            executor.write(
                sink=sink,
                tokens=tokens,  # type: ignore[arg-type]
                ctx=_make_context(),
                step_in_pipeline=5,
                sink_name="primary",
                pending_outcome=_default_pending(),
            )

        # state-1 = t0 primary (COMPLETED before the outcome loop raised);
        # state-2 = t1 divert anchor (closed FAILED by the Phase-2 envelope).
        completion_by_state = _complete_node_state_kwargs_by_state(execution)
        assert set(completion_by_state) == {"state-1", "state-2"}
        assert completion_by_state["state-1"]["status"] == NodeStateStatus.COMPLETED
        failed_kwargs = completion_by_state["state-2"]
        assert failed_kwargs["status"] == NodeStateStatus.FAILED
        assert failed_kwargs["error"].phase == "primary_audit_recording"
        assert failed_kwargs["error"].exception_type == "RuntimeError"

    def test_register_artifact_tier1_failure_closes_divert_anchor(self) -> None:
        """A TIER_1 error from register_artifact hits the best-effort arm:
        the diverted token's primary anchor is closed FAILED, the completed
        primary state is left untouched, and the original error re-raises."""
        executor, execution, _data_flow = _make_executor()
        sink = _make_sink(diversions=(RowDiversion(row_index=1, reason="bad", row_data={"x": 1}),))
        tokens = [_make_token("t0"), _make_token("t1")]
        execution.register_artifact.side_effect = AuditIntegrityError("artifact registry corrupted")

        with pytest.raises(AuditIntegrityError, match="artifact registry corrupted"):
            executor.write(
                sink=sink,
                tokens=tokens,  # type: ignore[arg-type]
                ctx=_make_context(),
                step_in_pipeline=5,
                sink_name="primary",
                pending_outcome=_default_pending(),
            )

        completion_by_state = _complete_node_state_kwargs_by_state(execution)
        assert set(completion_by_state) == {"state-1", "state-2"}
        assert completion_by_state["state-1"]["status"] == NodeStateStatus.COMPLETED
        failed_kwargs = completion_by_state["state-2"]
        assert failed_kwargs["status"] == NodeStateStatus.FAILED
        assert failed_kwargs["error"].phase == "primary_audit_recording"
        assert failed_kwargs["error"].exception_type == "AuditIntegrityError"

    def test_primary_completion_failure_closes_primary_and_divert_anchor(self) -> None:
        """A failure while completing the primary states themselves must close
        BOTH the not-yet-completed primary state and the divert anchor as
        FAILED (per-state progress tracking, mirroring the Phase-3 pattern)."""
        executor, execution, _data_flow = _make_executor()
        sink = _make_sink(diversions=(RowDiversion(row_index=1, reason="bad", row_data={"x": 1}),))
        tokens = [_make_token("t0"), _make_token("t1")]
        # First complete_node_state call (t0's COMPLETED attempt) raises;
        # the two cleanup closes that follow succeed.
        execution.complete_node_state.side_effect = [RuntimeError("audit write lost"), None, None]

        with pytest.raises(RuntimeError, match="audit write lost"):
            executor.write(
                sink=sink,
                tokens=tokens,  # type: ignore[arg-type]
                ctx=_make_context(),
                step_in_pipeline=5,
                sink_name="primary",
                pending_outcome=_default_pending(),
            )

        # The failed COMPLETED attempt for state-1 is also recorded by the
        # _CallRecorder, so filter by status rather than asserting uniqueness.
        complete_calls = execution.complete_node_state.call_args_list
        failed_by_state = {c.kwargs["state_id"]: c.kwargs for c in complete_calls if c.kwargs.get("status") == NodeStateStatus.FAILED}
        assert set(failed_by_state) == {"state-1", "state-2"}
        for state_id, kwargs in failed_by_state.items():
            assert kwargs["error"].phase == "primary_audit_recording"
            assert kwargs["error"].exception_type == "RuntimeError", state_id


class TestSystemErrorStateCleanup:
    """Regression: FrameworkBugError/AuditIntegrityError paths must close OPEN states.

    Bug: The `except (FrameworkBugError, AuditIntegrityError): raise` handlers
    in SinkExecutor.write() skipped cleanup, leaving node_states permanently
    OPEN in the audit trail — a Tier 1 integrity violation. The non-system
    exception handlers RIGHT BELOW each site show correct cleanup, but system
    error paths just re-raised.

    Fix: Best-effort cleanup before re-raising system errors. If cleanup itself
    fails, log and preserve the original error.
    """

    def test_failsink_begin_node_state_system_error_cleans_up_open_states(self) -> None:
        """When failsink begin_node_state raises AuditIntegrityError, all OPEN states close.

        Setup: 2 diverted tokens. Failsink begin_node_state succeeds for token 0
        then raises AuditIntegrityError for token 1. At that point:
        - 2 primary divert states are OPEN (from Phase 3 begin_node_state)
        - 1 failsink state is OPEN (token 0)
        All 3 must be closed as FAILED before the error propagates.
        """
        executor, execution, _data_flow = _make_executor()
        sink = _make_sink(
            diversions=(
                RowDiversion(row_index=0, reason="bad-0", row_data={"field": "v0"}),
                RowDiversion(row_index=1, reason="bad-1", row_data={"field": "v1"}),
            ),
        )
        failsink = _make_failsink()
        tokens = [_make_token("t0"), _make_token("t1")]

        # begin_node_state: 2 primary divert states succeed, then 1 failsink
        # state succeeds, then the 2nd failsink state raises AuditIntegrityError.
        call_count = [0]
        original_side_effect = execution.begin_node_state.side_effect

        def begin_state_with_error(**kwargs: Any) -> SimpleNamespace:
            call_count[0] += 1
            # Calls 1-2: primary divert states (OK)
            # Call 3: first failsink state (OK)
            # Call 4: second failsink state (BOOM)
            if call_count[0] == 4:
                raise AuditIntegrityError("FK violation on failsink begin_node_state")
            return original_side_effect(**kwargs)  # type: ignore[no-any-return]

        execution.begin_node_state.side_effect = begin_state_with_error

        with pytest.raises(AuditIntegrityError, match="FK violation"):
            executor.write(
                sink=sink,
                tokens=tokens,  # type: ignore[arg-type]
                ctx=_make_context(),
                step_in_pipeline=5,
                sink_name="primary",
                pending_outcome=_default_pending(),
                failsink=failsink,
                failsink_name="csv_failsink",
                failsink_edge_id="edge-divert-1",
            )

        begin_calls = execution.begin_node_state.call_args_list
        assert [(call.kwargs["token_id"], call.kwargs["node_id"]) for call in begin_calls] == [
            ("t0", "node-primary"),
            ("t1", "node-primary"),
            ("t0", "node-failsink"),
            ("t1", "node-failsink"),
        ]

        # Only the first three begin calls returned a state. The fourth raised
        # before opening a failsink state for t1, so cleanup must close exactly
        # state-1/state-2 (primary) and state-3 (failsink).
        completion_by_state = _complete_node_state_kwargs_by_state(execution)
        assert set(completion_by_state) == {"state-1", "state-2", "state-3"}
        for state_id, kwargs in completion_by_state.items():
            assert kwargs["status"] == NodeStateStatus.FAILED
            assert kwargs["error"].phase == "begin_node_state_failsink"
            assert kwargs["error"].exception_type == "AuditIntegrityError", state_id

    def test_failsink_mid_loop_system_error_cleans_up_remaining_states(self) -> None:
        """When the failsink completion loop raises AuditIntegrityError, remaining states close.

        Setup: 2 diverted tokens. complete_node_state succeeds for token 0's
        primary state, then raises AuditIntegrityError for token 0's failsink
        state. At that point:
        - Token 0: primary FAILED (completed), failsink OPEN (failed mid-loop)
        - Token 1: primary OPEN, failsink OPEN (not yet processed)
        The 3 remaining OPEN states must be closed as FAILED.
        """
        executor, execution, _data_flow = _make_executor()
        sink = _make_sink(
            diversions=(
                RowDiversion(row_index=0, reason="bad-0", row_data={"field": "v0"}),
                RowDiversion(row_index=1, reason="bad-1", row_data={"field": "v1"}),
            ),
        )
        failsink = _make_failsink()
        tokens = [_make_token("t0"), _make_token("t1")]

        # complete_node_state: let the first call succeed (token 0 primary → FAILED),
        # then raise on the second call (token 0 failsink → AuditIntegrityError).
        complete_count = [0]
        successful_completions: list[dict[str, Any]] = []

        def complete_with_error(**kwargs: Any) -> None:
            complete_count[0] += 1
            if complete_count[0] == 2:
                raise AuditIntegrityError("DB error completing failsink state")
            successful_completions.append(dict(kwargs))

        execution.complete_node_state.side_effect = complete_with_error

        with pytest.raises(AuditIntegrityError, match="DB error completing"):
            executor.write(
                sink=sink,
                tokens=tokens,  # type: ignore[arg-type]
                ctx=_make_context(),
                step_in_pipeline=5,
                sink_name="primary",
                pending_outcome=_default_pending(),
                failsink=failsink,
                failsink_name="csv_failsink",
                failsink_edge_id="edge-divert-1",
            )

        begin_calls = execution.begin_node_state.call_args_list
        assert [(call.kwargs["token_id"], call.kwargs["node_id"]) for call in begin_calls] == [
            ("t0", "node-primary"),
            ("t1", "node-primary"),
            ("t0", "node-failsink"),
            ("t1", "node-failsink"),
        ]

        attempted_completions = execution.complete_node_state.call_args_list
        assert [(call.kwargs["state_id"], call.kwargs["status"]) for call in attempted_completions] == [
            ("state-1", NodeStateStatus.FAILED),
            ("state-3", NodeStateStatus.COMPLETED),
            ("state-2", NodeStateStatus.FAILED),
            ("state-3", NodeStateStatus.FAILED),
            ("state-4", NodeStateStatus.FAILED),
        ]

        successful_by_state = _unique_completion_kwargs_by_state(successful_completions)
        assert set(successful_by_state) == {"state-1", "state-2", "state-3", "state-4"}
        assert successful_by_state["state-1"]["status"] == NodeStateStatus.FAILED
        assert successful_by_state["state-1"]["error"].phase == "write"
        assert successful_by_state["state-1"]["output_data"] == {
            "diverted_to": "csv_failsink",
            "reason": "bad-0",
        }
        for state_id in ("state-2", "state-3", "state-4"):
            kwargs = successful_by_state[state_id]
            assert kwargs["status"] == NodeStateStatus.FAILED
            assert kwargs["error"].phase == "failsink_audit_recording"
            assert kwargs["error"].exception_type == "AuditIntegrityError"


class TestDiversionIndexValidation:
    """Regression: SinkExecutor must reject out-of-range diversion indices."""

    def test_out_of_range_diversion_index_crashes(self) -> None:
        """row_index >= batch size is a plugin bug — crash before audit recording."""
        executor, execution, _data_flow = _make_executor()
        tokens = [_make_token("t1"), _make_token("t2")]
        # row_index=5 is out of range for a 2-token batch
        sink = _make_sink(
            diversions=(RowDiversion(row_index=5, reason="bad", row_data={"x": 1}),),
        )
        pending = _default_pending()

        with pytest.raises(PluginContractViolation, match=r"row_index=5.*batch has only 2 rows"):
            executor.write(
                sink,
                tokens,  # type: ignore[arg-type]
                _make_context(),
                step_in_pipeline=0,
                sink_name="out",
                pending_outcome=pending,
            )

        # Pre-opened states should be completed as FAILED (Phase 1 error path)
        assert execution.begin_node_state.call_count == 2
        failed_calls = [c for c in execution.complete_node_state.call_args_list if c.kwargs.get("status") == NodeStateStatus.FAILED]
        assert len(failed_calls) == 2


class TestUncoveredExceptArmCharacterization:
    """Characterize three error-cleanup arms that were untested before the
    SinkExecutor.write() decomposition (elspeth-f6a6ab0a46).

    These pin CURRENT behavior so the behavior-preserving extraction into phase
    helpers cannot silently break a cleanup path. Each targets an arm whose
    only prior protection was the (ineffective) TIER_1-guard count floor:

    * Test A: failsink begin_node_state GENERIC arm (non-TIER_1 error) — distinct
      from its TIER_1 sibling; completes states directly + sets divert_states_closed.
    * Test B: primary Phase-1 outer TIER_1 arm, flag False (best-effort cleanup).
    * Test C: failsink outer TIER_1 arm, divert_states_closed False (best-effort cleanup).

    RuntimeError is deliberately used for Test A because it is NOT in
    contract_errors.TIER_1_ERRORS (so it routes to the generic arm), whereas
    AuditIntegrityError/FrameworkBugError ARE in TIER_1_ERRORS (Tests B/C).
    """

    def test_failsink_begin_node_state_generic_error_cleans_up_open_states(self) -> None:
        """A NON-TIER_1 error from failsink begin_node_state hits the generic arm
        (sink.py 793-813), which completes the primary-divert anchors AND the
        partially-opened failsink state as FAILED with phase='begin_node_state_failsink',
        and sets divert_states_closed so the outer arm does not double-complete."""
        executor, execution, _data_flow = _make_executor()
        sink = _make_sink(
            diversions=(
                RowDiversion(row_index=0, reason="bad-0", row_data={"field": "v0"}),
                RowDiversion(row_index=1, reason="bad-1", row_data={"field": "v1"}),
            ),
        )
        failsink = _make_failsink()
        tokens = [_make_token("t0"), _make_token("t1")]

        # begin_node_state: calls 1-2 primary divert states OK, call 3 first
        # failsink state OK, call 4 (second failsink state) raises a non-TIER_1 error.
        call_count = [0]
        original_side_effect = execution.begin_node_state.side_effect

        def begin_state_with_error(**kwargs: Any) -> SimpleNamespace:
            call_count[0] += 1
            if call_count[0] == 4:
                raise RuntimeError("transient failsink begin failure")
            return original_side_effect(**kwargs)  # type: ignore[no-any-return]

        execution.begin_node_state.side_effect = begin_state_with_error

        with pytest.raises(RuntimeError, match="transient failsink begin failure"):
            executor.write(
                sink=sink,
                tokens=tokens,  # type: ignore[arg-type]
                ctx=_make_context(),
                step_in_pipeline=5,
                sink_name="primary",
                pending_outcome=_default_pending(),
                failsink=failsink,
                failsink_name="csv_failsink",
                failsink_edge_id="edge-divert-1",
            )

        # state-1/2 (primary divert anchors) + state-3 (1st failsink) closed by the generic arm.
        completion_by_state = _complete_node_state_kwargs_by_state(execution)
        assert set(completion_by_state) == {"state-1", "state-2", "state-3"}
        for state_id, kwargs in completion_by_state.items():
            assert kwargs["status"] == NodeStateStatus.FAILED
            assert kwargs["error"].phase == "begin_node_state_failsink"
            assert kwargs["error"].exception_type == "RuntimeError", state_id

    def test_primary_write_tier1_error_cleans_up_states_flag_false(self) -> None:
        """A TIER_1 error from primary sink.write() with NO boundary violation hits
        the outer TIER_1 arm (sink.py 599-602) with primary_states_closed_by_boundary_failure
        False, so best-effort cleanup closes every pre-opened primary state as FAILED."""
        executor, execution, _data_flow = _make_executor()
        sink = _make_sink()  # no diversions
        sink.write.side_effect = AuditIntegrityError("primary write audit failure")
        tokens = [_make_token("t0"), _make_token("t1")]

        with pytest.raises(AuditIntegrityError, match="primary write audit failure"):
            executor.write(
                sink=sink,
                tokens=tokens,  # type: ignore[arg-type]
                ctx=_make_context(),
                step_in_pipeline=5,
                sink_name="primary",
                pending_outcome=_default_pending(),
            )

        completion_by_state = _complete_node_state_kwargs_by_state(execution)
        assert set(completion_by_state) == {"state-1", "state-2"}
        for state_id, kwargs in completion_by_state.items():
            assert kwargs["status"] == NodeStateStatus.FAILED
            assert kwargs["error"].phase == "sink_write"
            assert kwargs["error"].exception_type == "AuditIntegrityError", state_id

    def test_failsink_write_tier1_error_cleans_up_states_flag_false(self) -> None:
        """A TIER_1 error from failsink.write() (failsink states already open, so
        divert_states_closed False) hits the outer TIER_1 arm (sink.py 913-916),
        best-effort closing the primary-divert anchor AND failsink state as FAILED."""
        executor, execution, _data_flow = _make_executor()
        diversions = (RowDiversion(row_index=0, reason="bad", row_data={"x": 1}),)
        sink = _make_sink(diversions=diversions, on_write_failure="csv_failsink")
        failsink = _make_failsink()
        failsink.write.side_effect = FrameworkBugError("failsink write framework bug")
        tokens = [_make_token("t0")]

        with pytest.raises(FrameworkBugError, match="failsink write framework bug"):
            executor.write(
                sink=sink,
                tokens=tokens,  # type: ignore[arg-type]
                ctx=_make_context(),
                step_in_pipeline=5,
                sink_name="primary",
                pending_outcome=_default_pending(),
                failsink=failsink,
                failsink_name="csv_failsink",
                failsink_edge_id="edge-failsink-1",
            )

        completion_by_state = _complete_node_state_kwargs_by_state(execution)
        assert set(completion_by_state) == {"state-1", "state-2"}
        for state_id, kwargs in completion_by_state.items():
            assert kwargs["status"] == NodeStateStatus.FAILED
            assert kwargs["error"].phase == "failsink_write"
            assert kwargs["error"].exception_type == "FrameworkBugError", state_id
