# tests/unit/engine/test_token_traversal_characterization.py
"""Characterization net for the RowProcessor DAG token-traversal state machine.

Pins the CURRENT observable behavior of ``_process_single_token`` and its
``_handle_*`` handler family at the RowProcessor surface BEFORE the
TokenTraversalEngine extraction (filigree elspeth-c49f33d6e4, component 4).
Every test here must pass unchanged against both the pre-move and post-move
trees.

The moved cluster is a cohesive unit: ``_process_single_token`` is the
orchestration loop; ``_handle_transform_node`` /
``_handle_transform_error_status`` / ``_handle_gate_node`` /
``_handle_gate_fork`` / ``_handle_terminal_token`` /
``_validate_coalesce_ordering`` are its private handlers. The extraction moves
all seven (plus the ``_Transform*`` / ``_Gate*`` discriminated-union outcome
types) into ``engine/token_traversal.py`` behind a call-time host protocol,
leaving thin delegates on RowProcessor for the three names referenced from
outside the cluster: ``_process_single_token`` (the SchedulerDrainHost seam
plus 10+ tests patch it), ``_handle_transform_node`` and
``_handle_transform_error_status`` (test_processor.py calls them directly).

The observables asserted here are the *return contract* — the
``_TransformContinue`` / ``_TransformTerminal`` / ``_GateContinue`` /
``_GateTerminal`` outcome objects, the ``(RowResult | tuple | None,
child_items)`` pair from the orchestration loop, and the invariant raises —
NOT processor internals, so the net survives the extraction. The union types
are imported from ``elspeth.engine.processor`` on purpose: the post-move tree
must keep re-exporting them from there, so this import doubles as a
re-export smoke test.

Each branch of the moved region is covered exactly once; deep behavioral
coverage (audit recording, coalesce sibling propagation, deaggregation child
queueing) lives in test_processor.py and is exercised by the full-suite gate.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from elspeth.contracts import RowResult, TokenInfo, TransformResult
from elspeth.contracts.enums import TerminalOutcome, TerminalPath
from elspeth.contracts.errors import MaxRetriesExceeded, OrchestrationInvariantError
from elspeth.contracts.results import GateResult
from elspeth.contracts.routing import RoutingAction
from elspeth.contracts.types import BranchName, CoalesceName, NodeID
from elspeth.core.config import GateSettings
from elspeth.engine.executors import GateOutcome
from elspeth.engine.processor import (
    _TransformContinue,
    _TransformTerminal,
)
from elspeth.testing import make_row, make_token_info
from tests.fixtures.factories import make_context
from tests.unit.engine.test_processor import (
    _make_contract,
    _make_factory,
    _make_mock_transform,
    _make_processor,
    _persist_token_for_scheduler,
)

# =============================================================================
# _process_single_token: orchestration-loop arms
# =============================================================================


class TestProcessSingleTokenOrchestration:
    """Characterize the traversal loop's branch structure and invariant raises."""

    def test_null_current_node_without_sink_context_raises(self) -> None:
        """current_node_id=None with no inherited/branch sink is an invariant violation."""
        _db, factory = _make_factory()
        ctx = make_context(landscape=factory.plugin_audit_writer())
        processor = _make_processor(factory, source_on_success="source_sink")
        token = make_token_info(data={"value": 1})

        with pytest.raises(OrchestrationInvariantError, match="current_node_id=None"):
            processor._process_single_token(token=token, ctx=ctx, current_node_id=None)

    def test_null_current_node_with_inherited_sink_completes_default_flow(self) -> None:
        """Explicit on_success_sink lets a nodeless terminal token complete."""
        _db, factory = _make_factory()
        ctx = make_context(landscape=factory.plugin_audit_writer())
        processor = _make_processor(factory, source_on_success="source_sink")
        token = make_token_info(data={"value": 1})

        result, child_items = processor._process_single_token(
            token=token,
            ctx=ctx,
            current_node_id=None,
            on_success_sink="terminal_sink",
        )

        assert isinstance(result, RowResult)
        assert (result.outcome, result.path) == (TerminalOutcome.SUCCESS, TerminalPath.DEFAULT_FLOW)
        assert result.sink_name == "terminal_sink"
        assert child_items == []

    def test_null_current_node_with_branch_sink_completes_via_branch_map(self) -> None:
        """branch_to_sink takes precedence over inherited sink for terminal routing."""
        _db, factory = _make_factory()
        ctx = make_context(landscape=factory.plugin_audit_writer())
        processor = _make_processor(
            factory,
            source_on_success="source_sink",
            branch_to_sink={BranchName("path_a"): "branch_sink"},
        )
        token = TokenInfo(
            row_id="row-1",
            token_id="tok-1",
            row_data=make_row({"value": 1}),
            branch_name="path_a",
        )

        result, _child_items = processor._process_single_token(
            token=token,
            ctx=ctx,
            current_node_id=None,
            on_success_sink="ignored_sink",
        )

        assert isinstance(result, RowResult)
        assert (result.outcome, result.path) == (TerminalOutcome.SUCCESS, TerminalPath.DEFAULT_FLOW)
        assert result.sink_name == "branch_sink"

    def test_structural_node_is_traversed_but_not_executed(self) -> None:
        """A node with no plugin (structural) advances to the next node without executing."""
        _db, factory = _make_factory()
        ctx = make_context(landscape=factory.plugin_audit_writer())
        source_node = NodeID("source-0")
        structural = NodeID("structural-1")
        processor = _make_processor(
            factory,
            source_on_success="terminal_sink",
            node_step_map={source_node: 0, structural: 1},
            node_to_next={source_node: structural, structural: None},
            node_to_plugin={},  # structural has no plugin
            structural_node_ids=frozenset({source_node, structural}),
        )
        token = make_token_info(data={"value": 1})

        result, child_items = processor._process_single_token(
            token=token,
            ctx=ctx,
            current_node_id=structural,
        )

        assert isinstance(result, RowResult)
        assert (result.outcome, result.path) == (TerminalOutcome.SUCCESS, TerminalPath.DEFAULT_FLOW)
        assert result.sink_name == "terminal_sink"
        assert child_items == []

    def test_inner_cycle_guard_raises_on_node_to_next_loop(self) -> None:
        """A self-referential node_to_next trips the inner-iteration cycle guard."""
        _db, factory = _make_factory()
        ctx = make_context(landscape=factory.plugin_audit_writer())
        source_node = NodeID("source-0")
        looping = NodeID("loop-1")
        processor = _make_processor(
            factory,
            node_step_map={source_node: 0, looping: 1},
            node_to_next={source_node: looping, looping: looping},  # cycle
            node_to_plugin={},  # structural → pure traversal, never executes
            structural_node_ids=frozenset({source_node, looping}),
        )
        token = make_token_info(data={"value": 1})

        with pytest.raises(OrchestrationInvariantError, match="Inner traversal exceeded"):
            processor._process_single_token(token=token, ctx=ctx, current_node_id=looping)

    def test_unknown_plugin_type_raises_type_error(self) -> None:
        """A node plugin that is neither TransformProtocol nor GateSettings is rejected."""
        _db, factory = _make_factory()
        ctx = make_context(landscape=factory.plugin_audit_writer())
        source_node = NodeID("source-0")
        weird = NodeID("weird-1")
        processor = _make_processor(
            factory,
            node_step_map={source_node: 0, weird: 1},
            node_to_next={source_node: weird, weird: None},
            node_to_plugin={weird: object()},  # not a transform, not a gate
        )
        token = make_token_info(data={"value": 1})

        with pytest.raises(TypeError, match="Unknown transform type"):
            processor._process_single_token(token=token, ctx=ctx, current_node_id=weird)

    def test_gate_route_to_sink_returns_gate_routed_terminal(self) -> None:
        """A gate that routes to a sink terminates the token with GATE_ROUTED."""
        _db, factory = _make_factory()
        ctx = make_context(landscape=factory.plugin_audit_writer())
        source_node = NodeID("source-0")
        gate_node = NodeID("gate-1")
        gate_config = GateSettings(name="router", input="default", condition="True", routes={"true": "error_sink", "false": "default"})
        processor = _make_processor(
            factory,
            node_step_map={source_node: 0, gate_node: 1},
            node_to_next={source_node: gate_node, gate_node: None},
            node_to_plugin={gate_node: gate_config},
        )
        token = make_token_info(row_id="row-1", token_id="tok-1", data={"value": 1})
        sink_outcome = GateOutcome(
            result=GateResult(row={"value": 1}, action=RoutingAction.route("true"), contract=_make_contract()),
            updated_token=token,
            sink_name="error_sink",
        )

        def _route(gate_config, node_id, token, ctx, token_manager=None):
            return sink_outcome

        processor._gate_executor.execute_config_gate = _route  # type: ignore[method-assign]
        result, _child_items = processor._process_single_token(token=token, ctx=ctx, current_node_id=gate_node)

        assert isinstance(result, RowResult)
        assert (result.outcome, result.path) == (TerminalOutcome.SUCCESS, TerminalPath.GATE_ROUTED)
        assert result.sink_name == "error_sink"

    def test_gate_continue_advances_to_terminal(self) -> None:
        """A CONTINUE gate outcome advances the token to the next node then terminates."""
        _db, factory = _make_factory()
        ctx = make_context(landscape=factory.plugin_audit_writer())
        source_node = NodeID("source-0")
        gate_node = NodeID("gate-1")
        gate_config = GateSettings(name="passgate", input="default", condition="True", routes={"true": "default", "false": "default"})
        processor = _make_processor(
            factory,
            source_on_success="terminal_sink",
            node_step_map={source_node: 0, gate_node: 1},
            node_to_next={source_node: gate_node, gate_node: None},
            node_to_plugin={gate_node: gate_config},
        )
        token = make_token_info(row_id="row-1", token_id="tok-1", data={"value": 1})
        continue_outcome = GateOutcome(
            result=GateResult(row={"value": 1}, action=RoutingAction.continue_(), contract=_make_contract()),
            updated_token=token,
        )

        def _continue(gate_config, node_id, token, ctx, token_manager=None):
            return continue_outcome

        processor._gate_executor.execute_config_gate = _continue  # type: ignore[method-assign]
        result, _child_items = processor._process_single_token(token=token, ctx=ctx, current_node_id=gate_node)

        assert isinstance(result, RowResult)
        assert (result.outcome, result.path) == (TerminalOutcome.SUCCESS, TerminalPath.DEFAULT_FLOW)
        assert result.sink_name == "terminal_sink"

    def test_gate_jump_to_node_absent_from_step_map_raises(self) -> None:
        """A gate jump to a node not in the DAG step map is an invariant violation."""
        _db, factory = _make_factory()
        ctx = make_context(landscape=factory.plugin_audit_writer())
        source_node = NodeID("source-0")
        gate_node = NodeID("gate-1")
        gate_config = GateSettings(name="jumper", input="default", condition="True", routes={"true": "default", "false": "default"})
        processor = _make_processor(
            factory,
            node_step_map={source_node: 0, gate_node: 1},
            node_to_next={source_node: gate_node, gate_node: None},
            node_to_plugin={gate_node: gate_config},
        )
        token = make_token_info(row_id="row-1", token_id="tok-1", data={"value": 1})
        jump_outcome = GateOutcome(
            result=GateResult(row={"value": 1}, action=RoutingAction.route("ghost"), contract=_make_contract()),
            updated_token=token,
            next_node_id=NodeID("ghost-node"),
        )

        def _jump(gate_config, node_id, token, ctx, token_manager=None):
            return jump_outcome

        processor._gate_executor.execute_config_gate = _jump  # type: ignore[method-assign]
        with pytest.raises(OrchestrationInvariantError, match="not in the DAG step map"):
            processor._process_single_token(token=token, ctx=ctx, current_node_id=gate_node)


# =============================================================================
# _handle_transform_node: transform arms (delegate kept on RowProcessor)
# =============================================================================


class TestHandleTransformNode:
    """Characterize the transform handler's outcome contract."""

    def test_single_row_success_returns_continue_with_on_success_sink(self) -> None:
        """A single-row success advances (Continue) and adopts transform.on_success as the sink."""
        _db, factory = _make_factory()
        ctx = make_context(landscape=factory.plugin_audit_writer())
        processor = _make_processor(factory)
        transform = _make_mock_transform(node_id="t-1", name="mapper", on_success="next_sink")
        token = make_token_info(row_id="row-1", token_id="tok-1", data={"value": 1})
        success = TransformResult.success(make_row({"value": 2}), success_reason={"action": "mapped"})

        def _exec(transform, token, ctx, attempt_offset=0):
            return success, token, None

        processor._execute_transform_with_retry = _exec  # type: ignore[method-assign]
        outcome = processor._handle_transform_node(
            transform=transform,
            current_token=token,
            ctx=ctx,
            node_id=NodeID("t-1"),
            child_items=[],
            coalesce_node_id=None,
            coalesce_name=None,
            current_on_success_sink="default",
        )

        assert isinstance(outcome, _TransformContinue)
        assert outcome.updated_sink == "next_sink"

    def test_multi_row_without_creates_tokens_raises(self) -> None:
        """A multi-row emission from a transform declaring creates_tokens=False is a config bug."""
        _db, factory = _make_factory()
        ctx = make_context(landscape=factory.plugin_audit_writer())
        processor = _make_processor(factory)
        transform = _make_mock_transform(node_id="t-1", name="splitter", creates_tokens=False)
        token = make_token_info(row_id="row-1", token_id="tok-1", data={"value": 1})
        contract = _make_contract()
        multi = TransformResult.success_multi(
            [make_row({"value": 2}, contract=contract), make_row({"value": 3}, contract=contract)],
            success_reason={"action": "split"},
        )

        def _exec(transform, token, ctx, attempt_offset=0):
            return multi, token, None

        processor._execute_transform_with_retry = _exec  # type: ignore[method-assign]
        with pytest.raises(RuntimeError, match="creates_tokens=False"):
            processor._handle_transform_node(
                transform=transform,
                current_token=token,
                ctx=ctx,
                node_id=NodeID("t-1"),
                child_items=[],
                coalesce_node_id=None,
                coalesce_name=None,
                current_on_success_sink="default",
            )

    def test_multi_row_empty_returns_filter_dropped_terminal(self) -> None:
        """An explicit zero-row emission terminates the token as FILTER_DROPPED."""
        _db, factory = _make_factory()
        ctx = make_context(landscape=factory.plugin_audit_writer())
        processor = _make_processor(factory)
        transform = _make_mock_transform(node_id="t-1", name="filter")
        token = make_token_info(row_id="row-1", token_id="tok-1", data={"value": 1})
        empty = TransformResult.success_empty(success_reason={"action": "filtered"})
        recorded: list[dict[str, object]] = []

        def _exec(transform, token, ctx, attempt_offset=0):
            return empty, token, None

        processor._execute_transform_with_retry = _exec  # type: ignore[method-assign]
        processor._data_flow.record_token_outcome = lambda **kwargs: recorded.append(kwargs)  # type: ignore[method-assign, assignment]
        outcome = processor._handle_transform_node(
            transform=transform,
            current_token=token,
            ctx=ctx,
            node_id=NodeID("t-1"),
            child_items=[],
            coalesce_node_id=None,
            coalesce_name=None,
            current_on_success_sink="default",
        )

        assert isinstance(outcome, _TransformTerminal)
        assert isinstance(outcome.result, RowResult)
        assert (outcome.result.outcome, outcome.result.path) == (TerminalOutcome.SUCCESS, TerminalPath.FILTER_DROPPED)

    def test_max_retries_exceeded_returns_unrouted_failure(self) -> None:
        """Exhausted retries terminate the token as an UNROUTED FAILURE."""
        _db, factory = _make_factory()
        ctx = make_context(landscape=factory.plugin_audit_writer())
        processor = _make_processor(factory)
        transform = _make_mock_transform(node_id="t-1", name="flaky")
        token = make_token_info(row_id="row-1", token_id="tok-1", data={"value": 1})
        recorded: list[dict[str, object]] = []

        def _exec(transform, token, ctx, attempt_offset=0):
            raise MaxRetriesExceeded(3, ValueError("boom"))

        processor._execute_transform_with_retry = _exec  # type: ignore[method-assign]
        processor._data_flow.record_token_outcome = lambda **kwargs: recorded.append(kwargs)  # type: ignore[method-assign, assignment]
        outcome = processor._handle_transform_node(
            transform=transform,
            current_token=token,
            ctx=ctx,
            node_id=NodeID("t-1"),
            child_items=[],
            coalesce_node_id=None,
            coalesce_name=None,
            current_on_success_sink="default",
        )

        assert isinstance(outcome, _TransformTerminal)
        assert isinstance(outcome.result, RowResult)
        assert (outcome.result.outcome, outcome.result.path) == (TerminalOutcome.FAILURE, TerminalPath.UNROUTED)
        assert len(recorded) == 1

    def test_retry_exhaustion_keeps_branch_loss_audit_reason_bounded(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Secret-bearing retry failures must not reach branch-loss audit text."""
        from elspeth.core.landscape.scheduler.branch_losses import record_coalesce_branch_loss

        db, factory = _make_factory()
        coalesce_name = CoalesceName("merge")
        processor = _make_processor(
            factory,
            coalesce_node_ids={coalesce_name: NodeID("coalesce::merge")},
            branch_to_coalesce={BranchName("path_a"): coalesce_name},
        )
        transform = _make_mock_transform(node_id="t-1", name="flaky")
        token = TokenInfo(
            row_id="row-1",
            token_id="tok-1",
            row_data=make_row({"value": 1}),
            branch_name="path_a",
        )
        _persist_token_for_scheduler(factory, token)
        ctx = make_context(landscape=factory.plugin_audit_writer())
        raw_secret = "https://blob.example/path?sig=ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"

        def _exec(transform, token, ctx, attempt_offset=0):
            raise MaxRetriesExceeded(3, ConnectionError(f"provider failed: {raw_secret}"))

        processor._execute_transform_with_retry = _exec  # type: ignore[method-assign]
        outcome = processor._handle_transform_node(
            transform=transform,
            current_token=token,
            ctx=ctx,
            node_id=NodeID("t-1"),
            child_items=[],
            coalesce_node_id=NodeID("coalesce::merge"),
            coalesce_name=coalesce_name,
            current_on_success_sink="default",
        )

        assert isinstance(outcome, _TransformTerminal)
        assert isinstance(outcome.result, RowResult)
        assert outcome.result.error is not None
        branch_loss = processor._pending_branch_losses.pop()
        with db.write_connection() as conn:
            assert record_coalesce_branch_loss(
                conn,
                run_id="test-run",
                coalesce_name=branch_loss.coalesce_name,
                row_id=branch_loss.row_id,
                branch_name=branch_loss.branch_name,
                token_id=branch_loss.token_id,
                reason=branch_loss.reason,
                recorded_by=branch_loss.recorded_by,
                now=datetime.now(UTC),
            )
        with (
            caplog.at_level(logging.WARNING, logger="elspeth.core.landscape.scheduler.branch_losses"),
            db.write_connection() as conn,
        ):
            assert not record_coalesce_branch_loss(
                conn,
                run_id="test-run",
                coalesce_name=branch_loss.coalesce_name,
                row_id=branch_loss.row_id,
                branch_name=branch_loss.branch_name,
                token_id=branch_loss.token_id,
                reason="max_retries_exceeded_replay",
                recorded_by=branch_loss.recorded_by,
                now=datetime.now(UTC),
            )

        [durable_loss] = factory.scheduler.list_coalesce_branch_losses(run_id="test-run")
        assert outcome.result.error.message == "<redacted-secret>"
        assert outcome.result.error.last_error == "<redacted-secret>"
        assert durable_loss.reason == "max_retries_exceeded"
        assert raw_secret not in caplog.text


# =============================================================================
# _handle_transform_error_status: quarantine vs route (delegate kept)
# =============================================================================


class TestHandleTransformErrorStatus:
    """Characterize the error-status handler's discard/route/invariant arms."""

    def test_discard_returns_quarantined_terminal(self) -> None:
        """error_sink='discard' quarantines the token (FAILURE / QUARANTINED_AT_SOURCE)."""
        _db, factory = _make_factory()
        processor = _make_processor(factory)
        token = make_token_info(row_id="row-1", token_id="tok-1", data={"value": 1})
        transform_result = TransformResult.error(reason={"reason": "invalid_input"})
        recorded: list[dict[str, object]] = []
        processor._data_flow.record_token_outcome = lambda **kwargs: recorded.append(kwargs)  # type: ignore[method-assign, assignment]

        outcome = processor._handle_transform_error_status(
            transform_result=transform_result,
            current_token=token,
            error_sink="discard",
            child_items=[],
        )

        assert isinstance(outcome, _TransformTerminal)
        assert isinstance(outcome.result, RowResult)
        assert (outcome.result.outcome, outcome.result.path) == (
            TerminalOutcome.FAILURE,
            TerminalPath.QUARANTINED_AT_SOURCE,
        )
        assert len(recorded) == 1

    def test_route_with_reason_returns_on_error_routed_terminal(self) -> None:
        """A named error_sink with a reason routes (ON_ERROR_ROUTED) and defers recording to the sink."""
        _db, factory = _make_factory()
        processor = _make_processor(factory)
        token = make_token_info(row_id="row-1", token_id="tok-1", data={"value": 1})
        transform_result = TransformResult.error(reason={"reason": "validation_failed"})

        outcome = processor._handle_transform_error_status(
            transform_result=transform_result,
            current_token=token,
            error_sink="error_sink",
            child_items=[],
        )

        assert isinstance(outcome, _TransformTerminal)
        assert isinstance(outcome.result, RowResult)
        assert (outcome.result.outcome, outcome.result.path) == (
            TerminalOutcome.FAILURE,
            TerminalPath.ON_ERROR_ROUTED,
        )
        assert outcome.result.sink_name == "error_sink"
        assert outcome.result.error is not None

    def test_route_without_reason_refuses_to_fabricate_audit_data(self) -> None:
        """A routed error with a falsy reason is refused rather than fabricating an error hash."""
        _db, factory = _make_factory()
        processor = _make_processor(factory)
        token = make_token_info(row_id="row-1", token_id="tok-1", data={"value": 1})
        transform_result = SimpleNamespace(reason=None)

        with pytest.raises(OrchestrationInvariantError, match="ROUTED_ON_ERROR requires transform_result"):
            processor._handle_transform_error_status(
                transform_result=transform_result,  # type: ignore[arg-type]
                current_token=token,
                error_sink="error_sink",
                child_items=[],
            )
