# tests/unit/engine/orchestrator/test_quarantine_router.py
"""Unit tests for QuarantineRouter — the source-quarantine routing seam.

Extracted from ``SourceIterationDriver.handle_quarantine_row``
(elspeth-27d7bfc14b). These drive the collaborator in isolation: the reachable
validation-error branches, the full happy-path collaboration sequence (token ->
FAILED node_state -> DIVERT routing_event -> RowCreated telemetry ->
PendingOutcome), and the plugin-error length bound. End-to-end audit behaviour
stays covered by tests/integration/pipeline/orchestrator/test_quarantine_routing.py.

Mock discipline: the Landscape recorder and ceremony are ``spec``-bound mocks
(the collaboration is asserted through them); the source/processor/ctx/config
inputs are plain ``SimpleNamespace`` fakes.
"""

from __future__ import annotations

from types import MappingProxyType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

from elspeth.contracts import PendingOutcome, SourceRow
from elspeth.contracts.enums import NodeStateStatus, RoutingMode, TerminalOutcome, TerminalPath
from elspeth.contracts.events import RowCreated
from elspeth.contracts.types import NodeID
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.engine.orchestrator.ceremony import RunCeremony
from elspeth.engine.orchestrator.quarantine_router import QUARANTINE_ERROR_MAX_CHARS, QuarantineRouter
from elspeth.engine.orchestrator.run_state import LoopContext
from elspeth.engine.orchestrator.types import ExecutionCounters, RouteValidationError

SOURCE_ID = NodeID("source-node")


def _make_source(*, name: str = "quarantine_source", on_validation_failure: str = "quarantine") -> SimpleNamespace:
    return SimpleNamespace(name=name, _on_validation_failure=on_validation_failure)


def _make_loop_ctx(*, sinks: tuple[str, ...] = ("quarantine",), validation_error_id: str | None = None) -> LoopContext:
    token = SimpleNamespace(token_id="tok-1", row_id="row-1")
    processor = SimpleNamespace(
        token_manager=SimpleNamespace(create_quarantine_token=lambda **kwargs: token),
        coordination_token=object(),
    )
    ctx = SimpleNamespace(pop_pending_quarantine_validation_error_id=lambda row: validation_error_id)
    return LoopContext(
        counters=ExecutionCounters(),
        pending_tokens={name: [] for name in sinks},
        processor=processor,
        ctx=ctx,
        config=SimpleNamespace(sinks={name: object() for name in sinks}),
        agg_transform_lookup=MappingProxyType({}),
        coalesce_executor=None,
        coalesce_node_map=MappingProxyType({}),
    )


def _make_factory() -> MagicMock:
    factory = MagicMock(spec=RecorderFactory)
    factory.execution.begin_node_state.return_value = SimpleNamespace(state_id="state-1")
    return factory


def _route(
    router: QuarantineRouter,
    factory: MagicMock,
    loop_ctx: LoopContext,
    source_item: SourceRow,
    *,
    source: SimpleNamespace,
    edge_map: dict[tuple[NodeID, str], str] | None = None,
) -> None:
    router.route(
        factory,
        "run-1",
        SOURCE_ID,
        source_item,
        0,
        source_item.source_row_index,
        0,
        edge_map if edge_map is not None else {(SOURCE_ID, "__quarantine__"): "edge-1"},
        loop_ctx,
        active_source=source,
    )


class TestQuarantineRouteValidation:
    """The reachable plugin-bug branches raise RouteValidationError."""

    def test_missing_destination_raises(self) -> None:
        router = QuarantineRouter(ceremony=MagicMock(spec=RunCeremony))
        # Empty-string destination passes SourceRow.__post_init__ (not None) but
        # is falsy at the router — the "plugin forgot the destination" case.
        item = SourceRow(
            row={"bad": "data"},
            is_quarantined=True,
            quarantine_error="validation failed",
            quarantine_destination="",
            source_row_index=0,
        )
        with pytest.raises(RouteValidationError, match="missing quarantine_destination"):
            _route(router, _make_factory(), _make_loop_ctx(), item, source=_make_source())

    def test_invalid_destination_raises(self) -> None:
        router = QuarantineRouter(ceremony=MagicMock(spec=RunCeremony))
        item = SourceRow.quarantined(
            row={"bad": "data"},
            error="validation failed",
            destination="nonexistent_sink",
            source_row_index=0,
        )
        with pytest.raises(RouteValidationError, match="invalid quarantine_destination='nonexistent_sink'"):
            _route(router, _make_factory(), _make_loop_ctx(), item, source=_make_source(on_validation_failure="nonexistent_sink"))

    def test_missing_quarantine_edge_raises(self) -> None:
        from elspeth.contracts.errors import OrchestrationInvariantError

        router = QuarantineRouter(ceremony=MagicMock(spec=RunCeremony))
        item = SourceRow.quarantined(row={"a": 1}, error="bad", destination="quarantine", source_row_index=0)
        with pytest.raises(OrchestrationInvariantError, match="no __quarantine__"):
            _route(router, _make_factory(), _make_loop_ctx(), item, source=_make_source(), edge_map={})


class TestQuarantineHappyPath:
    """A valid quarantined row produces the full audit collaboration."""

    def test_full_collaboration_sequence(self) -> None:
        ceremony = MagicMock(spec=RunCeremony)
        router = QuarantineRouter(ceremony=ceremony)
        loop_ctx = _make_loop_ctx()
        factory = _make_factory()
        item = SourceRow.quarantined(row={"amount": 100}, error="bad value", destination="quarantine", source_row_index=3)

        _route(router, factory, loop_ctx, item, source=_make_source())

        # Source node_state opened quarantined and completed FAILED.
        begin_kwargs = factory.execution.begin_node_state.call_args.kwargs
        assert begin_kwargs["quarantined"] is True
        assert begin_kwargs["node_id"] == SOURCE_ID
        assert begin_kwargs["step_index"] == 0
        assert factory.execution.complete_node_state.call_args.kwargs["status"] is NodeStateStatus.FAILED
        # DIVERT routing_event recorded on the __quarantine__ edge.
        routing_kwargs = factory.execution.record_routing_event.call_args.kwargs
        assert routing_kwargs["mode"] is RoutingMode.DIVERT
        assert routing_kwargs["edge_id"] == "edge-1"
        # RowCreated telemetry emitted after Landscape recording.
        (event,), _ = ceremony.emit_telemetry.call_args
        assert isinstance(event, RowCreated)
        # The created token reaches the destination bucket with a deferred
        # PendingOutcome (proves the token was created and routed).
        pending = loop_ctx.pending_tokens["quarantine"]
        assert len(pending) == 1
        token, outcome = pending[0]
        assert token.token_id == "tok-1"
        assert isinstance(outcome, PendingOutcome)
        assert outcome.outcome is TerminalOutcome.FAILURE
        assert outcome.path is TerminalPath.QUARANTINED_AT_SOURCE
        # Both counters bumped (quarantine is a FAILURE lifecycle subset).
        assert loop_ctx.counters.rows_quarantined == 1
        assert loop_ctx.counters.rows_failed == 1

    def test_overlong_error_is_bounded_on_audit_surfaces(self) -> None:
        router = QuarantineRouter(ceremony=MagicMock(spec=RunCeremony))
        loop_ctx = _make_loop_ctx()
        factory = _make_factory()
        long_error = "x" * (QUARANTINE_ERROR_MAX_CHARS + 5000)
        item = SourceRow.quarantined(row={"a": 1}, error=long_error, destination="quarantine", source_row_index=0)

        _route(router, factory, loop_ctx, item, source=_make_source())

        node_error = factory.execution.complete_node_state.call_args.kwargs["error"]
        routing_reason = factory.execution.record_routing_event.call_args.kwargs["reason"]
        # Same bounded text feeds node_state error and DIVERT reason.
        assert len(node_error.exception) <= QUARANTINE_ERROR_MAX_CHARS + 200
        assert node_error.exception.endswith("chars]")
        # SourceQuarantineReason is a TypedDict, so index the bounded text.
        assert routing_reason["quarantine_error"] == node_error.exception
