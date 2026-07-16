"""ADR-019 Phase 2 producer-pair regression tests."""

from __future__ import annotations

from elspeth.contracts.enums import TerminalOutcome, TerminalPath
from elspeth.contracts.plugin_context import PluginContext
from elspeth.engine.processor import RowProcessor
from elspeth.testing import make_token_info
from tests.unit.engine.test_processor import _make_factory, _make_processor


def test_processor_token_completed_telemetry_carries_two_axis_pair() -> None:
    processor = object.__new__(RowProcessor)
    processor._telemetry_manager = object()
    processor._run_id = "run-adr019-phase2"
    events: list[object] = []
    processor._emit_telemetry = events.append
    token = make_token_info(data={"value": 1})

    processor._emit_token_completed(
        token,
        outcome=TerminalOutcome.FAILURE,
        path=TerminalPath.UNROUTED,
    )

    assert len(events) == 1
    event = events[0]
    assert event.outcome == TerminalOutcome.FAILURE
    assert event.path == TerminalPath.UNROUTED


def test_processor_terminal_work_item_returns_default_flow_pair() -> None:
    _db, factory = _make_factory()
    processor = _make_processor(factory, source_on_success="source_sink")
    token = make_token_info(data={"value": 1})

    result, _child_items = processor._process_single_token(
        token=token,
        ctx=PluginContext(run_id="run-adr019-phase2", config={}),
        current_node_id=None,
        on_success_sink="terminal_sink",
    )

    assert result is not None
    assert not isinstance(result, tuple)
    assert result.outcome == TerminalOutcome.SUCCESS
    assert result.path == TerminalPath.DEFAULT_FLOW
    assert result.sink_name == "terminal_sink"
