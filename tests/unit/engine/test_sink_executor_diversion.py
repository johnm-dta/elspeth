"""Focused unit contracts for effect-only sink diversion execution.

End-to-end primary, discard, and linked-failsink behavior is exercised against
the real Landscape repositories in ``test_sink_effect_recovery.py``.  This
module keeps the local legacy-refusal contract; the old write/flush
compatibility-path tests were removed with that production path.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from elspeth.contracts import PendingOutcome, PluginSchema
from elspeth.contracts.enums import TerminalOutcome, TerminalPath
from elspeth.contracts.errors import OrchestrationInvariantError
from elspeth.contracts.plugin_context import PluginContext
from elspeth.engine.executors.sink import SinkExecutor


class _PermissiveSchema(PluginSchema):
    """Accept arbitrary rows for the executor boundary test."""


class _ForbiddenLegacySink:
    name = "legacy"
    node_id = "node-legacy"
    input_schema = _PermissiveSchema
    declared_required_fields: frozenset[str] = frozenset()

    def __init__(self) -> None:
        self.config: dict[str, object] = {}
        self.publication_calls = 0

    def write(self, rows: object, ctx: object) -> object:
        del rows, ctx
        self.publication_calls += 1
        raise AssertionError("legacy write must not be called")

    def flush(self) -> None:
        self.publication_calls += 1
        raise AssertionError("legacy flush must not be called")


class _Row:
    contract = SimpleNamespace(merge_for_batch=lambda other: other)

    def to_dict(self) -> dict[str, object]:
        return {"value": 1}


def _executor() -> SinkExecutor:
    return SinkExecutor(SimpleNamespace(), SimpleNamespace(), SimpleNamespace(), "run-1")  # type: ignore[arg-type]


def test_default_effect_lease_owner_is_unique_per_executor() -> None:
    first = _executor()
    second = _executor()

    assert first._worker_id.startswith("sink-effects:run-1:")
    assert second._worker_id.startswith("sink-effects:run-1:")
    assert first._worker_id != second._worker_id


def test_non_empty_legacy_execution_refuses_before_publication() -> None:
    sink = _ForbiddenLegacySink()
    token = SimpleNamespace(row_data=_Row())

    with pytest.raises(OrchestrationInvariantError, match="legacy publication is forbidden"):
        _executor().write(
            sink,  # type: ignore[arg-type]
            [token],  # type: ignore[list-item]
            PluginContext(run_id="run-1", config={}),
            1,
            sink_name="legacy",
            pending_outcome=PendingOutcome(outcome=TerminalOutcome.SUCCESS, path=TerminalPath.DEFAULT_FLOW),
        )

    assert sink.publication_calls == 0


def test_empty_batch_is_a_noop_without_effect_mode() -> None:
    sink = _ForbiddenLegacySink()

    artifact, counts = _executor().write(
        sink,  # type: ignore[arg-type]
        [],
        PluginContext(run_id="run-1", config={}),
        1,
        sink_name="legacy",
        pending_outcome=PendingOutcome(outcome=TerminalOutcome.SUCCESS, path=TerminalPath.DEFAULT_FLOW),
    )

    assert artifact is None
    assert counts.total == 0
    assert sink.publication_calls == 0
