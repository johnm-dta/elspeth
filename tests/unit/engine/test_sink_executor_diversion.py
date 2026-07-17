"""Focused unit contracts for effect-only sink diversion execution.

End-to-end primary, discard, and linked-failsink behavior is exercised against
the real Landscape repositories in ``test_sink_effect_recovery.py``.  This
module keeps the local legacy-refusal contract plus the cheap effect-path
precondition guards that need no Landscape database; the old write/flush
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


@pytest.mark.parametrize(
    ("pending_outcome", "match"),
    (
        pytest.param(
            None,
            "received pending_outcome=None",
            id="pending-outcome-none",
        ),
        pytest.param(
            PendingOutcome(outcome=TerminalOutcome.SUCCESS, path=TerminalPath.DEFAULT_FLOW),
            "effect state recovery requires the owning RecorderFactory",
            id="missing-recorder-factory",
        ),
    ),
)
def test_effect_write_precondition_guards_fire_before_publication(
    pending_outcome: PendingOutcome | None,
    match: str,
) -> None:
    """Cheap effect-path precondition guards refuse before any sink I/O."""
    sink = _ForbiddenLegacySink()
    token = SimpleNamespace(row_data=_Row())

    with pytest.raises(OrchestrationInvariantError, match=match):
        _executor().write(
            sink,  # type: ignore[arg-type]
            [token],  # type: ignore[list-item]
            PluginContext(run_id="run-1", config={}),
            1,
            sink_name="legacy",
            pending_outcome=pending_outcome,
            effect_mode="write",
        )

    assert sink.publication_calls == 0


@pytest.mark.parametrize(
    ("factory", "pending_outcome", "match"),
    (
        pytest.param(
            None,
            PendingOutcome(outcome=TerminalOutcome.SUCCESS, path=TerminalPath.DEFAULT_FLOW),
            "effect-capable sink execution requires the owning RecorderFactory",
            id="missing-factory",
        ),
        pytest.param(
            SimpleNamespace(),
            PendingOutcome(outcome=None, path=TerminalPath.BUFFERED),
            "buffered outcomes cannot cross the sink-effect publication boundary",
            id="buffered-outcome",
        ),
    ),
)
def test_write_primary_effect_ownership_and_buffered_outcome_guards(
    factory: object,
    pending_outcome: PendingOutcome,
    match: str,
) -> None:
    """Direct-call guards on the primary publication boundary.

    The public ``write()`` path checks factory ownership earlier (effect state
    recovery), so these commit-boundary guards are pinned by direct call; each
    fires before any argument beyond the executor's own state is touched.
    """
    executor = SinkExecutor(
        SimpleNamespace(),  # type: ignore[arg-type]
        SimpleNamespace(),  # type: ignore[arg-type]
        SimpleNamespace(),  # type: ignore[arg-type]
        "run-1",
        factory=factory,  # type: ignore[arg-type]
    )

    with pytest.raises(OrchestrationInvariantError, match=match):
        executor._write_primary_effect(
            sink=_ForbiddenLegacySink(),  # type: ignore[arg-type]
            effect_mode="write",
            rows=[],
            tokens=[],
            pending_outcome=pending_outcome,
            all_states=[],
            sink_name="legacy",
            sink_node_id="node-legacy",
        )


class _RequiredFieldsLegacySink(_ForbiddenLegacySink):
    declared_required_fields = frozenset({"must_exist"})


def test_validate_sink_input_rows_contracts_desync_is_orchestration_bug() -> None:
    """``contracts``, when provided, must pair 1:1 with ``rows``.

    Both production call sites build rows and contracts from the same tokens
    iterable, so a length desync is a refactor bug the Layer 2 backstop must
    surface loudly rather than silently mis-attributing per-row context.
    """
    with pytest.raises(OrchestrationInvariantError, match="must be paired 1:1"):
        SinkExecutor._validate_sink_input(
            _RequiredFieldsLegacySink(),  # type: ignore[arg-type]
            [{"must_exist": 1}, {"must_exist": 2}],
            skip_schema=True,
            contracts=[],
        )
