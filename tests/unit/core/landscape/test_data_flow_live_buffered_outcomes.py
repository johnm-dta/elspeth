"""Slice 3 (ADR-030 §E.4 / §G new-errors row): live-BUFFERED restore reads.

``token_outcomes`` has NO non-terminal uniqueness (the only unique index is
partial on ``completed=1``), so a deposed leader's unfenced intake could
historically write a SECOND BUFFERED acceptance which
``get_token_outcome``'s ``ORDER BY ... LIMIT 1`` silently swallowed.
``get_live_buffered_outcomes`` surfaces every live acceptance so the restore
path (``_derive_restored_batch_id``) can refuse loudly with Tier-1 instead of
silent latest-wins; ``find_duplicate_live_buffered_outcomes`` is the run-wide
belt for restore entry.
"""

from __future__ import annotations

from datetime import UTC, datetime

from elspeth.contracts import NodeType
from elspeth.contracts.audit import TokenRef
from elspeth.contracts.enums import TerminalOutcome, TerminalPath
from tests.fixtures.landscape import make_recorder_with_run, register_test_node

NOW = datetime(2026, 6, 12, 12, 0, 0, tzinfo=UTC)


def _setup_buffered_token():  # type: ignore[no-untyped-def]
    """Run + agg node + batch + one token; returns (setup, agg_node_id, batch, token)."""
    setup = make_recorder_with_run(run_id="run-buf-1")
    agg_node_id = register_test_node(
        setup.factory.data_flow, setup.run_id, "agg-node-1", node_type=NodeType.TRANSFORM, plugin_name="batch_stats"
    )
    batch = setup.factory.execution.create_batch(setup.run_id, agg_node_id)
    row = setup.factory.data_flow.create_row(
        setup.run_id,
        setup.source_node_id,
        0,
        {"id": 1},
        source_row_index=0,
        ingest_sequence=0,
    )
    token = setup.factory.data_flow.create_token(row_id=row.row_id)
    return setup, agg_node_id, batch, token


def test_single_live_buffered_outcome_is_returned() -> None:
    setup, _node_id, batch, token = _setup_buffered_token()
    ref = TokenRef(token_id=token.token_id, run_id=setup.run_id)
    outcome_id = setup.factory.data_flow.record_token_outcome(ref, None, TerminalPath.BUFFERED, batch_id=batch.batch_id)

    live = setup.factory.data_flow.get_live_buffered_outcomes(ref)

    assert [outcome.outcome_id for outcome in live] == [outcome_id]
    assert live[0].completed is False
    assert live[0].path is TerminalPath.BUFFERED
    assert live[0].batch_id == batch.batch_id


def test_duplicate_live_buffered_outcomes_are_all_returned_in_order() -> None:
    """The F2 image: no non-terminal uniqueness, so a second BUFFERED INSERT
    succeeds — the verb must surface BOTH, never latest-wins."""
    setup, _node_id, batch, token = _setup_buffered_token()
    ref = TokenRef(token_id=token.token_id, run_id=setup.run_id)
    first = setup.factory.data_flow.record_token_outcome(ref, None, TerminalPath.BUFFERED, batch_id=batch.batch_id)
    second = setup.factory.data_flow.record_token_outcome(ref, None, TerminalPath.BUFFERED, batch_id=batch.batch_id)

    live = setup.factory.data_flow.get_live_buffered_outcomes(ref)

    assert len(live) == 2
    assert {outcome.outcome_id for outcome in live} == {first, second}
    # Deterministic order: (recorded_at, outcome_id).
    assert [outcome.outcome_id for outcome in live] == sorted(
        [outcome.outcome_id for outcome in live],
        key=lambda oid: (next(o.recorded_at for o in live if o.outcome_id == oid), oid),
    )
    # The historical read silently took ONE of them — the new verb is what
    # lets the restore path refuse instead.
    silent = setup.factory.data_flow.get_token_outcome(token.token_id)
    assert silent is not None and silent.outcome_id in {first, second}


def test_flushed_token_buffered_history_is_exempt() -> None:
    """A token with a terminal outcome has NO live BUFFERED rows — its
    BUFFERED row is dead history (the batch flushed)."""
    setup, _node_id, batch, token = _setup_buffered_token()
    ref = TokenRef(token_id=token.token_id, run_id=setup.run_id)
    setup.factory.data_flow.record_token_outcome(ref, None, TerminalPath.BUFFERED, batch_id=batch.batch_id)
    setup.factory.data_flow.record_token_outcome(ref, TerminalOutcome.TRANSIENT, TerminalPath.BATCH_CONSUMED, batch_id=batch.batch_id)

    assert setup.factory.data_flow.get_live_buffered_outcomes(ref) == []
    assert setup.factory.data_flow.find_duplicate_live_buffered_outcomes(setup.run_id) == []


def test_no_outcomes_is_empty() -> None:
    setup, _node_id, _batch, token = _setup_buffered_token()
    ref = TokenRef(token_id=token.token_id, run_id=setup.run_id)
    assert setup.factory.data_flow.get_live_buffered_outcomes(ref) == []


def test_run_wide_duplicate_sweep_names_only_duplicated_tokens() -> None:
    setup, _node_id, batch, token = _setup_buffered_token()
    ref = TokenRef(token_id=token.token_id, run_id=setup.run_id)
    setup.factory.data_flow.record_token_outcome(ref, None, TerminalPath.BUFFERED, batch_id=batch.batch_id)
    setup.factory.data_flow.record_token_outcome(ref, None, TerminalPath.BUFFERED, batch_id=batch.batch_id)

    # A healthy sibling token with ONE live BUFFERED row stays out of the report.
    row2 = setup.factory.data_flow.create_row(
        setup.run_id,
        setup.source_node_id,
        1,
        {"id": 2},
        source_row_index=1,
        ingest_sequence=1,
    )
    token2 = setup.factory.data_flow.create_token(row_id=row2.row_id)
    setup.factory.data_flow.record_token_outcome(
        TokenRef(token_id=token2.token_id, run_id=setup.run_id), None, TerminalPath.BUFFERED, batch_id=batch.batch_id
    )

    assert setup.factory.data_flow.find_duplicate_live_buffered_outcomes(setup.run_id) == [(token.token_id, 2)]
