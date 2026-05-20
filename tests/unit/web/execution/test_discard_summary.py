"""Tests for web discard summaries under ADR-019 token outcomes."""

from elspeth.contracts.audit import DISCARD_SINK_NAME, TokenRef
from elspeth.contracts.enums import TerminalOutcome, TerminalPath
from elspeth.web.execution.discard_summary import load_discard_summaries_from_db
from tests.fixtures.landscape import make_recorder_with_run


def test_discard_summary_counts_completed_discard_path() -> None:
    setup = make_recorder_with_run(run_id="discard-summary-run", source_node_id="source-0")
    row = setup.data_flow.create_row(
        run_id=setup.run_id,
        source_node_id=setup.source_node_id,
        row_index=0,
        data={"id": "drop-me"},
        source_row_index=0,
        ingest_sequence=0,
    )
    token = setup.data_flow.create_token(row.row_id)
    setup.data_flow.record_token_outcome(
        ref=TokenRef(token_id=token.token_id, run_id=setup.run_id),
        outcome=TerminalOutcome.FAILURE,
        path=TerminalPath.SINK_DISCARDED,
        sink_name=DISCARD_SINK_NAME,
        error_hash="a" * 64,
    )

    summaries = load_discard_summaries_from_db(setup.db, [setup.run_id])

    assert summaries[setup.run_id].total == 1
    assert summaries[setup.run_id].sink_discards == 1
