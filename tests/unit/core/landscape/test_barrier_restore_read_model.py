from __future__ import annotations

from elspeth.contracts import NodeType
from elspeth.contracts.audit import TokenRef
from elspeth.contracts.enums import TerminalPath
from tests.fixtures.landscape import make_recorder_with_run, register_test_node


def test_barrier_restore_read_model_reports_duplicate_live_buffered_acceptances() -> None:
    setup = make_recorder_with_run(run_id="run-restore-read")
    agg_node_id = register_test_node(
        setup.factory.data_flow,
        setup.run_id,
        "agg-node-1",
        node_type=NodeType.TRANSFORM,
        plugin_name="batch_stats",
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
    ref = TokenRef(token_id=token.token_id, run_id=setup.run_id)

    setup.factory.data_flow.record_token_outcome(ref, None, TerminalPath.BUFFERED, batch_id=batch.batch_id)
    setup.factory.data_flow.record_token_outcome(ref, None, TerminalPath.BUFFERED, batch_id=batch.batch_id)

    duplicate_acceptances = setup.factory.barrier_restore.find_duplicate_live_buffered_acceptances(setup.run_id)

    assert duplicate_acceptances == [(token.token_id, 2)]
