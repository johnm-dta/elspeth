from __future__ import annotations

from elspeth.contracts import NodeStateStatus, NodeType
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


def test_barrier_restore_read_model_reports_max_node_state_attempts() -> None:
    setup = make_recorder_with_run(run_id="run-restore-attempts")
    node_id = register_test_node(setup.factory.data_flow, setup.run_id, "sink-node")
    row = setup.factory.data_flow.create_row(
        setup.run_id,
        setup.source_node_id,
        0,
        {"id": 1},
        source_row_index=0,
        ingest_sequence=0,
    )
    token = setup.factory.data_flow.create_token(row_id=row.row_id, token_id="token-attempt")
    setup.factory.execution.begin_node_state(token.token_id, node_id, setup.run_id, 1, {"id": 1}, attempt=0)
    setup.factory.execution.begin_node_state(token.token_id, node_id, setup.run_id, 2, {"id": 1}, attempt=3)

    assert setup.factory.barrier_restore.get_max_node_state_attempts(setup.run_id, [token.token_id]) == {token.token_id: 3}
    assert setup.factory.barrier_restore.get_max_node_state_attempts(
        setup.run_id,
        [token.token_id],
        step_index=1,
    ) == {token.token_id: 0}


def test_barrier_restore_read_model_reports_open_coalesce_hold_state_ids() -> None:
    setup = make_recorder_with_run(run_id="run-restore-open-holds")
    node_id = register_test_node(setup.factory.data_flow, setup.run_id, "coalesce-node")
    row = setup.factory.data_flow.create_row(
        setup.run_id,
        setup.source_node_id,
        0,
        {"id": 1},
        source_row_index=0,
        ingest_sequence=0,
    )
    token = setup.factory.data_flow.create_token(row_id=row.row_id, token_id="token-held")
    setup.factory.execution.begin_node_state(
        token.token_id,
        node_id,
        setup.run_id,
        1,
        {"id": 1},
        state_id="state-low",
        attempt=0,
    )
    setup.factory.execution.begin_node_state(
        token.token_id,
        node_id,
        setup.run_id,
        1,
        {"id": 1},
        state_id="state-high",
        attempt=2,
    )

    assert setup.factory.barrier_restore.get_open_node_state_ids(
        setup.run_id,
        node_ids=[node_id],
        token_ids=[token.token_id],
    ) == {token.token_id: "state-high"}


def test_barrier_restore_read_model_reports_completed_coalesce_row_ids() -> None:
    setup = make_recorder_with_run(run_id="run-restore-completed")
    node_id = register_test_node(setup.factory.data_flow, setup.run_id, "coalesce-node")
    other_node_id = register_test_node(setup.factory.data_flow, setup.run_id, "other-coalesce-node")
    row = setup.factory.data_flow.create_row(
        setup.run_id,
        setup.source_node_id,
        0,
        {"id": 1},
        source_row_index=0,
        ingest_sequence=0,
        row_id="row-done",
    )
    token = setup.factory.data_flow.create_token(row_id=row.row_id, token_id="token-done")
    open_row = setup.factory.data_flow.create_row(
        setup.run_id,
        setup.source_node_id,
        1,
        {"id": 2},
        source_row_index=1,
        ingest_sequence=1,
        row_id="row-open",
    )
    open_token = setup.factory.data_flow.create_token(row_id=open_row.row_id, token_id="token-open")
    state = setup.factory.execution.begin_node_state(
        token.token_id,
        node_id,
        setup.run_id,
        1,
        {"id": 1},
    )
    setup.factory.execution.complete_node_state(
        state.state_id,
        NodeStateStatus.COMPLETED,
        output_data={"id": 1},
        duration_ms=1.0,
    )
    setup.factory.execution.begin_node_state(
        open_token.token_id,
        node_id,
        setup.run_id,
        1,
        {"id": 2},
    )

    assert setup.factory.barrier_restore.get_completed_row_ids_for_nodes(
        setup.run_id,
        frozenset({node_id, other_node_id}),
    ) == {(node_id, "row-done")}
    assert (
        setup.factory.barrier_restore.get_completed_row_ids_for_nodes(
            "other-run",
            frozenset({node_id}),
        )
        == set()
    )
    assert (
        setup.factory.barrier_restore.has_completed_row_for_node(
            run_id=setup.run_id,
            node_id=node_id,
            row_id="row-done",
        )
        is True
    )
    assert (
        setup.factory.barrier_restore.has_completed_row_for_node(
            run_id=setup.run_id,
            node_id=other_node_id,
            row_id="row-done",
        )
        is False
    )
    assert (
        setup.factory.barrier_restore.has_completed_row_for_node(
            run_id="other-run",
            node_id=node_id,
            row_id="row-done",
        )
        is False
    )
    assert (
        setup.factory.barrier_restore.has_completed_row_for_node(
            run_id=setup.run_id,
            node_id=node_id,
            row_id="row-open",
        )
        is False
    )
