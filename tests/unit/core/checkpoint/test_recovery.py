"""Unit tests for RecoveryManager resume and row-recovery behavior."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ConfigDict
from sqlalchemy import Connection, select, update

from elspeth.contracts import (
    Checkpoint,
    Determinism,
    NodeType,
    PayloadStore,
    PluginSchema,
    ResumedRow,
    RunStatus,
    TerminalOutcome,
    TerminalPath,
)
from elspeth.contracts.barrier_scalars import AggregationNodeScalars, BarrierScalars
from elspeth.contracts.contract_records import ContractAuditRecord
from elspeth.contracts.errors import AuditIntegrityError, EmptyResumeStateError, OrchestrationInvariantError
from elspeth.contracts.scheduler import TokenWorkStatus
from elspeth.contracts.schema_contract import FieldContract, SchemaContract
from elspeth.contracts.types import NodeID
from elspeth.core.checkpoint import CheckpointCorruptionError, CheckpointManager, RecoveryManager
from elspeth.core.checkpoint.manager import IncompatibleCheckpointError
from elspeth.core.checkpoint.recovery import _DELEGATION_PATHS, IncompleteTokenSpec
from elspeth.core.dag import ExecutionGraph
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.schema import (
    checkpoints_table,
    nodes_table,
    rows_table,
    run_sources_table,
    runs_table,
    token_outcomes_table,
    token_work_items_table,
    tokens_table,
)
from tests.fixtures.landscape import make_landscape_db
from tests.helpers.checkpoint import checkpoint_draft


@pytest.fixture
def db() -> LandscapeDB:
    return make_landscape_db()


@pytest.fixture
def checkpoint_manager(db: LandscapeDB) -> CheckpointManager:
    return CheckpointManager(db)


@pytest.fixture
def recovery_manager(db: LandscapeDB, checkpoint_manager: CheckpointManager) -> RecoveryManager:
    return RecoveryManager(db, checkpoint_manager)


def _create_contract() -> tuple[str, str]:
    contract = SchemaContract(
        mode="FIXED",
        fields=(
            FieldContract(
                normalized_name="id",
                original_name="id",
                python_type=int,
                required=True,
                source="declared",
            ),
        ),
        locked=True,
    )
    return ContractAuditRecord.from_contract(contract).to_json(), contract.version_hash()


def _create_graph(*, node_id: str = "checkpoint-node", config: dict[str, Any] | None = None) -> ExecutionGraph:
    graph = ExecutionGraph()
    graph.add_node(node_id, node_type=NodeType.TRANSFORM, plugin_name="test", config=config or {})
    return graph


def _create_checkpoint(
    checkpoint_manager: CheckpointManager,
    *,
    run_id: str,
    sequence_number: int,
    graph: ExecutionGraph,
    barrier_scalars: BarrierScalars | None = None,
) -> Checkpoint:
    return checkpoint_manager.create_checkpoint(
        draft=checkpoint_draft(
            run_id=run_id,
            sequence_number=sequence_number,
            graph=graph,
            barrier_scalars=barrier_scalars,
        )
    )


def _insert_run(
    conn: Connection,
    run_id: str,
    *,
    status: RunStatus | str,
    with_contract: bool = False,
    contract_json_override: str | None = None,
) -> None:
    """Insert a ``runs`` row, plus a ``run_sources`` row when a contract is requested.

    ADR-025 §3 Decision 5 (G6): the schema contract lives exclusively on
    ``run_sources.schema_contract_json``; the run-level singleton columns
    were deleted along with their accessors. The helper auto-creates a
    SOURCE node "source-node" when a contract is requested so the
    ``run_sources`` foreign-key constraint holds; callers that need to
    inspect the source node explicitly insert additional nodes via
    :func:`_insert_node` after this helper returns.
    """
    schema_contract_json: str | None = None
    schema_contract_hash: str | None = None
    if with_contract:
        schema_contract_json, schema_contract_hash = _create_contract()
    if contract_json_override is not None:
        schema_contract_json = contract_json_override
        # Intentionally mismatched when override is used for corruption tests.
        schema_contract_hash = "deadbeefdeadbeef"

    conn.execute(
        runs_table.insert().values(
            run_id=run_id,
            started_at=datetime.now(UTC),
            config_hash="cfg",
            settings_json="{}",
            canonical_version="sha256-rfc8785-v1",
            status=status,
            openrouter_catalog_sha256="0" * 64,
            openrouter_catalog_source="bundled",
        )
    )

    if schema_contract_json is not None:
        # Ensure the SOURCE node exists before writing run_sources (FK constraint).
        # The node may already have been inserted by an earlier helper call;
        # check first instead of relying on ON CONFLICT semantics, since the
        # in-memory SQLite engine doesn't return rowcount reliably for
        # INSERT OR IGNORE under all dialects.
        existing_node = conn.execute(
            select(nodes_table.c.node_id).where(nodes_table.c.node_id == "source-node").where(nodes_table.c.run_id == run_id)
        ).fetchone()
        if existing_node is None:
            _insert_node(conn, run_id, "source-node", node_type=NodeType.SOURCE)
        conn.execute(
            run_sources_table.insert().values(
                run_id=run_id,
                source_node_id="source-node",
                source_name="primary",
                plugin_name="test_source",
                lifecycle_state="loaded",
                config_hash="src_cfg",
                schema_json="{}",
                schema_contract_json=schema_contract_json,
                schema_contract_hash=schema_contract_hash,
                field_resolution_json=None,
                recorded_at=datetime.now(UTC),
            )
        )


def _insert_node(conn: Connection, run_id: str, node_id: str, *, node_type: NodeType = NodeType.TRANSFORM) -> None:
    conn.execute(
        nodes_table.insert().values(
            node_id=node_id,
            run_id=run_id,
            plugin_name="test",
            node_type=node_type,
            plugin_version="1.0.0",
            determinism=Determinism.DETERMINISTIC,
            config_hash="node_cfg",
            config_json="{}",
            registered_at=datetime.now(UTC),
        )
    )


def _insert_row(conn: Connection, run_id: str, row_id: str, *, row_index: int, source_data_ref: str | None) -> None:
    conn.execute(
        rows_table.insert().values(
            row_id=row_id,
            run_id=run_id,
            source_node_id="source-node",
            row_index=row_index,
            source_row_index=row_index,
            ingest_sequence=row_index,
            source_data_hash=f"hash-{row_id}",
            source_data_ref=source_data_ref,
            created_at=datetime.now(UTC),
        )
    )


def _insert_token(conn: Connection, run_id: str, token_id: str, row_id: str) -> None:
    conn.execute(
        tokens_table.insert().values(
            token_id=token_id,
            row_id=row_id,
            run_id=run_id,
            created_at=datetime.now(UTC),
        )
    )


def _insert_terminal_outcome(
    conn: Connection,
    run_id: str,
    token_id: str,
    *,
    outcome: TerminalOutcome | None = TerminalOutcome.SUCCESS,
    path: TerminalPath | None = None,
    completed: bool | None = None,
) -> None:
    resolved_outcome = outcome
    resolved_path = path or TerminalPath.DEFAULT_FLOW
    resolved_completed = completed if completed is not None else outcome is not None
    if path is not None:
        resolved_path = path
    if completed is not None:
        resolved_completed = completed

    conn.execute(
        token_outcomes_table.insert().values(
            outcome_id=f"out-{token_id}",
            run_id=run_id,
            token_id=token_id,
            outcome=resolved_outcome.value if resolved_outcome is not None else None,
            path=resolved_path.value,
            completed=1 if resolved_completed else 0,
            recorded_at=datetime.now(UTC),
            sink_name="sink",
        )
    )


def _insert_blocked_work_item(
    conn: Connection,
    run_id: str,
    token_id: str,
    row_id: str,
    *,
    barrier_key: str | None,
    queue_key: str | None = None,
    status: TokenWorkStatus = TokenWorkStatus.BLOCKED,
) -> None:
    """Insert a scheduler-journal work item (F1: journal owns buffered tokens).

    ``barrier_key`` non-NULL marks a barrier hold (aggregation node_id or
    coalesce name); ``barrier_key=None`` with a ``queue_key`` models an
    ADR-028 queue-hold, which must stay IN the re-drive work set.
    """
    now = datetime.now(UTC)
    conn.execute(
        token_work_items_table.insert().values(
            work_item_id=f"wi-{token_id}",
            run_id=run_id,
            token_id=token_id,
            row_id=row_id,
            node_id=None,
            step_index=0,
            ingest_sequence=0,
            row_payload_json="{}",
            status=status.value,
            queue_key=queue_key,
            barrier_key=barrier_key,
            attempt=0,
            available_at=now,
            barrier_blocked_at=now if barrier_key is not None else None,
            created_at=now,
            updated_at=now,
        )
    )


def _create_failed_run_with_checkpoint(
    db: LandscapeDB,
    checkpoint_manager: CheckpointManager,
    run_id: str,
    *,
    status: RunStatus | str = RunStatus.FAILED,
    checkpoint_node_id: str = "checkpoint-node",
    with_contract: bool = True,
    barrier_scalars: BarrierScalars | None = None,
    graph: ExecutionGraph | None = None,
) -> ExecutionGraph:
    active_graph = graph or _create_graph(node_id=checkpoint_node_id)

    with db.connection() as conn:
        _insert_run(conn, run_id, status=status, with_contract=with_contract)
        # _insert_run auto-creates "source-node" + run_sources when with_contract=True
        # (per ADR-025 §3 Decision 5). Only insert explicitly when no contract is set.
        if not with_contract:
            _insert_node(conn, run_id, "source-node", node_type=NodeType.SOURCE)
        _insert_node(conn, run_id, checkpoint_node_id)
        _insert_row(conn, run_id, "row-0", row_index=0, source_data_ref=None)
        _insert_token(conn, run_id, "tok-0", "row-0")

    _create_checkpoint(
        checkpoint_manager,
        run_id=run_id,
        sequence_number=1,
        barrier_scalars=barrier_scalars,
        graph=active_graph,
    )
    return active_graph


def test_can_resume_returns_false_for_missing_run(recovery_manager: RecoveryManager) -> None:
    check = recovery_manager.can_resume("missing", _create_graph())
    assert check.can_resume is False
    assert check.reason == "Run missing not found"


def test_can_resume_rejects_completed_run(db: LandscapeDB, recovery_manager: RecoveryManager) -> None:
    with db.connection() as conn:
        _insert_run(conn, "run-completed", status=RunStatus.COMPLETED)

    check = recovery_manager.can_resume("run-completed", _create_graph())
    assert check.can_resume is False
    assert check.reason == "Run already completed successfully"


def test_can_resume_rejects_running_run_with_live_seat(db: LandscapeDB, recovery_manager: RecoveryManager) -> None:
    """RUNNING + LIVE seat is refused: run is held by an active leader.

    Slice-4 flip (§H test #2(c)): RUNNING + absent/expired seat is now
    RESUMABLE (dead-leader takeover).  This test pins the live-seat refusal
    that must remain — only the expired-seat arm flipped.
    """
    from elspeth.contracts.coordination import mint_worker_id
    from elspeth.core.landscape.run_coordination_repository import RunCoordinationRepository

    with db.connection() as conn:
        _insert_run(conn, "run-running", status=RunStatus.RUNNING)
    # Register a live leader seat so the guard fires the refusal.
    leader_id = mint_worker_id("run-running")
    RunCoordinationRepository(db.engine).register_run_leader(
        run_id="run-running",
        worker_id=leader_id,
        now=datetime.now(UTC),
        window_seconds=80.0,
    )

    check = recovery_manager.can_resume("run-running", _create_graph())
    assert check.can_resume is False
    assert check.reason is not None
    assert "in progress under live leader" in check.reason


@pytest.mark.parametrize(
    "status",
    [RunStatus.COMPLETED_WITH_FAILURES, RunStatus.EMPTY],
)
def test_can_resume_rejects_terminal_statuses_even_when_checkpoint_exists(
    db: LandscapeDB,
    checkpoint_manager: CheckpointManager,
    recovery_manager: RecoveryManager,
    status: RunStatus,
) -> None:
    run_id = f"run-terminal-{status.value}"
    graph = _create_failed_run_with_checkpoint(db, checkpoint_manager, run_id, status=status)

    check = recovery_manager.can_resume(run_id, graph)

    assert check.can_resume is False
    assert check.reason == f"Run status {status.value!r} is not resumable"


def test_can_resume_rejects_corrupt_stored_run_status(
    db: LandscapeDB,
    checkpoint_manager: CheckpointManager,
    recovery_manager: RecoveryManager,
) -> None:
    run_id = "run-corrupt-status"
    graph = _create_failed_run_with_checkpoint(db, checkpoint_manager, run_id, status="bogus")

    with pytest.raises(CheckpointCorruptionError, match="invalid status 'bogus'"):
        recovery_manager.can_resume(run_id, graph)


def test_can_resume_rejects_failed_run_without_checkpoint(db: LandscapeDB, recovery_manager: RecoveryManager) -> None:
    with db.connection() as conn:
        _insert_run(conn, "run-no-checkpoint", status=RunStatus.FAILED)

    check = recovery_manager.can_resume("run-no-checkpoint", _create_graph())
    assert check.can_resume is False
    assert check.reason == "Run has no resume baseline (run predates run-start checkpointing or checkpointing was disabled)"


def test_can_resume_reason_for_missing_baseline_is_journal_flavoured(db: LandscapeDB, recovery_manager: RecoveryManager) -> None:
    """F1 Task 3.2: the missing-checkpoint refuse speaks journal semantics.

    Post-F1 the checkpoint row is only the resume BASELINE (scalars +
    topology anchor); buffered work lives in journal BLOCKED rows. The
    refuse reason must not present the checkpoint as the recovery store —
    if it mentions checkpointing at all, it is in the baseline framing.
    """
    with db.connection() as conn:
        _insert_run(conn, "run-no-baseline", status=RunStatus.FAILED)

    check = recovery_manager.can_resume("run-no-baseline", _create_graph())
    assert not check.can_resume
    assert check.reason is not None
    assert "checkpoint" not in check.reason.lower() or "baseline" in check.reason.lower()


def test_can_resume_returns_reason_when_checkpoint_format_is_incompatible(
    db: LandscapeDB, recovery_manager: RecoveryManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    with db.connection() as conn:
        _insert_run(conn, "run-incompatible", status=RunStatus.FAILED)

    def _raise_incompatible(_run_id: str) -> None:
        raise IncompatibleCheckpointError("bad checkpoint format")

    monkeypatch.setattr(recovery_manager._checkpoint_manager, "get_latest_checkpoint", _raise_incompatible)
    check = recovery_manager.can_resume("run-incompatible", _create_graph())
    assert check.can_resume is False
    assert check.reason == "bad checkpoint format"


def test_get_unprocessed_rows_rejects_incompatible_checkpoint_format(
    db: LandscapeDB,
    checkpoint_manager: CheckpointManager,
    recovery_manager: RecoveryManager,
) -> None:
    run_id = "run-workset-incompatible-format"
    _create_failed_run_with_checkpoint(db, checkpoint_manager, run_id)
    with db.engine.begin() as conn:
        conn.execute(update(checkpoints_table).where(checkpoints_table.c.run_id == run_id).values(format_version=None))

    with pytest.raises(IncompatibleCheckpointError, match="missing format_version"):
        recovery_manager.get_unprocessed_rows(run_id)


def test_can_resume_rejects_topology_mismatch(
    db: LandscapeDB,
    checkpoint_manager: CheckpointManager,
    recovery_manager: RecoveryManager,
) -> None:
    run_id = "run-topology-mismatch"
    original_graph = _create_failed_run_with_checkpoint(
        db,
        checkpoint_manager,
        run_id,
        graph=_create_graph(node_id="checkpoint-node", config={"version": 1}),
    )
    assert original_graph.has_node("checkpoint-node")

    changed_graph = _create_graph(node_id="checkpoint-node", config={"version": 2})
    check = recovery_manager.can_resume(run_id, changed_graph)
    assert check.can_resume is False
    assert check.reason is not None


def test_can_resume_true_for_failed_run_with_valid_checkpoint(
    db: LandscapeDB, checkpoint_manager: CheckpointManager, recovery_manager: RecoveryManager
) -> None:
    run_id = "run-resumable"
    graph = _create_failed_run_with_checkpoint(db, checkpoint_manager, run_id)

    check = recovery_manager.can_resume(run_id, graph)
    assert check.can_resume is True
    assert check.reason is None


def test_get_resume_point_returns_none_when_run_cannot_resume(recovery_manager: RecoveryManager) -> None:
    assert recovery_manager.get_resume_point("missing", _create_graph()) is None


def test_get_resume_point_returns_none_if_checkpoint_missing_after_can_resume(
    db: LandscapeDB,
    recovery_manager: RecoveryManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with db.connection() as conn:
        _insert_run(conn, "run-race", status=RunStatus.FAILED, with_contract=True)

    monkeypatch.setattr(recovery_manager, "can_resume", lambda _run_id, _graph: type("Check", (), {"can_resume": True})())
    monkeypatch.setattr(recovery_manager._checkpoint_manager, "get_latest_checkpoint", lambda _run_id: None)

    assert recovery_manager.get_resume_point("run-race", _create_graph()) is None


def test_get_resume_point_restores_barrier_scalars(
    db: LandscapeDB,
    checkpoint_manager: CheckpointManager,
    recovery_manager: RecoveryManager,
) -> None:
    """F1: the checkpoint contributes only the underivable barrier scalars."""
    run_id = "run-resume-point"
    graph = _create_failed_run_with_checkpoint(
        db,
        checkpoint_manager,
        run_id,
        barrier_scalars=BarrierScalars(
            aggregation={"agg-node": AggregationNodeScalars(count_fire_offset=1.5, condition_fire_offset=None)},
            coalesce={},
        ),
    )

    resume_point = recovery_manager.get_resume_point(run_id, graph)
    assert resume_point is not None
    assert resume_point.sequence_number == 1
    assert resume_point.barrier_scalars is not None
    assert resume_point.barrier_scalars.aggregation["agg-node"].count_fire_offset == 1.5


def test_get_resume_point_caches_barrier_scalars_parse(
    db: LandscapeDB,
    checkpoint_manager: CheckpointManager,
    recovery_manager: RecoveryManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resume inspection should not parse the same barrier-scalars payload twice."""
    import importlib

    recovery_module: Any = importlib.import_module("elspeth.core.checkpoint.recovery")

    run_id = "run-scalars-parse"
    graph = _create_failed_run_with_checkpoint(
        db,
        checkpoint_manager,
        run_id,
        barrier_scalars=BarrierScalars(
            aggregation={"agg-node": AggregationNodeScalars(count_fire_offset=2.0, condition_fire_offset=None)},
            coalesce={},
        ),
    )

    original_checkpoint_loads = recovery_module.checkpoint_loads
    parsed_payloads: list[str] = []

    def counting_checkpoint_loads(payload: str) -> Any:
        parsed_payloads.append(payload)
        return original_checkpoint_loads(payload)

    monkeypatch.setattr(recovery_module, "checkpoint_loads", counting_checkpoint_loads)

    first = recovery_manager.get_resume_point(run_id, graph)
    second = recovery_manager.get_resume_point(run_id, graph)
    assert first is not None
    assert second is not None
    assert first.barrier_scalars is not None
    assert recovery_manager.get_unprocessed_rows(run_id) == ["row-0"]
    assert parsed_payloads == [first.checkpoint.barrier_scalars_json]


def test_get_unprocessed_rows_returns_empty_when_no_checkpoint(recovery_manager: RecoveryManager) -> None:
    assert recovery_manager.get_unprocessed_rows("missing-run") == []


def test_get_unprocessed_rows_orders_by_ingest_sequence(
    db: LandscapeDB,
    checkpoint_manager: CheckpointManager,
    recovery_manager: RecoveryManager,
) -> None:
    """Resume replay order is global ingest order, not source-local row_index."""
    run_id = "run-ingest-order"
    graph = _create_graph(node_id="checkpoint-node")
    with db.connection() as conn:
        _insert_run(conn, run_id, status=RunStatus.FAILED, with_contract=True)
        _insert_node(conn, run_id, "checkpoint-node")
        _insert_row(conn, run_id, "row-a", row_index=0, source_data_ref=None)
        _insert_row(conn, run_id, "row-b", row_index=1, source_data_ref=None)
        _insert_token(conn, run_id, "token-checkpoint", "row-a")
        conn.execute(rows_table.update().where(rows_table.c.row_id == "row-a").values(ingest_sequence=10))
        conn.execute(rows_table.update().where(rows_table.c.row_id == "row-b").values(ingest_sequence=5))

    _create_checkpoint(
        checkpoint_manager,
        run_id=run_id,
        sequence_number=1,
        barrier_scalars=None,
        graph=graph,
    )

    assert recovery_manager.get_unprocessed_rows(run_id) == ["row-b", "row-a"]


@pytest.mark.parametrize("delegation_path", [TerminalPath.FORK_PARENT, TerminalPath.EXPAND_PARENT])
def test_get_unprocessed_rows_uses_terminal_path_delegation_set(
    db: LandscapeDB,
    checkpoint_manager: CheckpointManager,
    recovery_manager: RecoveryManager,
    delegation_path: TerminalPath,
) -> None:
    """ADR-019: fork/expand parents are delegation markers by path."""
    expected_delegation_paths = (
        TerminalPath.FORK_PARENT.value,
        TerminalPath.EXPAND_PARENT.value,
    )
    assert expected_delegation_paths == _DELEGATION_PATHS
    run_id = f"run-resume-{delegation_path.value}"
    graph = _create_graph(node_id="checkpoint-node")
    with db.connection() as conn:
        _insert_run(conn, run_id, status=RunStatus.FAILED, with_contract=True)
        _insert_node(conn, run_id, "checkpoint-node")
        _insert_row(conn, run_id, "row-resume-delegation", row_index=0, source_data_ref=None)
        _insert_token(conn, run_id, "token-resume-delegation-parent", "row-resume-delegation")
        _insert_terminal_outcome(
            conn,
            run_id,
            "token-resume-delegation-parent",
            outcome=TerminalOutcome.TRANSIENT,
            path=delegation_path,
            completed=True,
        )

    _create_checkpoint(
        checkpoint_manager,
        run_id=run_id,
        sequence_number=1,
        barrier_scalars=None,
        graph=graph,
    )

    assert recovery_manager.get_unprocessed_rows(run_id) == ["row-resume-delegation"]


def test_get_unprocessed_rows_handles_fork_and_excludes_buffered_rows(
    db: LandscapeDB,
    checkpoint_manager: CheckpointManager,
    recovery_manager: RecoveryManager,
) -> None:
    """F1: buffered = journal BLOCKED rows with a barrier_key, not checkpoint blobs."""
    run_id = "run-unprocessed-complex"
    graph = _create_graph(node_id="checkpoint-node")
    with db.connection() as conn:
        _insert_run(conn, run_id, status=RunStatus.FAILED, with_contract=True)
        _insert_node(conn, run_id, "checkpoint-node")

        # row-completed: one completed token -> should be excluded.
        _insert_row(conn, run_id, "row-completed", row_index=0, source_data_ref=None)
        _insert_token(conn, run_id, "tok-completed", "row-completed")
        _insert_terminal_outcome(conn, run_id, "tok-completed", outcome=TerminalOutcome.SUCCESS, path=TerminalPath.DEFAULT_FLOW)

        # row-delegation-only: FORKED parent only, no child terminal -> should be included.
        _insert_row(conn, run_id, "row-delegation-only", row_index=1, source_data_ref=None)
        _insert_token(conn, run_id, "tok-parent", "row-delegation-only")
        _insert_terminal_outcome(conn, run_id, "tok-parent", outcome=TerminalOutcome.TRANSIENT, path=TerminalPath.FORK_PARENT)

        # row-child-pending: one completed child + one pending child -> should be included.
        _insert_row(conn, run_id, "row-child-pending", row_index=2, source_data_ref=None)
        _insert_token(conn, run_id, "tok-child-ok", "row-child-pending")
        _insert_terminal_outcome(conn, run_id, "tok-child-ok", outcome=TerminalOutcome.SUCCESS, path=TerminalPath.DEFAULT_FLOW)
        _insert_token(conn, run_id, "tok-child-pending", "row-child-pending")

        # row-buffered: appears incomplete but all incomplete tokens are held
        # at a barrier in the journal -> excluded (restored, not re-driven).
        _insert_row(conn, run_id, "row-buffered", row_index=3, source_data_ref=None)
        _insert_token(conn, run_id, "tok-buffered", "row-buffered")
        _insert_blocked_work_item(conn, run_id, "tok-buffered", "row-buffered", barrier_key="agg-node")

        # row-mixed-buffering: one incomplete token buffered + one incomplete token
        # not buffered -> must remain unprocessed.
        _insert_row(conn, run_id, "row-mixed-buffering", row_index=4, source_data_ref=None)
        _insert_token(conn, run_id, "tok-mixed-buffered", "row-mixed-buffering")
        _insert_token(conn, run_id, "tok-mixed-pending", "row-mixed-buffering")
        _insert_blocked_work_item(conn, run_id, "tok-mixed-buffered", "row-mixed-buffering", barrier_key="agg-node")

    _create_checkpoint(
        checkpoint_manager,
        run_id=run_id,
        sequence_number=10,
        barrier_scalars=None,
        graph=graph,
    )

    unprocessed = recovery_manager.get_unprocessed_rows(run_id)
    assert unprocessed == ["row-delegation-only", "row-child-pending", "row-mixed-buffering"]


def test_get_unprocessed_rows_chunks_buffered_token_query(
    db: LandscapeDB,
    checkpoint_manager: CheckpointManager,
    recovery_manager: RecoveryManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: buffered-token filtering must chunk the row_id IN clause
    to avoid exceeding SQLite's SQLITE_MAX_VARIABLE_NUMBER bind limit.

    By setting _METADATA_CHUNK_SIZE=1, each row_id gets its own query.
    The filtering logic must still produce correct results across chunks.
    """
    monkeypatch.setattr("elspeth.core.checkpoint.recovery._METADATA_CHUNK_SIZE", 1)

    run_id = "run-chunk-buffered"
    graph = _create_graph(node_id="checkpoint-node")
    with db.connection() as conn:
        _insert_run(conn, run_id, status=RunStatus.FAILED, with_contract=True)
        _insert_node(conn, run_id, "checkpoint-node")

        # row-a: incomplete token is buffered (journal BLOCKED) -> excluded
        _insert_row(conn, run_id, "row-a", row_index=0, source_data_ref=None)
        _insert_token(conn, run_id, "tok-a", "row-a")
        _insert_blocked_work_item(conn, run_id, "tok-a", "row-a", barrier_key="agg-node")

        # row-b: incomplete token is NOT buffered -> included
        _insert_row(conn, run_id, "row-b", row_index=1, source_data_ref=None)
        _insert_token(conn, run_id, "tok-b", "row-b")

        # row-c: incomplete token is buffered (journal BLOCKED) -> excluded
        _insert_row(conn, run_id, "row-c", row_index=2, source_data_ref=None)
        _insert_token(conn, run_id, "tok-c", "row-c")
        _insert_blocked_work_item(conn, run_id, "tok-c", "row-c", barrier_key="agg-node")

    _create_checkpoint(
        checkpoint_manager,
        run_id=run_id,
        sequence_number=1,
        barrier_scalars=None,
        graph=graph,
    )

    unprocessed = recovery_manager.get_unprocessed_rows(run_id)
    # row-a and row-c excluded (buffered), row-b included
    assert unprocessed == ["row-b"]


def test_get_unprocessed_rows_excludes_coalesce_buffered_rows(
    db: LandscapeDB,
    checkpoint_manager: CheckpointManager,
    recovery_manager: RecoveryManager,
) -> None:
    """Rows whose only incomplete tokens are held at a coalesce barrier are excluded."""
    run_id = "run-coalesce-buffered"
    graph = _create_graph(node_id="checkpoint-node")
    with db.connection() as conn:
        _insert_run(conn, run_id, status=RunStatus.FAILED, with_contract=True)
        _insert_node(conn, run_id, "checkpoint-node")

        # row-coalesce-buffered: incomplete token held at a coalesce barrier
        # (journal BLOCKED row, barrier_key = coalesce name) -> excluded
        _insert_row(conn, run_id, "row-coalesce-buffered", row_index=0, source_data_ref=None)
        _insert_token(conn, run_id, "tok-coalesce-buf", "row-coalesce-buffered")
        _insert_blocked_work_item(conn, run_id, "tok-coalesce-buf", "row-coalesce-buffered", barrier_key="merge1")

        # row-pending: incomplete token NOT buffered anywhere -> included
        _insert_row(conn, run_id, "row-pending", row_index=1, source_data_ref=None)
        _insert_token(conn, run_id, "tok-pending", "row-pending")

    _create_checkpoint(
        checkpoint_manager,
        run_id=run_id,
        sequence_number=1,
        barrier_scalars=None,
        graph=graph,
    )

    unprocessed = recovery_manager.get_unprocessed_rows(run_id)
    assert unprocessed == ["row-pending"]


def test_get_unprocessed_rows_combines_aggregation_and_coalesce_buffered(
    db: LandscapeDB,
    checkpoint_manager: CheckpointManager,
    recovery_manager: RecoveryManager,
) -> None:
    """BLOCKED barrier holds for aggregation nodes and coalesce names both exclude."""
    run_id = "run-combined-buffered"
    graph = _create_graph(node_id="checkpoint-node")
    with db.connection() as conn:
        _insert_run(conn, run_id, status=RunStatus.FAILED, with_contract=True)
        _insert_node(conn, run_id, "checkpoint-node")

        # row-agg: held at an aggregation barrier (barrier_key = node_id) -> excluded
        _insert_row(conn, run_id, "row-agg", row_index=0, source_data_ref=None)
        _insert_token(conn, run_id, "tok-agg", "row-agg")
        _insert_blocked_work_item(conn, run_id, "tok-agg", "row-agg", barrier_key="agg-node")

        # row-coal: held at a coalesce barrier (barrier_key = coalesce name) -> excluded
        _insert_row(conn, run_id, "row-coal", row_index=1, source_data_ref=None)
        _insert_token(conn, run_id, "tok-coal", "row-coal")
        _insert_blocked_work_item(conn, run_id, "tok-coal", "row-coal", barrier_key="merge1")

        # row-free: not buffered -> included
        _insert_row(conn, run_id, "row-free", row_index=2, source_data_ref=None)
        _insert_token(conn, run_id, "tok-free", "row-free")

    _create_checkpoint(
        checkpoint_manager,
        run_id=run_id,
        sequence_number=1,
        barrier_scalars=None,
        graph=graph,
    )

    unprocessed = recovery_manager.get_unprocessed_rows(run_id)
    assert unprocessed == ["row-free"]


def test_get_unprocessed_rows_coalesce_multi_branch_collects_all_tokens(
    db: LandscapeDB,
    checkpoint_manager: CheckpointManager,
    recovery_manager: RecoveryManager,
) -> None:
    """All branch tokens held at the same coalesce barrier are recognized as buffered."""
    run_id = "run-coalesce-multi"
    graph = _create_graph(node_id="checkpoint-node")
    with db.connection() as conn:
        _insert_run(conn, run_id, status=RunStatus.FAILED, with_contract=True)
        _insert_node(conn, run_id, "checkpoint-node")

        # row with two tokens, both held at the coalesce barrier -> excluded
        _insert_row(conn, run_id, "row-multi", row_index=0, source_data_ref=None)
        _insert_token(conn, run_id, "tok-branch-a", "row-multi")
        _insert_token(conn, run_id, "tok-branch-b", "row-multi")
        _insert_blocked_work_item(conn, run_id, "tok-branch-a", "row-multi", barrier_key="merge1")
        _insert_blocked_work_item(conn, run_id, "tok-branch-b", "row-multi", barrier_key="merge1")

    _create_checkpoint(
        checkpoint_manager,
        run_id=run_id,
        sequence_number=1,
        barrier_scalars=None,
        graph=graph,
    )

    unprocessed = recovery_manager.get_unprocessed_rows(run_id)
    assert unprocessed == []


def test_buffered_exclusion_reads_journal_not_blob(
    db: LandscapeDB,
    checkpoint_manager: CheckpointManager,
    recovery_manager: RecoveryManager,
) -> None:
    """F1 Task 3.2: the exclusion set is sourced from journal BLOCKED rows.

    Token t1 is BLOCKED at a barrier in the journal and incomplete in
    token_outcomes -> its row r1 is NOT in get_unprocessed_rows (buffered
    work is restored at processor construction, not re-driven). The
    checkpoint row carries no buffered-token blob at all.
    """
    run_id = "run-journal-exclusion"
    graph = _create_graph(node_id="checkpoint-node")
    with db.connection() as conn:
        _insert_run(conn, run_id, status=RunStatus.FAILED, with_contract=True)
        _insert_node(conn, run_id, "checkpoint-node")
        _insert_row(conn, run_id, "r1", row_index=0, source_data_ref=None)
        _insert_token(conn, run_id, "t1", "r1")
        _insert_blocked_work_item(conn, run_id, "t1", "r1", barrier_key="agg-node")

    _create_checkpoint(
        checkpoint_manager,
        run_id=run_id,
        sequence_number=1,
        barrier_scalars=None,
        graph=graph,
    )

    assert "r1" not in set(recovery_manager.get_unprocessed_rows(run_id))


def test_queue_held_token_stays_in_redrive_work_set(
    db: LandscapeDB,
    checkpoint_manager: CheckpointManager,
    recovery_manager: RecoveryManager,
) -> None:
    """ADR-028 queue-holds (BLOCKED, barrier_key NULL) are NOT barrier-buffered.

    A queue-held token is throttled, not waiting at a barrier — nothing
    restores it into an executor buffer on resume, so excluding it from the
    re-drive work set would silently drop the row.
    """
    run_id = "run-queue-hold"
    graph = _create_graph(node_id="checkpoint-node")
    with db.connection() as conn:
        _insert_run(conn, run_id, status=RunStatus.FAILED, with_contract=True)
        _insert_node(conn, run_id, "checkpoint-node")
        _insert_row(conn, run_id, "row-queued", row_index=0, source_data_ref=None)
        _insert_token(conn, run_id, "tok-queued", "row-queued")
        _insert_blocked_work_item(conn, run_id, "tok-queued", "row-queued", barrier_key=None, queue_key="llm-rate-limit")

    _create_checkpoint(
        checkpoint_manager,
        run_id=run_id,
        sequence_number=1,
        barrier_scalars=None,
        graph=graph,
    )

    assert "row-queued" in recovery_manager.get_unprocessed_rows(run_id)
    incomplete = recovery_manager.get_incomplete_tokens_by_row(run_id)
    assert "tok-queued" in {spec.token_id for spec in incomplete.get("row-queued", [])}


def test_post_checkpoint_buffered_token_is_restored_not_redriven(
    db: LandscapeDB,
    checkpoint_manager: CheckpointManager,
    recovery_manager: RecoveryManager,
) -> None:
    """A BLOCKED journal row written AFTER the latest checkpoint still excludes.

    The checkpoint contributes only scalars (a resume baseline); journal
    BLOCKED rows are authoritative for buffered membership regardless of
    whether they pre- or post-date the checkpoint sequence.
    """
    run_id = "run-post-checkpoint-buffer"
    graph = _create_graph(node_id="checkpoint-node")
    with db.connection() as conn:
        _insert_run(conn, run_id, status=RunStatus.FAILED, with_contract=True)
        _insert_node(conn, run_id, "checkpoint-node")
        _insert_row(conn, run_id, "row-late-buffer", row_index=0, source_data_ref=None)
        _insert_token(conn, run_id, "tok-late-buffer", "row-late-buffer")

    # Checkpoint (the resume baseline, with scalars) predates the BLOCKED row.
    _create_checkpoint(
        checkpoint_manager,
        run_id=run_id,
        sequence_number=1,
        barrier_scalars=BarrierScalars(
            aggregation={"agg-node": AggregationNodeScalars(count_fire_offset=None, condition_fire_offset=None)},
            coalesce={},
        ),
        graph=graph,
    )

    with db.connection() as conn:
        _insert_blocked_work_item(conn, run_id, "tok-late-buffer", "row-late-buffer", barrier_key="agg-node")

    assert recovery_manager.get_unprocessed_rows(run_id) == []
    assert recovery_manager.get_incomplete_tokens_by_row(run_id) == {}


def test_get_incomplete_tokens_by_row_excludes_journal_blocked_tokens(
    db: LandscapeDB,
    checkpoint_manager: CheckpointManager,
    recovery_manager: RecoveryManager,
) -> None:
    """The mid-DAG continuation index must not re-drive barrier-held tokens."""
    run_id = "run-incomplete-journal"
    graph = _create_graph(node_id="checkpoint-node")
    with db.connection() as conn:
        _insert_run(conn, run_id, status=RunStatus.FAILED, with_contract=True)
        _insert_node(conn, run_id, "checkpoint-node")
        _insert_row(conn, run_id, "row-mixed", row_index=0, source_data_ref=None)
        _insert_token(conn, run_id, "tok-held", "row-mixed")
        _insert_token(conn, run_id, "tok-live", "row-mixed")
        _insert_blocked_work_item(conn, run_id, "tok-held", "row-mixed", barrier_key="merge1")

    _create_checkpoint(
        checkpoint_manager,
        run_id=run_id,
        sequence_number=1,
        barrier_scalars=None,
        graph=graph,
    )

    by_row = recovery_manager.get_incomplete_tokens_by_row(run_id)
    token_ids = {spec.token_id for spec in by_row.get("row-mixed", [])}
    assert "tok-held" not in token_ids
    assert "tok-live" in token_ids


def test_count_blocked_barrier_items_counts_only_barrier_holds(
    db: LandscapeDB,
    recovery_manager: RecoveryManager,
) -> None:
    """Public journal-count surface: barrier holds in, queue-holds and READY out.

    Consumed by the resume coordinator's quiescence gate and the CLI resume
    preflight (the inline CLI query was absorbed here in F1 Task 3.2).
    """
    run_id = "run-blocked-count"
    with db.connection() as conn:
        _insert_run(conn, run_id, status=RunStatus.FAILED, with_contract=True)
        _insert_row(conn, run_id, "row-1", row_index=0, source_data_ref=None)
        _insert_token(conn, run_id, "tok-barrier-1", "row-1")
        _insert_token(conn, run_id, "tok-barrier-2", "row-1")
        _insert_token(conn, run_id, "tok-queue", "row-1")
        _insert_token(conn, run_id, "tok-ready", "row-1")
        _insert_blocked_work_item(conn, run_id, "tok-barrier-1", "row-1", barrier_key="agg-node")
        _insert_blocked_work_item(conn, run_id, "tok-barrier-2", "row-1", barrier_key="merge1")
        _insert_blocked_work_item(conn, run_id, "tok-queue", "row-1", barrier_key=None, queue_key="llm-rate-limit")
        _insert_blocked_work_item(conn, run_id, "tok-ready", "row-1", barrier_key=None, status=TokenWorkStatus.READY)

    assert recovery_manager.count_blocked_barrier_items(run_id) == 2
    assert recovery_manager.count_blocked_barrier_items("other-run") == 0


class _SimpleSchema(PluginSchema):
    model_config = ConfigDict(strict=False)
    id: int


class _EmptySchema(PluginSchema):
    model_config = ConfigDict(strict=False)


def test_get_unprocessed_row_data_returns_empty_when_no_rows(
    recovery_manager: RecoveryManager,
    payload_store: PayloadStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(recovery_manager, "get_unprocessed_rows", lambda _run_id: [])
    assert recovery_manager.get_unprocessed_row_data("run", payload_store, source_schema_class=_SimpleSchema) == []


def test_get_unprocessed_row_data_errors_when_row_missing_from_metadata(
    recovery_manager: RecoveryManager,
    payload_store: PayloadStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(recovery_manager, "get_unprocessed_rows", lambda _run_id: ["row-missing"])
    with pytest.raises(AuditIntegrityError, match="Row row-missing not found in database"):
        recovery_manager.get_unprocessed_row_data("run", payload_store, source_schema_class=_SimpleSchema)


def test_get_unprocessed_row_data_errors_on_missing_source_data_ref(
    db: LandscapeDB,
    recovery_manager: RecoveryManager,
    payload_store: PayloadStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with db.connection() as conn:
        _insert_run(conn, "run-meta", status=RunStatus.FAILED)
        _insert_node(conn, "run-meta", "source-node", node_type=NodeType.SOURCE)
        _insert_row(conn, "run-meta", "row-1", row_index=1, source_data_ref=None)

    monkeypatch.setattr(recovery_manager, "get_unprocessed_rows", lambda _run_id: ["row-1"])
    with pytest.raises(ValueError, match="has no source_data_ref"):
        recovery_manager.get_unprocessed_row_data("run-meta", payload_store, source_schema_class=_SimpleSchema)


def test_get_unprocessed_row_data_errors_when_payload_purged(
    db: LandscapeDB,
    recovery_manager: RecoveryManager,
    payload_store: PayloadStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_ref = "f" * 64
    with db.connection() as conn:
        _insert_run(conn, "run-purged", status=RunStatus.FAILED)
        _insert_node(conn, "run-purged", "source-node", node_type=NodeType.SOURCE)
        _insert_row(conn, "run-purged", "row-1", row_index=1, source_data_ref=missing_ref)

    monkeypatch.setattr(recovery_manager, "get_unprocessed_rows", lambda _run_id: ["row-1"])
    with pytest.raises(ValueError, match="payload has been purged"):
        recovery_manager.get_unprocessed_row_data("run-purged", payload_store, source_schema_class=_SimpleSchema)


def test_get_unprocessed_row_data_errors_when_schema_discards_all_fields(
    db: LandscapeDB,
    recovery_manager: RecoveryManager,
    payload_store: PayloadStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload_ref = payload_store.store(b'{"id": 123}')
    with db.connection() as conn:
        _insert_run(conn, "run-empty-schema", status=RunStatus.FAILED)
        _insert_node(conn, "run-empty-schema", "source-node", node_type=NodeType.SOURCE)
        _insert_row(conn, "run-empty-schema", "row-1", row_index=1, source_data_ref=payload_ref)

    monkeypatch.setattr(recovery_manager, "get_unprocessed_rows", lambda _run_id: ["row-1"])
    with pytest.raises(ValueError, match="Schema validation returned empty data"):
        recovery_manager.get_unprocessed_row_data("run-empty-schema", payload_store, source_schema_class=_EmptySchema)


def test_get_unprocessed_row_data_chunked_lookup_and_type_restoration(
    db: LandscapeDB,
    recovery_manager: RecoveryManager,
    payload_store: PayloadStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload_ref_a = payload_store.store(b'{"id": "1"}')
    payload_ref_b = payload_store.store(b'{"id": "2"}')
    with db.connection() as conn:
        _insert_run(conn, "run-chunked", status=RunStatus.FAILED)
        _insert_node(conn, "run-chunked", "source-node", node_type=NodeType.SOURCE)
        _insert_row(conn, "run-chunked", "row-a", row_index=2, source_data_ref=payload_ref_a)
        _insert_row(conn, "run-chunked", "row-b", row_index=5, source_data_ref=payload_ref_b)

    monkeypatch.setattr(recovery_manager, "get_unprocessed_rows", lambda _run_id: ["row-a", "row-b"])
    monkeypatch.setattr("elspeth.core.checkpoint.recovery._METADATA_CHUNK_SIZE", 1)

    rows = recovery_manager.get_unprocessed_row_data("run-chunked", payload_store, source_schema_class=_SimpleSchema)
    assert rows == [
        ResumedRow(row_id="row-a", row_index=2, source_node_id=NodeID("source-node"), row_data={"id": 1}),
        ResumedRow(row_id="row-b", row_index=5, source_node_id=NodeID("source-node"), row_data={"id": 2}),
    ]


def test_verify_contract_integrity_returns_contract(
    db: LandscapeDB,
    recovery_manager: RecoveryManager,
) -> None:
    with db.connection() as conn:
        _insert_run(conn, "run-contract-ok", status=RunStatus.FAILED, with_contract=True)

    contract = recovery_manager.verify_contract_integrity("run-contract-ok")
    assert isinstance(contract, SchemaContract)
    assert contract.mode == "FIXED"
    assert len(contract.fields) == 1


def test_verify_contract_integrity_raises_empty_resume_state_when_no_sources_recorded(
    db: LandscapeDB,
    recovery_manager: RecoveryManager,
) -> None:
    """Per ADR-025 §3 (elspeth-241608388f), absence of ``run_sources`` rows
    is the interpretable ``EmptyResumeStateError`` ("nothing to resume")
    case — NOT ``CheckpointCorruptionError``. The previous mapping caused
    the CLI's outer ``try`` (which has no audit-corruption handler) to
    bubble an unhandled traceback for what should be a clean exit-1
    "this run is not resumable" message.
    """
    with db.connection() as conn:
        _insert_run(conn, "run-contract-missing", status=RunStatus.FAILED, with_contract=False)

    with pytest.raises(EmptyResumeStateError) as exc_info:
        recovery_manager.verify_contract_integrity("run-contract-missing")
    assert exc_info.value.run_id == "run-contract-missing"
    # Subclass relationship is load-bearing: every ``except OrchestrationInvariantError``
    # catch must still match this exception so callers without explicit
    # EmptyResumeStateError handling do not silently miss it.
    assert isinstance(exc_info.value, OrchestrationInvariantError)


def test_verify_contract_integrity_raises_on_hash_mismatch(
    db: LandscapeDB,
    recovery_manager: RecoveryManager,
) -> None:
    valid_contract_json, _ = _create_contract()
    tampered = valid_contract_json.replace('"version_hash":"', '"version_hash":"deadbeef')
    with db.connection() as conn:
        _insert_run(
            conn,
            "run-contract-bad-hash",
            status=RunStatus.FAILED,
            contract_json_override=tampered,
        )

    with pytest.raises(CheckpointCorruptionError, match="Contract integrity verification failed"):
        recovery_manager.verify_contract_integrity("run-contract-bad-hash")


def test_verify_contract_integrity_raises_on_malformed_json(
    db: LandscapeDB,
    recovery_manager: RecoveryManager,
) -> None:
    """Malformed contract JSON must raise CheckpointCorruptionError, not raw JSONDecodeError.

    When schema_contract_json is garbage (not valid JSON), the recorder's
    get_run_contract will raise json.JSONDecodeError (via ContractAuditRecord.from_json).
    verify_contract_integrity must catch this and wrap it as CheckpointCorruptionError
    for consistent corruption handling.
    """
    with db.connection() as conn:
        _insert_run(
            conn,
            "run-contract-malformed",
            status=RunStatus.FAILED,
            contract_json_override="not valid json {{{",
        )

    with pytest.raises(CheckpointCorruptionError, match="Contract integrity verification failed"):
        recovery_manager.verify_contract_integrity("run-contract-malformed")


def test_verify_contract_integrity_raises_on_missing_keys(
    db: LandscapeDB,
    recovery_manager: RecoveryManager,
) -> None:
    """Contract JSON missing required keys must raise CheckpointCorruptionError.

    If the stored JSON is valid but missing 'mode' or 'fields', the KeyError
    from ContractAuditRecord.from_json must be wrapped as CheckpointCorruptionError.
    """
    with db.connection() as conn:
        _insert_run(
            conn,
            "run-contract-missing-keys",
            status=RunStatus.FAILED,
            contract_json_override='{"unexpected": "schema"}',
        )

    with pytest.raises(CheckpointCorruptionError, match="Contract integrity verification failed"):
        recovery_manager.verify_contract_integrity("run-contract-missing-keys")


def test_get_run_private_helper_returns_none_for_missing_run(recovery_manager: RecoveryManager) -> None:
    assert recovery_manager._get_run("missing-run") is None


def test_get_run_private_helper_returns_row_for_existing_run(
    db: LandscapeDB,
    recovery_manager: RecoveryManager,
) -> None:
    with db.connection() as conn:
        _insert_run(conn, "run-present", status=RunStatus.FAILED)

    row = recovery_manager._get_run("run-present")
    assert row is not None
    assert row.run_id == "run-present"


def test_get_unprocessed_rows_returns_empty_when_checkpoint_manager_returns_none(
    db: LandscapeDB,
    recovery_manager: RecoveryManager,
) -> None:
    with db.connection() as conn:
        _insert_run(conn, "run-without-cp", status=RunStatus.FAILED)

    assert recovery_manager.get_unprocessed_rows("run-without-cp") == []


def test_get_unprocessed_rows_handles_delegation_token_with_completed_leaf(
    db: LandscapeDB,
    checkpoint_manager: CheckpointManager,
    recovery_manager: RecoveryManager,
) -> None:
    run_id = "run-fork-complete"
    graph = _create_graph(node_id="checkpoint-node")
    with db.connection() as conn:
        _insert_run(conn, run_id, status=RunStatus.FAILED, with_contract=True)
        _insert_node(conn, run_id, "checkpoint-node")
        _insert_row(conn, run_id, "row-forked-complete", row_index=1, source_data_ref=None)
        _insert_token(conn, run_id, "tok-parent", "row-forked-complete")
        _insert_terminal_outcome(conn, run_id, "tok-parent", outcome=TerminalOutcome.TRANSIENT, path=TerminalPath.FORK_PARENT)
        _insert_token(conn, run_id, "tok-child", "row-forked-complete")
        _insert_terminal_outcome(conn, run_id, "tok-child", outcome=TerminalOutcome.SUCCESS, path=TerminalPath.DEFAULT_FLOW)

    _create_checkpoint(
        checkpoint_manager,
        run_id=run_id,
        sequence_number=1,
        barrier_scalars=None,
        graph=graph,
    )

    assert recovery_manager.get_unprocessed_rows(run_id) == []


def test_get_unprocessed_row_data_corrupt_utf8_raises_audit_integrity(
    db: LandscapeDB,
    recovery_manager: RecoveryManager,
    payload_store: PayloadStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Corrupt UTF-8 payload must raise AuditIntegrityError, not UnicodeDecodeError.

    Regression: If persisted payload bytes are not valid UTF-8 (e.g. disk corruption,
    encoding mismatch), get_unprocessed_row_data must raise AuditIntegrityError
    with a clear message rather than leaking a raw UnicodeDecodeError.
    """
    corrupt_bytes = b"\xff\xfe invalid"
    payload_ref = payload_store.store(corrupt_bytes)
    with db.connection() as conn:
        _insert_run(conn, "run-corrupt-utf8", status=RunStatus.FAILED)
        _insert_node(conn, "run-corrupt-utf8", "source-node", node_type=NodeType.SOURCE)
        _insert_row(conn, "run-corrupt-utf8", "row-1", row_index=0, source_data_ref=payload_ref)

    monkeypatch.setattr(recovery_manager, "get_unprocessed_rows", lambda _run_id: ["row-1"])
    with pytest.raises(AuditIntegrityError, match="Corrupt payload"):
        recovery_manager.get_unprocessed_row_data("run-corrupt-utf8", payload_store, source_schema_class=_SimpleSchema)


def test_get_unprocessed_row_data_non_dict_json_raises_audit_integrity(
    db: LandscapeDB,
    recovery_manager: RecoveryManager,
    payload_store: PayloadStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """JSON payload that decodes to non-dict must raise AuditIntegrityError.

    Regression: If persisted payload is valid JSON but not a dict (e.g. a list),
    get_unprocessed_row_data must raise AuditIntegrityError with "expected dict"
    rather than silently passing a list where a dict is expected.
    """
    non_dict_bytes = b"[1, 2, 3]"
    payload_ref = payload_store.store(non_dict_bytes)
    with db.connection() as conn:
        _insert_run(conn, "run-non-dict", status=RunStatus.FAILED)
        _insert_node(conn, "run-non-dict", "source-node", node_type=NodeType.SOURCE)
        _insert_row(conn, "run-non-dict", "row-1", row_index=0, source_data_ref=payload_ref)

    monkeypatch.setattr(recovery_manager, "get_unprocessed_rows", lambda _run_id: ["row-1"])
    with pytest.raises(AuditIntegrityError, match="expected dict"):
        recovery_manager.get_unprocessed_row_data("run-non-dict", payload_store, source_schema_class=_SimpleSchema)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_get_unprocessed_row_data_rejects_non_finite_json_constant(
    db: LandscapeDB,
    recovery_manager: RecoveryManager,
    payload_store: PayloadStore,
    monkeypatch: pytest.MonkeyPatch,
    constant: str,
) -> None:
    payload_ref = payload_store.store(f'{{"id": {constant}}}'.encode())
    with db.connection() as conn:
        _insert_run(conn, f"run-non-finite-{constant}", status=RunStatus.FAILED)
        _insert_node(conn, f"run-non-finite-{constant}", "source-node", node_type=NodeType.SOURCE)
        _insert_row(conn, f"run-non-finite-{constant}", "row-1", row_index=0, source_data_ref=payload_ref)

    monkeypatch.setattr(recovery_manager, "get_unprocessed_rows", lambda _run_id: ["row-1"])
    with pytest.raises(AuditIntegrityError, match="non-finite JSON constant"):
        recovery_manager.get_unprocessed_row_data(
            f"run-non-finite-{constant}",
            payload_store,
            source_schema_class=_SimpleSchema,
        )


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_get_unprocessed_row_data_by_source_rejects_non_finite_json_constant(
    db: LandscapeDB,
    recovery_manager: RecoveryManager,
    payload_store: PayloadStore,
    monkeypatch: pytest.MonkeyPatch,
    constant: str,
) -> None:
    payload_ref = payload_store.store(f'{{"id": {constant}}}'.encode())
    with db.connection() as conn:
        _insert_run(conn, f"run-by-source-non-finite-{constant}", status=RunStatus.FAILED)
        _insert_node(conn, f"run-by-source-non-finite-{constant}", "source-node", node_type=NodeType.SOURCE)
        _insert_row(conn, f"run-by-source-non-finite-{constant}", "row-1", row_index=0, source_data_ref=payload_ref)

    monkeypatch.setattr(recovery_manager, "get_unprocessed_rows", lambda _run_id: ["row-1"])
    with pytest.raises(AuditIntegrityError, match="non-finite JSON constant"):
        recovery_manager.get_unprocessed_row_data_by_source(
            f"run-by-source-non-finite-{constant}",
            payload_store,
            source_schema_classes={NodeID("source-node"): _SimpleSchema},
        )


def test_get_resume_point_reads_latest_checkpoint_after_can_resume(
    db: LandscapeDB,
    checkpoint_manager: CheckpointManager,
    recovery_manager: RecoveryManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "run-latest-checkpoint"
    graph = _create_failed_run_with_checkpoint(db, checkpoint_manager, run_id)
    _create_checkpoint(
        checkpoint_manager,
        run_id=run_id,
        sequence_number=99,
        barrier_scalars=None,
        graph=graph,
    )

    # Force can_resume to succeed so we exercise the second get_latest_checkpoint call path.
    monkeypatch.setattr(recovery_manager, "can_resume", lambda _run_id, _graph: type("Check", (), {"can_resume": True})())
    point = recovery_manager.get_resume_point(run_id, graph)

    assert point is not None
    assert point.sequence_number == 99


def test_get_resume_point_revalidates_checkpoint_loaded_after_can_resume(
    db: LandscapeDB,
    checkpoint_manager: CheckpointManager,
    recovery_manager: RecoveryManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "run-latest-checkpoint-revalidate"
    graph = _create_failed_run_with_checkpoint(db, checkpoint_manager, run_id)
    with db.connection() as conn:
        conn.execute(
            checkpoints_table.insert().values(
                checkpoint_id="cp-later-incompatible",
                run_id=run_id,
                sequence_number=99,
                barrier_scalars_json=None,
                created_at=datetime.now(UTC),
                upstream_topology_hash="x" * 64,
                format_version=Checkpoint.CURRENT_FORMAT_VERSION,
            )
        )

    # Simulate a checkpoint appearing after can_resume validated an earlier checkpoint.
    monkeypatch.setattr(recovery_manager, "can_resume", lambda _run_id, _graph: type("Check", (), {"can_resume": True})())

    assert recovery_manager.get_resume_point(run_id, graph) is None


def test_get_unprocessed_rows_excludes_diverted_rows(
    db: LandscapeDB,
    checkpoint_manager: CheckpointManager,
    recovery_manager: RecoveryManager,
) -> None:
    """DIVERTED rows are terminal — they must not be requeued on resume.

    Regression test for elspeth-46b30e2917: the manual terminal_outcome_values
    list omitted DIVERTED, causing diverted rows to appear as 'incomplete'
    during recovery and get reprocessed (duplicate side effects).
    """
    run_id = "run-diverted-terminal"
    graph = _create_graph(node_id="checkpoint-node")
    with db.connection() as conn:
        _insert_run(conn, run_id, status=RunStatus.FAILED, with_contract=True)
        _insert_node(conn, run_id, "checkpoint-node")

        # row-diverted: one token diverted to failsink -> terminal, exclude.
        _insert_row(conn, run_id, "row-diverted", row_index=0, source_data_ref=None)
        _insert_token(conn, run_id, "tok-diverted", "row-diverted")
        _insert_terminal_outcome(
            conn, run_id, "tok-diverted", outcome=TerminalOutcome.TRANSIENT, path=TerminalPath.SINK_FALLBACK_TO_FAILSINK
        )

        # row-pending: no terminal outcome -> should be included.
        _insert_row(conn, run_id, "row-pending", row_index=1, source_data_ref=None)
        _insert_token(conn, run_id, "tok-pending", "row-pending")

    _create_checkpoint(
        checkpoint_manager,
        run_id=run_id,
        sequence_number=1,
        barrier_scalars=None,
        graph=graph,
    )

    unprocessed = recovery_manager.get_unprocessed_rows(run_id)

    assert "row-diverted" not in unprocessed, "DIVERTED row should be excluded — it is terminal"
    assert "row-pending" in unprocessed


# ── IncompleteTokenSpec construction-time identity validation ───────────────
# IncompleteTokenSpec is a Tier-1-sourced identity type (built directly from
# tokens_table columns) that reaches reconstruct_token_row BEFORE any TokenInfo
# guard fires. NOT NULL (the DB constraint) is not the same as non-empty, so an
# empty-string identity could produce valid-looking but meaningless audit work.
# Mirrors TokenInfo.__post_init__ (contracts/identity.py) — see tests/unit/
# contracts/test_identity.py for the sibling pattern.


def _valid_incomplete_token_spec_kwargs() -> dict[str, Any]:
    return {
        "token_id": "tok-1",
        "row_id": "row-1",
        "branch_name": None,
        "fork_group_id": None,
        "join_group_id": None,
        "expand_group_id": None,
        "token_data_ref": None,
        "step_in_pipeline": 1,
        "max_attempt": -1,
    }


def test_incomplete_token_spec_accepts_valid_identity() -> None:
    spec = IncompleteTokenSpec(**_valid_incomplete_token_spec_kwargs())
    assert spec.token_id == "tok-1"
    assert spec.row_id == "row-1"


@pytest.mark.parametrize("field", ["token_id", "row_id"])
def test_incomplete_token_spec_rejects_empty_identity(field: str) -> None:
    kwargs = _valid_incomplete_token_spec_kwargs()
    kwargs[field] = ""
    with pytest.raises(ValueError, match=f"{field} must not be empty"):
        IncompleteTokenSpec(**kwargs)


@pytest.mark.parametrize("field", ["token_id", "row_id"])
def test_incomplete_token_spec_rejects_non_str_identity(field: str) -> None:
    kwargs = _valid_incomplete_token_spec_kwargs()
    kwargs[field] = 123
    with pytest.raises(TypeError, match=f"{field} must be str"):
        IncompleteTokenSpec(**kwargs)


@pytest.mark.parametrize("field", ["branch_name", "fork_group_id", "join_group_id", "expand_group_id", "token_data_ref"])
def test_incomplete_token_spec_rejects_empty_optional_string(field: str) -> None:
    # NULL is the legitimate "not applicable" value for these columns; an empty
    # string is anomalous. token_data_ref in particular is used as a payload-store
    # key in reconstruct_token_row before any TokenInfo exists, so "" must crash
    # here rather than surface as a misleading "payload purged" error.
    kwargs = _valid_incomplete_token_spec_kwargs()
    kwargs[field] = ""
    with pytest.raises(ValueError, match=f"{field} must be None or non-empty"):
        IncompleteTokenSpec(**kwargs)


@pytest.mark.parametrize("field", ["branch_name", "fork_group_id", "join_group_id", "expand_group_id", "token_data_ref"])
def test_incomplete_token_spec_accepts_none_optional_string(field: str) -> None:
    kwargs = _valid_incomplete_token_spec_kwargs()
    kwargs[field] = None
    spec = IncompleteTokenSpec(**kwargs)
    assert getattr(spec, field) is None
