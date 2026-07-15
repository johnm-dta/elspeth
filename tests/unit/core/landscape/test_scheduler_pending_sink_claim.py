"""Fail-closed admission tests for durable PENDING_SINK work."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import event, insert, select, update

from elspeth.contracts import NodeType, TerminalOutcome, TerminalPath
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.scheduler import TokenWorkStatus
from elspeth.contracts.schema_contract import PipelineRow, SchemaContract
from elspeth.core.landscape.database import LandscapeDB, Tier1Engine
from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository
from elspeth.core.landscape.schema import (
    nodes_table,
    rows_table,
    runs_table,
    scheduler_events_table,
    token_work_items_table,
    tokens_table,
)

RUN_ID = "run-pending-sink-admission"
NOW = datetime(2026, 7, 16, 9, 0, tzinfo=UTC)


@pytest.fixture
def pending_sink() -> Iterator[tuple[Tier1Engine, TokenSchedulerRepository, str, str]]:
    db = LandscapeDB.in_memory()
    repo = TokenSchedulerRepository(db.engine)
    payload = _seed_prerequisites(db.engine)
    work_item_id = _seed_pending_sink(repo, payload=payload)
    try:
        yield db.engine, repo, work_item_id, payload
    finally:
        db.close()


@pytest.mark.parametrize(
    "corrupt_values",
    [
        pytest.param({"row_payload_json": ""}, id="empty-row-payload"),
        pytest.param({"pending_sink_name": None}, id="missing-sink-name"),
        pytest.param({"pending_sink_name": ""}, id="empty-sink-name"),
        pytest.param({"pending_outcome": None}, id="missing-outcome"),
        pytest.param({"pending_outcome": "unknown"}, id="unknown-outcome"),
        pytest.param({"pending_path": None}, id="missing-path"),
        pytest.param({"pending_path": "unknown"}, id="unknown-path"),
        pytest.param(
            {
                "pending_outcome": TerminalOutcome.FAILURE.value,
                "pending_path": TerminalPath.DEFAULT_FLOW.value,
            },
            id="inconsistent-outcome-path",
        ),
        pytest.param(
            {
                "pending_outcome": TerminalOutcome.FAILURE.value,
                "pending_path": TerminalPath.ON_ERROR_ROUTED.value,
                "pending_error_hash": None,
                "pending_error_message": "boom",
            },
            id="on-error-missing-error-hash",
        ),
        pytest.param(
            {
                "pending_outcome": TerminalOutcome.FAILURE.value,
                "pending_path": TerminalPath.ON_ERROR_ROUTED.value,
                "pending_error_hash": "a" * 64,
                "pending_error_message": None,
            },
            id="on-error-missing-error-message",
        ),
        pytest.param(
            {
                "pending_error_hash": "a" * 64,
                "pending_error_message": "stale error",
            },
            id="success-with-error-evidence",
        ),
        pytest.param(
            {
                "pending_path": TerminalPath.COALESCED.value,
                "join_group_id": None,
            },
            id="coalesced-missing-join-identity",
        ),
    ],
)
def test_claim_pending_sink_rejects_incomplete_bundle_without_mutation(
    pending_sink: tuple[Tier1Engine, TokenSchedulerRepository, str, str],
    corrupt_values: dict[str, object],
) -> None:
    engine, repo, work_item_id, _payload = pending_sink
    with engine.begin() as conn:
        conn.execute(update(token_work_items_table).where(token_work_items_table.c.work_item_id == work_item_id).values(**corrupt_values))
    before = _durable_image(engine, work_item_id)

    with pytest.raises(AuditIntegrityError, match="complete durable sink bundle"):
        repo.claim_pending_sink(run_id=RUN_ID, lease_owner="redrive-worker", lease_seconds=30, now=NOW + timedelta(seconds=3))

    assert _durable_image(engine, work_item_id) == before


@pytest.mark.parametrize(
    "bundle_values",
    [
        pytest.param(
            {
                "pending_outcome": TerminalOutcome.SUCCESS.value,
                "pending_path": TerminalPath.DEFAULT_FLOW.value,
                "pending_error_hash": None,
                "pending_error_message": None,
            },
            id="default-flow",
        ),
        pytest.param(
            {
                "pending_outcome": TerminalOutcome.SUCCESS.value,
                "pending_path": TerminalPath.GATE_ROUTED.value,
                "pending_error_hash": None,
                "pending_error_message": None,
            },
            id="gate-routed",
        ),
        pytest.param(
            {
                "pending_outcome": TerminalOutcome.FAILURE.value,
                "pending_path": TerminalPath.ON_ERROR_ROUTED.value,
                "pending_error_hash": "a" * 64,
                "pending_error_message": "boom",
            },
            id="on-error-routed",
        ),
        pytest.param(
            {
                "pending_outcome": TerminalOutcome.SUCCESS.value,
                "pending_path": TerminalPath.COALESCED.value,
                "pending_error_hash": None,
                "pending_error_message": None,
                "join_group_id": "join-1",
            },
            id="coalesced",
        ),
    ],
)
def test_claim_pending_sink_accepts_complete_legal_bundle(
    pending_sink: tuple[Tier1Engine, TokenSchedulerRepository, str, str],
    bundle_values: dict[str, object],
) -> None:
    engine, repo, work_item_id, payload = pending_sink
    with engine.begin() as conn:
        conn.execute(update(token_work_items_table).where(token_work_items_table.c.work_item_id == work_item_id).values(**bundle_values))

    claimed = repo.claim_pending_sink(
        run_id=RUN_ID,
        lease_owner="redrive-worker",
        lease_seconds=30,
        now=NOW + timedelta(seconds=3),
    )

    assert claimed is not None
    assert claimed.status is TokenWorkStatus.LEASED
    assert claimed.lease_owner == "redrive-worker"
    assert claimed.lease_expires_at == NOW + timedelta(seconds=33)
    assert claimed.work_item_id == work_item_id
    assert claimed.attempt == 1
    assert claimed.row_payload_json == payload
    assert claimed.pending_sink_name == "sink-a"
    for field_name, expected in bundle_values.items():
        assert getattr(claimed, field_name) == expected


def test_claim_pending_sink_update_rechecks_bundle_atomically(
    pending_sink: tuple[Tier1Engine, TokenSchedulerRepository, str, str],
) -> None:
    engine, repo, work_item_id, _payload = pending_sink
    before = _durable_image(engine, work_item_id)
    injected: list[bool] = []

    @event.listens_for(engine, "before_cursor_execute")
    def invalidate_bundle_between_select_and_update(conn, cursor, statement, parameters, context, executemany) -> None:
        if injected or not statement.lstrip().upper().startswith("UPDATE TOKEN_WORK_ITEMS"):
            return
        injected.append(True)
        cursor.execute(
            "UPDATE token_work_items SET pending_sink_name = NULL WHERE work_item_id = ?",
            (work_item_id,),
        )

    try:
        with pytest.raises(AuditIntegrityError, match="complete durable sink bundle"):
            repo.claim_pending_sink(run_id=RUN_ID, lease_owner="redrive-worker", lease_seconds=30, now=NOW + timedelta(seconds=3))
    finally:
        event.remove(engine, "before_cursor_execute", invalidate_bundle_between_select_and_update)

    assert injected == [True]
    # The competing mutation and the refused claim share the write transaction;
    # the invariant error must roll both back, including event evidence.
    assert _durable_image(engine, work_item_id) == before


def _seed_prerequisites(engine: Tier1Engine) -> str:
    payload = TokenSchedulerRepository.serialize_row_payload(
        PipelineRow({"id": 1}, SchemaContract(mode="OBSERVED", fields=(), locked=True))
    )
    with engine.begin() as conn:
        conn.execute(
            insert(runs_table).values(
                run_id=RUN_ID,
                started_at=NOW,
                config_hash="config",
                settings_json="{}",
                canonical_version="v1",
                status="running",
                openrouter_catalog_sha256="0" * 64,
                openrouter_catalog_source="bundled",
            )
        )
        for node_id, node_type, plugin_name in (
            ("source-0", NodeType.SOURCE, "csv"),
            ("normalize", NodeType.TRANSFORM, "identity"),
        ):
            conn.execute(
                insert(nodes_table).values(
                    run_id=RUN_ID,
                    node_id=node_id,
                    plugin_name=plugin_name,
                    node_type=node_type.value,
                    plugin_version="1.0",
                    determinism="deterministic",
                    config_hash="config",
                    config_json="{}",
                    registered_at=NOW,
                )
            )
        conn.execute(
            insert(rows_table).values(
                row_id="row-1",
                run_id=RUN_ID,
                source_node_id="source-0",
                row_index=0,
                source_row_index=0,
                ingest_sequence=0,
                source_data_hash="row-hash",
                created_at=NOW,
            )
        )
        conn.execute(insert(tokens_table).values(token_id="token-1", row_id="row-1", run_id=RUN_ID, created_at=NOW))
    return payload


def _seed_pending_sink(repo: TokenSchedulerRepository, *, payload: str) -> str:
    item = repo.enqueue_ready(
        run_id=RUN_ID,
        token_id="token-1",
        row_id="row-1",
        node_id="normalize",
        step_index=1,
        ingest_sequence=0,
        available_at=NOW,
        row_payload_json=payload,
    )
    claimed = repo.claim_ready(run_id=RUN_ID, lease_owner="producer-worker", lease_seconds=30, now=NOW + timedelta(seconds=1))
    assert claimed is not None
    repo.mark_pending_sink(
        work_item_id=item.work_item_id,
        row_payload_json=payload,
        sink_name="sink-a",
        outcome=TerminalOutcome.SUCCESS.value,
        path=TerminalPath.DEFAULT_FLOW.value,
        error_hash=None,
        error_message=None,
        now=NOW + timedelta(seconds=2),
        expected_lease_owner="producer-worker",
    )
    return item.work_item_id


def _durable_image(engine: Tier1Engine, work_item_id: str) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    with engine.connect() as conn:
        row = dict(
            conn.execute(select(token_work_items_table).where(token_work_items_table.c.work_item_id == work_item_id)).mappings().one()
        )
        events = tuple(
            dict(event_row)
            for event_row in conn.execute(
                select(scheduler_events_table)
                .where(scheduler_events_table.c.run_id == RUN_ID)
                .order_by(scheduler_events_table.c.recorded_at, scheduler_events_table.c.event_id)
            ).mappings()
        )
    return row, events
