"""Unit tests for PurgeManager retention and deletion behavior."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy import Connection

from elspeth.contracts import (
    CallStatus,
    CallType,
    Determinism,
    NodeStateStatus,
    NodeType,
    RoutingMode,
    RunStatus,
)
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.schema import (
    calls_table,
    edges_table,
    node_states_table,
    nodes_table,
    operations_table,
    routing_events_table,
    rows_table,
    runs_table,
    tokens_table,
)
from elspeth.core.retention.purge import PurgeManager
from tests.fixtures.landscape import make_landscape_db
from tests.fixtures.stores import MockPayloadStore


@pytest.fixture
def db() -> LandscapeDB:
    """Fresh in-memory database per test."""
    return make_landscape_db()


def _create_run(
    conn: Connection,
    run_id: str,
    *,
    status: RunStatus,
    completed_at: datetime | None,
) -> None:
    conn.execute(
        runs_table.insert().values(
            run_id=run_id,
            started_at=datetime.now(UTC),
            completed_at=completed_at,
            config_hash="cfg",
            settings_json="{}",
            canonical_version="sha256-rfc8785-v1",
            status=status,
            reproducibility_grade="replay_reproducible",
        )
    )


def _create_node(conn: Connection, run_id: str, node_id: str) -> None:
    conn.execute(
        nodes_table.insert().values(
            node_id=node_id,
            run_id=run_id,
            plugin_name="test",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0.0",
            determinism=Determinism.DETERMINISTIC,
            config_hash="node_cfg",
            config_json="{}",
            registered_at=datetime.now(UTC),
        )
    )


def _create_row(
    conn: Connection,
    run_id: str,
    node_id: str,
    row_id: str,
    *,
    row_index: int,
    source_data_ref: str | None,
) -> None:
    conn.execute(
        rows_table.insert().values(
            row_id=row_id,
            run_id=run_id,
            source_node_id=node_id,
            row_index=row_index,
            source_data_hash=f"hash-{row_id}",
            source_data_ref=source_data_ref,
            created_at=datetime.now(UTC),
        )
    )


def _create_token(conn: Connection, row_id: str, token_id: str) -> None:
    conn.execute(
        tokens_table.insert().values(
            token_id=token_id,
            row_id=row_id,
            created_at=datetime.now(UTC),
        )
    )


def _create_node_state(
    conn: Connection,
    *,
    state_id: str,
    token_id: str,
    run_id: str,
    node_id: str,
) -> None:
    conn.execute(
        node_states_table.insert().values(
            state_id=state_id,
            token_id=token_id,
            run_id=run_id,
            node_id=node_id,
            step_index=0,
            attempt=0,
            status=NodeStateStatus.COMPLETED,
            input_hash="in_hash",
            output_hash="out_hash",
            started_at=datetime.now(UTC),
        )
    )


def _create_operation(
    conn: Connection,
    *,
    operation_id: str,
    run_id: str,
    node_id: str,
    input_data_ref: str | None = None,
    output_data_ref: str | None = None,
) -> None:
    conn.execute(
        operations_table.insert().values(
            operation_id=operation_id,
            run_id=run_id,
            node_id=node_id,
            operation_type="sink_write",
            started_at=datetime.now(UTC),
            status="completed",
            input_data_ref=input_data_ref,
            output_data_ref=output_data_ref,
        )
    )


def _create_call_for_state(
    conn: Connection,
    *,
    call_id: str,
    state_id: str,
    request_ref: str | None,
    response_ref: str | None,
) -> None:
    conn.execute(
        calls_table.insert().values(
            call_id=call_id,
            state_id=state_id,
            operation_id=None,
            call_index=0,
            call_type=CallType.HTTP,
            status=CallStatus.SUCCESS,
            request_hash="req_hash",
            request_ref=request_ref,
            response_hash="res_hash",
            response_ref=response_ref,
            created_at=datetime.now(UTC),
        )
    )


def _create_call_for_operation(
    conn: Connection,
    *,
    call_id: str,
    operation_id: str,
    request_ref: str | None,
    response_ref: str | None,
) -> None:
    conn.execute(
        calls_table.insert().values(
            call_id=call_id,
            state_id=None,
            operation_id=operation_id,
            call_index=0,
            call_type=CallType.HTTP,
            status=CallStatus.SUCCESS,
            request_hash="req_hash",
            request_ref=request_ref,
            response_hash="res_hash",
            response_ref=response_ref,
            created_at=datetime.now(UTC),
        )
    )


def _create_edge(conn: Connection, *, edge_id: str, run_id: str, from_node: str, to_node: str) -> None:
    conn.execute(
        edges_table.insert().values(
            edge_id=edge_id,
            run_id=run_id,
            from_node_id=from_node,
            to_node_id=to_node,
            label="continue",
            default_mode=RoutingMode.MOVE,
            created_at=datetime.now(UTC),
        )
    )


def _create_routing_event(
    conn: Connection,
    *,
    event_id: str,
    state_id: str,
    edge_id: str,
    reason_ref: str | None,
) -> None:
    conn.execute(
        routing_events_table.insert().values(
            event_id=event_id,
            state_id=state_id,
            edge_id=edge_id,
            routing_group_id=f"rg-{uuid4().hex[:12]}",
            ordinal=0,
            mode=RoutingMode.MOVE,
            reason_hash="reason_hash",
            reason_ref=reason_ref,
            created_at=datetime.now(UTC),
        )
    )


class _ControlledStore(MockPayloadStore):
    """Payload store with controllable failure modes."""

    def __init__(
        self,
        *,
        fail_exists_for: set[str] | None = None,
        fail_delete_for: set[str] | None = None,
        false_delete_for: set[str] | None = None,
    ) -> None:
        super().__init__()
        self._fail_exists_for = fail_exists_for or set()
        self._fail_delete_for = fail_delete_for or set()
        self._false_delete_for = false_delete_for or set()

    def exists(self, content_hash: str) -> bool:
        if content_hash in self._fail_exists_for:
            raise OSError("exists failed")
        return super().exists(content_hash)

    def delete(self, content_hash: str) -> bool:
        if content_hash in self._fail_delete_for:
            raise OSError("delete failed")
        if content_hash in self._false_delete_for:
            return False
        return super().delete(content_hash)


class TestFindExpiredPayloadRefs:
    def test_find_expired_row_payloads_defaults_as_of_to_now(self, db: LandscapeDB) -> None:
        manager = PurgeManager(db, MockPayloadStore())
        now = datetime.now(UTC)
        old = now - timedelta(days=365)
        recent = now - timedelta(days=1)

        with db.connection() as conn:
            _create_run(conn, "run-old-default-now", status=RunStatus.COMPLETED, completed_at=old)
            _create_node(conn, "run-old-default-now", "node-old-default-now")
            _create_row(
                conn,
                "run-old-default-now",
                "node-old-default-now",
                "row-old-default-now",
                row_index=0,
                source_data_ref="ref-old-default-now",
            )

            _create_run(conn, "run-recent-default-now", status=RunStatus.COMPLETED, completed_at=recent)
            _create_node(conn, "run-recent-default-now", "node-recent-default-now")
            _create_row(
                conn,
                "run-recent-default-now",
                "node-recent-default-now",
                "row-recent-default-now",
                row_index=0,
                source_data_ref="ref-recent-default-now",
            )

        refs = set(manager.find_expired_row_payloads(retention_days=30))
        assert "ref-old-default-now" in refs
        assert "ref-recent-default-now" not in refs

    def test_find_expired_payload_refs_defaults_as_of_to_now(self, db: LandscapeDB) -> None:
        manager = PurgeManager(db, MockPayloadStore())
        now = datetime.now(UTC)
        old = now - timedelta(days=365)
        recent = now - timedelta(days=1)

        with db.connection() as conn:
            _create_run(conn, "run-old-default-now-all-refs", status=RunStatus.COMPLETED, completed_at=old)
            _create_node(conn, "run-old-default-now-all-refs", "node-old-default-now-all-refs")
            _create_row(
                conn,
                "run-old-default-now-all-refs",
                "node-old-default-now-all-refs",
                "row-old-default-now-all-refs",
                row_index=0,
                source_data_ref="ref-old-default-now-all-refs",
            )

            _create_run(conn, "run-recent-default-now-all-refs", status=RunStatus.COMPLETED, completed_at=recent)
            _create_node(conn, "run-recent-default-now-all-refs", "node-recent-default-now-all-refs")
            _create_row(
                conn,
                "run-recent-default-now-all-refs",
                "node-recent-default-now-all-refs",
                "row-recent-default-now-all-refs",
                row_index=0,
                source_data_ref="ref-recent-default-now-all-refs",
            )

        refs = set(manager.find_expired_payload_refs(retention_days=30))
        assert "ref-old-default-now-all-refs" in refs
        assert "ref-recent-default-now-all-refs" not in refs

    def test_find_expired_row_payloads_distinct_and_respects_status_and_cutoff(self, db: LandscapeDB) -> None:
        manager = PurgeManager(db, MockPayloadStore())
        now = datetime(2026, 2, 8, tzinfo=UTC)
        old = now - timedelta(days=45)
        recent = now - timedelta(days=2)

        with db.connection() as conn:
            _create_run(conn, "run-old-completed", status=RunStatus.COMPLETED, completed_at=old)
            _create_node(conn, "run-old-completed", "node-old-completed")
            _create_row(
                conn,
                "run-old-completed",
                "node-old-completed",
                "row-old-1",
                row_index=0,
                source_data_ref="ref-old-shared",
            )
            _create_row(
                conn,
                "run-old-completed",
                "node-old-completed",
                "row-old-2",
                row_index=1,
                source_data_ref="ref-old-shared",
            )

            _create_run(conn, "run-old-failed", status=RunStatus.FAILED, completed_at=old)
            _create_node(conn, "run-old-failed", "node-old-failed")
            _create_row(
                conn,
                "run-old-failed",
                "node-old-failed",
                "row-failed",
                row_index=0,
                source_data_ref="ref-old-failed",
            )

            _create_run(conn, "run-recent", status=RunStatus.COMPLETED, completed_at=recent)
            _create_node(conn, "run-recent", "node-recent")
            _create_row(
                conn,
                "run-recent",
                "node-recent",
                "row-recent",
                row_index=0,
                source_data_ref="ref-recent",
            )

            _create_run(conn, "run-running", status=RunStatus.RUNNING, completed_at=None)
            _create_node(conn, "run-running", "node-running")
            _create_row(
                conn,
                "run-running",
                "node-running",
                "row-running",
                row_index=0,
                source_data_ref="ref-running",
            )

        expired = set(manager.find_expired_row_payloads(retention_days=30, as_of=now))
        assert expired == {"ref-old-shared", "ref-old-failed"}

    def test_find_expired_payload_refs_includes_all_ref_types_and_excludes_active_shared_refs(self, db: LandscapeDB) -> None:
        manager = PurgeManager(db, MockPayloadStore())
        now = datetime(2026, 2, 8, tzinfo=UTC)
        old = now - timedelta(days=40)
        recent = now - timedelta(days=1)

        with db.connection() as conn:
            # Expired run with row/call/routing refs
            _create_run(conn, "expired-run", status=RunStatus.COMPLETED, completed_at=old)
            _create_node(conn, "expired-run", "expired-node-a")
            _create_node(conn, "expired-run", "expired-node-b")
            _create_row(
                conn,
                "expired-run",
                "expired-node-a",
                "expired-row",
                row_index=0,
                source_data_ref="ref-row-expired",
            )
            _create_token(conn, "expired-row", "tok-expired")
            _create_node_state(
                conn,
                state_id="state-expired",
                token_id="tok-expired",
                run_id="expired-run",
                node_id="expired-node-a",
            )
            _create_call_for_state(
                conn,
                call_id="call-state-expired",
                state_id="state-expired",
                request_ref="ref-state-req-expired",
                response_ref="ref-state-res-expired",
            )
            _create_operation(
                conn,
                operation_id="op-expired",
                run_id="expired-run",
                node_id="expired-node-a",
                input_data_ref="ref-op-input-expired",
                output_data_ref="ref-op-output-expired",
            )
            _create_call_for_operation(
                conn,
                call_id="call-op-expired",
                operation_id="op-expired",
                request_ref="ref-op-req-expired",
                response_ref="ref-op-res-expired",
            )
            _create_edge(
                conn,
                edge_id="edge-expired",
                run_id="expired-run",
                from_node="expired-node-a",
                to_node="expired-node-b",
            )
            _create_routing_event(
                conn,
                event_id="route-expired",
                state_id="state-expired",
                edge_id="edge-expired",
                reason_ref="ref-routing-expired",
            )

            # Active run reusing one payload ref (must be excluded)
            _create_run(conn, "active-run", status=RunStatus.COMPLETED, completed_at=recent)
            _create_node(conn, "active-run", "active-node")
            _create_row(
                conn,
                "active-run",
                "active-node",
                "active-row",
                row_index=0,
                source_data_ref="ref-op-res-expired",
            )
            _create_operation(
                conn,
                operation_id="op-active",
                run_id="active-run",
                node_id="active-node",
                input_data_ref="ref-op-input-expired",
                output_data_ref=None,
            )

        refs = set(manager.find_expired_payload_refs(retention_days=30, as_of=now))
        assert "ref-row-expired" in refs
        assert "ref-op-output-expired" in refs
        assert "ref-state-req-expired" in refs
        assert "ref-state-res-expired" in refs
        assert "ref-op-req-expired" in refs
        assert "ref-routing-expired" in refs
        assert "ref-op-input-expired" not in refs
        assert "ref-op-res-expired" not in refs

    def test_find_affected_run_ids_covers_row_state_call_op_call_and_routing_refs(self, db: LandscapeDB) -> None:
        manager = PurgeManager(db, MockPayloadStore())
        now = datetime(2026, 2, 8, tzinfo=UTC)
        old = now - timedelta(days=40)

        with db.connection() as conn:
            # Row ref run
            _create_run(conn, "run-row", status=RunStatus.COMPLETED, completed_at=old)
            _create_node(conn, "run-row", "node-row")
            _create_row(conn, "run-row", "node-row", "row-row", row_index=0, source_data_ref="ref-row")

            # State call ref run
            _create_run(conn, "run-state-call", status=RunStatus.COMPLETED, completed_at=old)
            _create_node(conn, "run-state-call", "node-state-call")
            _create_row(
                conn,
                "run-state-call",
                "node-state-call",
                "row-state-call",
                row_index=0,
                source_data_ref=None,
            )
            _create_token(conn, "row-state-call", "tok-state-call")
            _create_node_state(
                conn,
                state_id="state-state-call",
                token_id="tok-state-call",
                run_id="run-state-call",
                node_id="node-state-call",
            )
            _create_call_for_state(
                conn,
                call_id="call-state-call",
                state_id="state-state-call",
                request_ref="ref-state-call",
                response_ref=None,
            )

            # Operation call ref run
            _create_run(conn, "run-op-call", status=RunStatus.COMPLETED, completed_at=old)
            _create_node(conn, "run-op-call", "node-op-call")
            _create_operation(
                conn,
                operation_id="op-op-call",
                run_id="run-op-call",
                node_id="node-op-call",
            )
            _create_call_for_operation(
                conn,
                call_id="call-op-call",
                operation_id="op-op-call",
                request_ref=None,
                response_ref="ref-op-call",
            )

            # Operation input/output ref run
            _create_run(conn, "run-op-io", status=RunStatus.COMPLETED, completed_at=old)
            _create_node(conn, "run-op-io", "node-op-io")
            _create_operation(
                conn,
                operation_id="op-op-io",
                run_id="run-op-io",
                node_id="node-op-io",
                input_data_ref="ref-op-input",
                output_data_ref="ref-op-output",
            )

            # Routing ref run
            _create_run(conn, "run-routing", status=RunStatus.COMPLETED, completed_at=old)
            _create_node(conn, "run-routing", "node-routing-a")
            _create_node(conn, "run-routing", "node-routing-b")
            _create_row(
                conn,
                "run-routing",
                "node-routing-a",
                "row-routing",
                row_index=0,
                source_data_ref=None,
            )
            _create_token(conn, "row-routing", "tok-routing")
            _create_node_state(
                conn,
                state_id="state-routing",
                token_id="tok-routing",
                run_id="run-routing",
                node_id="node-routing-a",
            )
            _create_edge(
                conn,
                edge_id="edge-routing",
                run_id="run-routing",
                from_node="node-routing-a",
                to_node="node-routing-b",
            )
            _create_routing_event(
                conn,
                event_id="event-routing",
                state_id="state-routing",
                edge_id="edge-routing",
                reason_ref="ref-routing",
            )

        affected = manager._find_affected_run_ids(
            ["ref-row", "ref-state-call", "ref-op-call", "ref-op-input", "ref-op-output", "ref-routing"]
        )
        assert affected == {"run-row", "run-state-call", "run-op-call", "run-op-io", "run-routing"}


class TestPurgePayloads:
    def test_purge_payloads_tracks_deleted_skipped_and_failures(self, db: LandscapeDB, monkeypatch: pytest.MonkeyPatch) -> None:
        store = _ControlledStore()
        ok_ref = store.store(b"ok")
        false_ref = store.store(b"false")
        exists_error_ref = store.store(b"exists_err")
        delete_error_ref = store.store(b"delete_err")

        store._false_delete_for.add(false_ref)
        store._fail_exists_for.add(exists_error_ref)
        store._fail_delete_for.add(delete_error_ref)

        manager = PurgeManager(db, store)

        monkeypatch.setattr(manager, "_find_affected_run_ids", lambda refs: {"run-affected"} if refs else set())

        grade_updates: list[str] = []

        def _record_grade_update(db_obj: LandscapeDB, run_id: str) -> None:
            del db_obj
            grade_updates.append(run_id)

        monkeypatch.setattr("elspeth.core.retention.purge.update_grade_after_purge", _record_grade_update)

        values = iter([10.0, 13.25])
        monkeypatch.setattr("elspeth.core.retention.purge.perf_counter", lambda: next(values))

        result = manager.purge_payloads([ok_ref, "missing-ref", false_ref, exists_error_ref, delete_error_ref])

        assert result.deleted_count == 1
        assert result.skipped_count == 1
        assert set(result.failed_refs) == {false_ref, exists_error_ref, delete_error_ref}
        assert result.duration_seconds == 3.25
        assert grade_updates == ["run-affected"]

    def test_purge_payloads_only_passes_deleted_refs_to_affected_run_lookup(self, db: LandscapeDB, monkeypatch: pytest.MonkeyPatch) -> None:
        store = _ControlledStore()
        deleted_ref = store.store(b"deleted")
        failed_ref = store.store(b"failed")
        store._false_delete_for.add(failed_ref)

        manager = PurgeManager(db, store)

        captured_refs: list[str] = []

        def _capture_deleted_refs(refs: list[str]) -> set[str]:
            captured_refs.extend(refs)
            return set()

        monkeypatch.setattr(manager, "_find_affected_run_ids", _capture_deleted_refs)
        monkeypatch.setattr(
            "elspeth.core.retention.purge.update_grade_after_purge",
            lambda db_obj, run_id: None,
        )

        result = manager.purge_payloads([deleted_ref, failed_ref])

        assert result.deleted_count == 1
        assert result.failed_refs == [failed_ref]
        assert captured_refs == [deleted_ref]

    def test_purge_payloads_empty_input(self, db: LandscapeDB, monkeypatch: pytest.MonkeyPatch) -> None:
        manager = PurgeManager(db, MockPayloadStore())

        monkeypatch.setattr(
            "elspeth.core.retention.purge.update_grade_after_purge",
            lambda db_obj, run_id: None,
        )

        result = manager.purge_payloads([])
        assert result.deleted_count == 0
        assert result.skipped_count == 0
        assert result.failed_refs == []
        assert result.bytes_freed == 0
        assert result.duration_seconds >= 0
