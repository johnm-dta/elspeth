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
    ReproducibilityGrade,
    RoutingMode,
    RunStatus,
)
from elspeth.contracts.errors import AuditIntegrityError
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
    status: RunStatus | str,
    completed_at: datetime | None,
    reproducibility_grade: ReproducibilityGrade | str = ReproducibilityGrade.REPLAY_REPRODUCIBLE,
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
            reproducibility_grade=(
                reproducibility_grade.value if isinstance(reproducibility_grade, ReproducibilityGrade) else reproducibility_grade
            ),
        )
    )


def _create_node(
    conn: Connection,
    run_id: str,
    node_id: str,
    *,
    determinism: Determinism = Determinism.DETERMINISTIC,
) -> None:
    conn.execute(
        nodes_table.insert().values(
            node_id=node_id,
            run_id=run_id,
            plugin_name="test",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0.0",
            determinism=determinism,
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


def _create_token(conn: Connection, run_id: str, row_id: str, token_id: str) -> None:
    conn.execute(
        tokens_table.insert().values(
            token_id=token_id,
            row_id=row_id,
            run_id=run_id,
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


class TestPurgeResultValidation:
    """__post_init__ validation for PurgeResult — negative counts and tuple coercion."""

    def test_valid_construction(self) -> None:
        from elspeth.core.retention.purge import PurgeResult

        result = PurgeResult(deleted_count=5, skipped_count=2, failed_refs=("ref-1",), grade_update_failures=(), duration_seconds=1.5)
        assert result.deleted_count == 5
        assert result.failed_refs == ("ref-1",)

    def test_rejects_negative_deleted_count(self) -> None:
        from elspeth.core.retention.purge import PurgeResult

        with pytest.raises(ValueError, match="deleted_count must be non-negative"):
            PurgeResult(deleted_count=-1, skipped_count=0, failed_refs=(), grade_update_failures=(), duration_seconds=0.0)

    def test_rejects_negative_skipped_count(self) -> None:
        from elspeth.core.retention.purge import PurgeResult

        with pytest.raises(ValueError, match="skipped_count must be non-negative"):
            PurgeResult(deleted_count=0, skipped_count=-1, failed_refs=(), grade_update_failures=(), duration_seconds=0.0)

    def test_rejects_negative_duration(self) -> None:
        from elspeth.core.retention.purge import PurgeResult

        with pytest.raises(ValueError, match="duration_seconds must be non-negative"):
            PurgeResult(deleted_count=0, skipped_count=0, failed_refs=(), grade_update_failures=(), duration_seconds=-0.01)

    def test_coerces_list_failed_refs_to_tuple(self) -> None:
        from elspeth.core.retention.purge import PurgeResult

        result = PurgeResult(
            deleted_count=0, skipped_count=0, failed_refs=("ref-a", "ref-b"), grade_update_failures=(), duration_seconds=0.0
        )
        assert isinstance(result.failed_refs, tuple)
        assert result.failed_refs == ("ref-a", "ref-b")

    def test_accepts_zero_counts(self) -> None:
        from elspeth.core.retention.purge import PurgeResult

        result = PurgeResult(deleted_count=0, skipped_count=0, failed_refs=(), grade_update_failures=(), duration_seconds=0.0)
        assert result.deleted_count == 0
        assert result.skipped_count == 0
        assert result.duration_seconds == 0.0


class TestFindExpiredPayloadRefs:
    @pytest.mark.parametrize(
        ("status", "completed_at", "match"),
        (
            ("completed", None, "status='completed' but completed_at is NULL"),
            ("running", datetime(2026, 2, 8, tzinfo=UTC), "status='running' but completed_at is set"),
            ("not-a-run-status", None, "Invalid run status 'not-a-run-status'"),
        ),
    )
    def test_find_expired_payload_refs_raises_on_corrupt_run_lifecycle(
        self,
        db: LandscapeDB,
        status: str,
        completed_at: datetime | None,
        match: str,
    ) -> None:
        """Retention must crash on corrupt Tier-1 run lifecycle data."""
        manager = PurgeManager(db, MockPayloadStore())

        with db.connection() as conn:
            _create_run(conn, "run-corrupt-lifecycle", status=status, completed_at=completed_at)
            _create_node(conn, "run-corrupt-lifecycle", "node-corrupt-lifecycle")
            _create_row(
                conn,
                "run-corrupt-lifecycle",
                "node-corrupt-lifecycle",
                "row-corrupt-lifecycle",
                row_index=0,
                source_data_ref="ref-corrupt-lifecycle",
            )

        with pytest.raises(AuditIntegrityError, match=match):
            manager.find_expired_payload_refs(retention_days=30, as_of=datetime(2026, 3, 10, tzinfo=UTC))

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

    def test_find_expired_payload_refs_distinct_and_respects_status_and_cutoff(self, db: LandscapeDB) -> None:
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

        expired = set(manager.find_expired_payload_refs(retention_days=30, as_of=now))
        assert "ref-old-shared" in expired
        assert "ref-old-failed" in expired
        assert "ref-recent" not in expired
        assert "ref-running" not in expired

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
            _create_token(conn, "expired-run", "expired-row", "tok-expired")
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
            _create_token(conn, "run-state-call", "row-state-call", "tok-state-call")
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
            _create_token(conn, "run-routing", "row-routing", "tok-routing")
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
    def test_purge_payloads_downgrades_replay_run_when_deleted_ref_is_replay_critical(self, db: LandscapeDB) -> None:
        """A real blob purge must downgrade replay-only runs that need that response."""
        store = MockPayloadStore()
        response_ref = store.store(b'{"answer": 42}')
        manager = PurgeManager(db, store)
        old = datetime(2026, 1, 1, tzinfo=UTC)

        with db.connection() as conn:
            _create_run(
                conn,
                "run-replay-critical-purge",
                status=RunStatus.COMPLETED,
                completed_at=old,
                reproducibility_grade=ReproducibilityGrade.REPLAY_REPRODUCIBLE,
            )
            _create_node(
                conn,
                "run-replay-critical-purge",
                "node-replay-critical-purge",
                determinism=Determinism.NON_DETERMINISTIC,
            )
            _create_row(
                conn,
                "run-replay-critical-purge",
                "node-replay-critical-purge",
                "row-replay-critical-purge",
                row_index=0,
                source_data_ref=None,
            )
            _create_token(conn, "run-replay-critical-purge", "row-replay-critical-purge", "tok-replay-critical-purge")
            _create_node_state(
                conn,
                state_id="state-replay-critical-purge",
                token_id="tok-replay-critical-purge",
                run_id="run-replay-critical-purge",
                node_id="node-replay-critical-purge",
            )
            _create_call_for_state(
                conn,
                call_id="call-replay-critical-purge",
                state_id="state-replay-critical-purge",
                request_ref=None,
                response_ref=response_ref,
            )

        result = manager.purge_payloads([response_ref])

        assert result.deleted_count == 1
        assert result.failed_refs == ()
        assert store.exists(response_ref) is False
        with db.connection() as conn:
            grade = (
                conn.execute(runs_table.select().where(runs_table.c.run_id == "run-replay-critical-purge")).fetchone().reproducibility_grade
            )
        assert grade == ReproducibilityGrade.ATTRIBUTABLE_ONLY

    def test_purge_payloads_counts_delete_false_as_skipped_not_failed(
        self,
        db: LandscapeDB,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """PayloadStore.delete(False) means not found, not deletion failure."""
        store = _ControlledStore()
        stale_ref = store.store(b"already gone")
        store._false_delete_for.add(stale_ref)
        manager = PurgeManager(db, store)
        affected_lookup_inputs: list[list[str]] = []

        def _record_affected_lookup(refs: list[str]) -> set[str]:
            affected_lookup_inputs.append(refs)
            return set()

        monkeypatch.setattr(manager, "_find_affected_run_ids", _record_affected_lookup)

        result = manager.purge_payloads([stale_ref])

        assert result.deleted_count == 0
        assert result.skipped_count == 1
        assert result.failed_refs == ()
        assert affected_lookup_inputs == [[]]

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

        def _record_grade_update(db_obj: LandscapeDB, run_id: str, *, deleted_refs: list[str] | None = None) -> None:
            del db_obj
            assert deleted_refs == [ok_ref, exists_error_ref]
            grade_updates.append(run_id)

        monkeypatch.setattr("elspeth.core.retention.purge.update_grade_after_purge", _record_grade_update)

        values = iter([10.0, 13.25])
        monkeypatch.setattr("elspeth.core.retention.purge.perf_counter", lambda: next(values))

        result = manager.purge_payloads([ok_ref, "missing-ref", false_ref, exists_error_ref, delete_error_ref])

        assert result.deleted_count == 2
        assert result.skipped_count == 2
        assert set(result.failed_refs) == {delete_error_ref}
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
            lambda db_obj, run_id, *, deleted_refs=None: None,
        )

        result = manager.purge_payloads([deleted_ref, failed_ref])

        assert result.deleted_count == 1
        assert result.skipped_count == 1
        assert result.failed_refs == ()
        assert captured_refs == [deleted_ref]

    def test_purge_payloads_empty_input(self, db: LandscapeDB, monkeypatch: pytest.MonkeyPatch) -> None:
        manager = PurgeManager(db, MockPayloadStore())

        monkeypatch.setattr(
            "elspeth.core.retention.purge.update_grade_after_purge",
            lambda db_obj, run_id, *, deleted_refs=None: None,
        )

        result = manager.purge_payloads([])
        assert result.deleted_count == 0
        assert result.skipped_count == 0
        assert result.failed_refs == ()
        assert result.duration_seconds >= 0

    def test_partial_failure_accounting_invariant(self, db: LandscapeDB, monkeypatch: pytest.MonkeyPatch) -> None:
        """Mixed success/skip/failure: accounting invariant holds and grade updates scope correctly.

        Scenario: 7 refs total
          - 3 deleted successfully
          - 2 missing refs skipped
          - 1 delete-call fails (OSError)
          - 1 delete returns False and is skipped

        Verify:
          1. deleted_count + skipped_count + len(failed_refs) == 7
          2. Grade updates run only for runs linked to deleted refs
          3. Failed refs don't trigger grade updates
        """
        store = _ControlledStore()

        # Store payloads that will be deleted, delete-failed, or reported missing by delete()
        ok_ref_1 = store.store(b"ok-1")
        ok_ref_2 = store.store(b"ok-2")
        exists_check_irrelevant_ref = store.store(b"exists-check-irrelevant")
        delete_fail_ref = store.store(b"delete-fail")
        false_delete_ref = store.store(b"false-delete")

        store._fail_exists_for.add(exists_check_irrelevant_ref)
        store._fail_delete_for.add(delete_fail_ref)
        store._false_delete_for.add(false_delete_ref)

        manager = PurgeManager(db, store)

        # Map deleted refs → affected run IDs
        def _mock_affected(refs: list[str]) -> set[str]:
            # Only deleted refs should arrive here
            affected = set()
            if ok_ref_1 in refs:
                affected.add("run-alpha")
            if ok_ref_2 in refs:
                affected.add("run-beta")
            if exists_check_irrelevant_ref in refs:
                affected.add("run-gamma")
            # Failed/skipped refs must NOT appear
            assert delete_fail_ref not in refs
            assert false_delete_ref not in refs
            assert "missing-ref-1" not in refs
            assert "missing-ref-2" not in refs
            return affected

        monkeypatch.setattr(manager, "_find_affected_run_ids", _mock_affected)

        grade_updates: list[str] = []
        monkeypatch.setattr(
            "elspeth.core.retention.purge.update_grade_after_purge",
            lambda db_obj, run_id, *, deleted_refs=None: grade_updates.append(run_id),
        )

        all_refs = [
            ok_ref_1,
            "missing-ref-1",
            exists_check_irrelevant_ref,
            ok_ref_2,
            delete_fail_ref,
            "missing-ref-2",
            false_delete_ref,
        ]

        result = manager.purge_payloads(all_refs)

        # Accounting invariant
        assert result.deleted_count == 3
        assert result.skipped_count == 3
        assert len(result.failed_refs) == 1
        assert result.deleted_count + result.skipped_count + len(result.failed_refs) == len(all_refs)

        # Failed refs are only hard delete failures.
        assert set(result.failed_refs) == {delete_fail_ref}

        # Grade updates ran for all runs linked to successful deletions
        assert set(grade_updates) == {"run-alpha", "run-beta", "run-gamma"}

        # No grade update failures (all mocked to succeed)
        assert result.grade_update_failures == ()


class TestInterruptedRunNotPurgeEligible:
    """Regression test for Phase 0 fix #5: Interrupted run purge.

    Bug: The purge query used `status != "running"`, which included
    "interrupted" runs. Interrupted runs need their payloads preserved
    for resume, so purging them would break resume functionality.

    Fix: Changed to `status.in_(("completed", "failed"))` — only terminal
    statuses are purge-eligible. "interrupted" and "running" are excluded.
    """

    def test_interrupted_run_payloads_not_in_expired_payload_refs(self, db: LandscapeDB) -> None:
        """Interrupted runs must NOT appear in find_expired_payload_refs results."""
        manager = PurgeManager(db, MockPayloadStore())
        now = datetime(2026, 2, 14, tzinfo=UTC)
        old = now - timedelta(days=60)

        with db.connection() as conn:
            _create_run(conn, "run-interrupted-all", status=RunStatus.INTERRUPTED, completed_at=old)
            _create_node(conn, "run-interrupted-all", "node-interrupted-all")
            _create_row(
                conn,
                "run-interrupted-all",
                "node-interrupted-all",
                "row-interrupted-all",
                row_index=0,
                source_data_ref="ref-interrupted-all",
            )

        refs = set(manager.find_expired_payload_refs(retention_days=30, as_of=now))
        assert "ref-interrupted-all" not in refs

    def test_interrupted_run_protects_shared_blobs(self, db: LandscapeDB) -> None:
        """Shared blobs between expired and interrupted runs must not be purged.

        Bug: elspeth-d4297f57fa — payloads are content-addressable and shared
        across runs. The run_active_condition must include "interrupted" so
        the anti-join protects blobs still needed by interrupted runs.
        """
        manager = PurgeManager(db, MockPayloadStore())
        now = datetime(2026, 2, 14, tzinfo=UTC)
        old = now - timedelta(days=60)

        with db.connection() as conn:
            # Expired completed run with a shared blob
            _create_run(conn, "run-completed-old", status=RunStatus.COMPLETED, completed_at=old)
            _create_node(conn, "run-completed-old", "node-completed-old")
            _create_row(
                conn,
                "run-completed-old",
                "node-completed-old",
                "row-completed-old",
                row_index=0,
                source_data_ref="shared-blob-ref",
            )

            # Interrupted run referencing the SAME blob (content-addressable)
            _create_run(conn, "run-interrupted-shared", status=RunStatus.INTERRUPTED, completed_at=old)
            _create_node(conn, "run-interrupted-shared", "node-interrupted-shared")
            _create_row(
                conn,
                "run-interrupted-shared",
                "node-interrupted-shared",
                "row-interrupted-shared",
                row_index=0,
                source_data_ref="shared-blob-ref",
            )

        refs = set(manager.find_expired_payload_refs(retention_days=30, as_of=now))
        # The shared blob must NOT be purged because an interrupted run needs it
        assert "shared-blob-ref" not in refs


class TestPurgeGradeUpdateFailureResilience:
    """Regression test for elspeth-dfc66ddc10: grade update failure after irreversible payload deletion.

    Bug: After payloads are irreversibly deleted, the grade update loop
    (update_grade_after_purge) could raise for one run_id and abort the
    entire loop — leaving remaining runs with stale REPLAY_REPRODUCIBLE
    grades when their payloads no longer exist.

    Fix: Each grade update is wrapped individually. Failures are collected
    in PurgeResult.grade_update_failures so the operator knows which runs
    need manual correction.
    """

    def test_purge_result_has_grade_update_failures_field(self) -> None:
        """PurgeResult must expose grade_update_failures as a tuple of run_ids."""
        from elspeth.core.retention.purge import PurgeResult

        result = PurgeResult(
            deleted_count=1,
            skipped_count=0,
            failed_refs=(),
            grade_update_failures=("run-bad",),
            duration_seconds=1.0,
        )
        assert result.grade_update_failures == ("run-bad",)

    def test_purge_result_grade_update_failures_defaults_empty(self) -> None:
        """PurgeResult with no grade failures should have empty tuple."""
        from elspeth.core.retention.purge import PurgeResult

        result = PurgeResult(
            deleted_count=0,
            skipped_count=0,
            failed_refs=(),
            grade_update_failures=(),
            duration_seconds=0.0,
        )
        assert result.grade_update_failures == ()

    def test_grade_update_failure_does_not_abort_remaining_updates(self, db: LandscapeDB, monkeypatch: pytest.MonkeyPatch) -> None:
        """If update_grade_after_purge raises a transient error for one run, other runs still get updated.

        Uses RuntimeError to simulate a transient DB failure.
        AuditIntegrityError must NOT be swallowed — see separate test below.
        """
        store = _ControlledStore()
        ref = store.store(b"payload")
        manager = PurgeManager(db, store)

        # Simulate 3 affected runs
        monkeypatch.setattr(
            manager,
            "_find_affected_run_ids",
            lambda refs: {"run-ok-1", "run-bad", "run-ok-2"} if refs else set(),
        )

        grade_updates: list[str] = []

        def _failing_grade_update(db_obj: LandscapeDB, run_id: str, *, deleted_refs: list[str] | None = None) -> None:
            del db_obj
            assert deleted_refs == [ref]
            if run_id == "run-bad":
                raise RuntimeError(f"Transient DB failure for run '{run_id}'")
            grade_updates.append(run_id)

        monkeypatch.setattr(
            "elspeth.core.retention.purge.update_grade_after_purge",
            _failing_grade_update,
        )

        result = manager.purge_payloads([ref])

        # All non-failing runs should still get their grade update
        assert set(grade_updates) == {"run-ok-1", "run-ok-2"}
        # The failure should be reported in the result
        assert result.grade_update_failures == ("run-bad",)
        # Payloads were still deleted (irreversible)
        assert result.deleted_count == 1

    def test_grade_update_audit_integrity_error_propagates(self, db: LandscapeDB, monkeypatch: pytest.MonkeyPatch) -> None:
        """AuditIntegrityError from update_grade_after_purge must crash — never swallowed.

        Per Data Manifesto: corruption in the audit trail must crash immediately.
        A corrupt reproducibility_grade is evidence tampering if silently ignored.
        """
        from elspeth.contracts.errors import AuditIntegrityError

        store = _ControlledStore()
        ref = store.store(b"payload")
        manager = PurgeManager(db, store)

        monkeypatch.setattr(
            manager,
            "_find_affected_run_ids",
            lambda refs: {"run-corrupt"} if refs else set(),
        )

        def _integrity_failure(db_obj: LandscapeDB, run_id: str, *, deleted_refs: list[str] | None = None) -> None:
            assert deleted_refs == [ref]
            raise AuditIntegrityError(f"NULL reproducibility_grade for run {run_id} — audit data corruption")

        monkeypatch.setattr(
            "elspeth.core.retention.purge.update_grade_after_purge",
            _integrity_failure,
        )

        with pytest.raises(AuditIntegrityError, match="audit data corruption"):
            manager.purge_payloads([ref])

    def test_grade_update_failures_logged(self, db: LandscapeDB, monkeypatch: pytest.MonkeyPatch) -> None:
        """Grade update failures (transient, non-integrity) must be logged with structlog."""
        import structlog.testing

        store = _ControlledStore()
        ref = store.store(b"payload")
        manager = PurgeManager(db, store)

        monkeypatch.setattr(
            manager,
            "_find_affected_run_ids",
            lambda refs: {"run-fail"} if refs else set(),
        )

        def _transient_fail(db_obj: LandscapeDB, run_id: str, *, deleted_refs: list[str] | None = None) -> None:
            assert deleted_refs == [ref]
            raise RuntimeError(f"Transient failure for run '{run_id}'")

        monkeypatch.setattr(
            "elspeth.core.retention.purge.update_grade_after_purge",
            _transient_fail,
        )

        with structlog.testing.capture_logs() as cap_logs:
            result = manager.purge_payloads([ref])

        assert result.grade_update_failures == ("run-fail",)
        log_events = [e["event"] for e in cap_logs]
        assert "grade_update_failed" in log_events
        failed_log = next(e for e in cap_logs if e["event"] == "grade_update_failed")
        assert failed_log["run_id"] == "run-fail"


class TestPurgeUnboundedIN:
    """Regression test for Phase 0 fix #11: Purge unbounded IN clause.

    Bug: _find_affected_run_ids built a single IN clause with all refs,
    which could exceed SQLite's SQLITE_MAX_VARIABLE_NUMBER limit (999).

    Fix: Chunked refs into _PURGE_CHUNK_SIZE=100 batches before building
    IN clauses.
    """

    def test_find_affected_run_ids_with_many_refs(self, db: LandscapeDB) -> None:
        """_find_affected_run_ids handles >999 refs without SQLite error.

        Creates a real in-memory SQLite database with enough rows, then
        queries with >999 refs to verify the chunking works.
        """
        manager = PurgeManager(db, MockPayloadStore())
        now = datetime(2026, 2, 14, tzinfo=UTC)
        old = now - timedelta(days=60)

        with db.connection() as conn:
            _create_run(conn, "run-chunk-test", status=RunStatus.COMPLETED, completed_at=old)
            _create_node(conn, "run-chunk-test", "node-chunk-test")

            # Create one real ref that exists in the database
            _create_row(
                conn,
                "run-chunk-test",
                "node-chunk-test",
                "row-chunk-test",
                row_index=0,
                source_data_ref="ref-known",
            )

        # Build a list of >999 refs (most won't match anything)
        refs = [f"ref-fake-{i}" for i in range(1050)]
        refs.append("ref-known")  # This one should match

        # This would fail with sqlite3.OperationalError before the fix
        affected = manager._find_affected_run_ids(refs)
        assert "run-chunk-test" in affected
