# tests/core/retention/test_purge.py
"""Tests for PurgeManager - PayloadStore retention management."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest
from sqlalchemy import Connection, Table

from elspeth.contracts import (
    CallStatus,
    CallType,
    Determinism,
    NodeStateStatus,
    NodeType,
    RoutingMode,
    RunStatus,
)

if TYPE_CHECKING:
    from elspeth.core.landscape.database import LandscapeDB


@pytest.fixture(scope="module")
def landscape_db() -> LandscapeDB:
    """Module-scoped database for test performance."""
    from elspeth.core.landscape.database import LandscapeDB

    return LandscapeDB.in_memory()


def _create_state(
    conn: Connection,
    states_table: Table,
    state_id: str,
    token_id: str,
    node_id: str,
    run_id: str,
) -> None:
    """Helper to create a node_state record."""
    conn.execute(
        states_table.insert().values(
            state_id=state_id,
            token_id=token_id,
            node_id=node_id,
            run_id=run_id,
            step_index=0,
            attempt=0,
            status=NodeStateStatus.COMPLETED,
            input_hash="input123",
            output_hash="output123",
            started_at=datetime.now(UTC),
        )
    )


def _create_token(
    conn: Connection,
    tokens_table: Table,
    token_id: str,
    row_id: str,
) -> None:
    """Helper to create a token record.

    Note: tokens_table does NOT have run_id - tokens link to runs through rows.
    """
    conn.execute(
        tokens_table.insert().values(
            token_id=token_id,
            row_id=row_id,
            branch_name=None,
            created_at=datetime.now(UTC),
        )
    )


def _create_call(
    conn: Connection,
    calls_table: Table,
    call_id: str,
    state_id: str,
    *,
    request_ref: str | None = None,
    response_ref: str | None = None,
) -> None:
    """Helper to create an external call record."""
    conn.execute(
        calls_table.insert().values(
            call_id=call_id,
            state_id=state_id,
            call_index=0,
            call_type=CallType.LLM,
            status=CallStatus.SUCCESS,
            request_hash="req_hash",
            request_ref=request_ref,
            response_hash="resp_hash",
            response_ref=response_ref,
            created_at=datetime.now(UTC),
        )
    )


def _create_edge(
    conn: Connection,
    edges_table: Table,
    edge_id: str,
    run_id: str,
    from_node_id: str,
    to_node_id: str,
) -> None:
    """Helper to create an edge record."""
    conn.execute(
        edges_table.insert().values(
            edge_id=edge_id,
            run_id=run_id,
            from_node_id=from_node_id,
            to_node_id=to_node_id,
            label="continue",
            default_mode=RoutingMode.MOVE,
            created_at=datetime.now(UTC),
        )
    )


def _create_routing_event(
    conn: Connection,
    routing_events_table: Table,
    event_id: str,
    state_id: str,
    edge_id: str,
    *,
    reason_ref: str | None = None,
) -> None:
    """Helper to create a routing event record."""
    conn.execute(
        routing_events_table.insert().values(
            event_id=event_id,
            state_id=state_id,
            edge_id=edge_id,
            routing_group_id=str(uuid4()),
            ordinal=0,
            mode=RoutingMode.MOVE,
            reason_hash="reason_hash",
            reason_ref=reason_ref,
            created_at=datetime.now(UTC),
        )
    )


def _create_run(
    conn: Connection,
    runs_table: Table,
    run_id: str,
    *,
    completed_at: datetime | None = None,
    status: RunStatus = RunStatus.COMPLETED,
    reproducibility_grade: str = "full_reproducible",
) -> None:
    """Helper to create a run record."""
    conn.execute(
        runs_table.insert().values(
            run_id=run_id,
            started_at=datetime.now(UTC),
            completed_at=completed_at,
            config_hash="abc123",
            settings_json="{}",
            canonical_version="1.0.0",
            status=status,
            reproducibility_grade=reproducibility_grade,
        )
    )


def _create_node(
    conn: Connection,
    nodes_table: Table,
    node_id: str,
    run_id: str,
) -> None:
    """Helper to create a node record."""
    conn.execute(
        nodes_table.insert().values(
            node_id=node_id,
            run_id=run_id,
            plugin_name="test_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            determinism=Determinism.DETERMINISTIC,
            config_hash="config123",
            config_json="{}",
            registered_at=datetime.now(UTC),
        )
    )


def _create_row(
    conn: Connection,
    rows_table: Table,
    row_id: str,
    run_id: str,
    node_id: str,
    row_index: int,
    *,
    source_data_ref: str | None = None,
    source_data_hash: str = "hash123",
) -> None:
    """Helper to create a row record."""
    conn.execute(
        rows_table.insert().values(
            row_id=row_id,
            run_id=run_id,
            source_node_id=node_id,
            row_index=row_index,
            source_data_hash=source_data_hash,
            source_data_ref=source_data_ref,
            created_at=datetime.now(UTC),
        )
    )


class MockPayloadStore:
    """Mock PayloadStore for testing PurgeManager."""

    def __init__(self) -> None:
        self._storage: dict[str, bytes] = {}
        self.delete_calls: list[str] = []

    def store(self, content: bytes) -> str:
        """Store content and return hash."""
        import hashlib

        content_hash = hashlib.sha256(content).hexdigest()
        self._storage[content_hash] = content
        return content_hash

    def exists(self, content_hash: str) -> bool:
        """Check if content exists."""
        return content_hash in self._storage

    def delete(self, content_hash: str) -> bool:
        """Delete content by hash. Returns True if deleted."""
        self.delete_calls.append(content_hash)
        if content_hash in self._storage:
            del self._storage[content_hash]
            return True
        return False

    def retrieve(self, content_hash: str) -> bytes:
        """Retrieve content by hash."""
        if content_hash not in self._storage:
            raise KeyError(f"Payload not found: {content_hash}")
        return self._storage[content_hash]


class TestPurgeResult:
    """Tests for PurgeResult dataclass."""

    def test_purge_result_fields(self) -> None:
        from elspeth.core.retention.purge import PurgeResult

        result = PurgeResult(
            deleted_count=5,
            bytes_freed=1024,
            skipped_count=2,
            failed_refs=["abc", "def"],
            duration_seconds=1.5,
        )

        assert result.deleted_count == 5
        assert result.bytes_freed == 1024
        assert result.skipped_count == 2
        assert result.failed_refs == ["abc", "def"]
        assert result.duration_seconds == 1.5


class TestFindExpiredRowPayloads:
    """Tests for find_expired_row_payloads method."""

    def test_find_expired_row_payloads(self, landscape_db: LandscapeDB) -> None:
        """Finds row payloads older than retention period."""
        from elspeth.core.landscape.schema import nodes_table, rows_table, runs_table
        from elspeth.core.retention.purge import PurgeManager

        db = landscape_db
        store = MockPayloadStore()
        manager = PurgeManager(db, store)

        # Create a run completed 60 days ago
        run_id = str(uuid4())
        node_id = str(uuid4())
        old_completed_at = datetime.now(UTC) - timedelta(days=60)
        test_ref = f"ref_for_old_payload_{uuid4()}"

        with db.connection() as conn:
            _create_run(
                conn,
                runs_table,
                run_id,
                completed_at=old_completed_at,
                status=RunStatus.COMPLETED,
            )
            _create_node(conn, nodes_table, node_id, run_id)
            _create_row(
                conn,
                rows_table,
                row_id=str(uuid4()),
                run_id=run_id,
                node_id=node_id,
                row_index=0,
                source_data_ref=test_ref,
                source_data_hash="hash_old",
            )

        # Find payloads older than 30 days
        expired = manager.find_expired_row_payloads(retention_days=30)

        assert test_ref in expired

    def test_find_expired_respects_retention(self, landscape_db: LandscapeDB) -> None:
        """Does not flag recent payloads."""
        from elspeth.core.landscape.schema import nodes_table, rows_table, runs_table
        from elspeth.core.retention.purge import PurgeManager

        db = landscape_db
        store = MockPayloadStore()
        manager = PurgeManager(db, store)

        # Create a run completed 10 days ago (within retention)
        run_id = str(uuid4())
        node_id = str(uuid4())
        recent_completed_at = datetime.now(UTC) - timedelta(days=10)
        test_ref = f"ref_for_recent_payload_{uuid4()}"

        with db.connection() as conn:
            _create_run(
                conn,
                runs_table,
                run_id,
                completed_at=recent_completed_at,
                status=RunStatus.COMPLETED,
            )
            _create_node(conn, nodes_table, node_id, run_id)
            _create_row(
                conn,
                rows_table,
                row_id=str(uuid4()),
                run_id=run_id,
                node_id=node_id,
                row_index=0,
                source_data_ref=test_ref,
                source_data_hash="hash_recent",
            )

        # Find payloads older than 30 days - should NOT include recent
        expired = manager.find_expired_row_payloads(retention_days=30)

        assert test_ref not in expired

    def test_find_expired_ignores_incomplete_runs(self, landscape_db: LandscapeDB) -> None:
        """Does not flag payloads from incomplete runs."""
        from elspeth.core.landscape.schema import nodes_table, rows_table, runs_table
        from elspeth.core.retention.purge import PurgeManager

        db = landscape_db
        store = MockPayloadStore()
        manager = PurgeManager(db, store)

        # Create a run from 60 days ago that is still running
        run_id = str(uuid4())
        node_id = str(uuid4())
        test_ref = f"ref_for_running_payload_{uuid4()}"

        with db.connection() as conn:
            _create_run(
                conn,
                runs_table,
                run_id,
                completed_at=None,  # Not completed
                status=RunStatus.RUNNING,
            )
            _create_node(conn, nodes_table, node_id, run_id)
            _create_row(
                conn,
                rows_table,
                row_id=str(uuid4()),
                run_id=run_id,
                node_id=node_id,
                row_index=0,
                source_data_ref=test_ref,
                source_data_hash="hash_running",
            )

        # Find payloads older than 30 days - should NOT include running
        expired = manager.find_expired_row_payloads(retention_days=30)

        assert test_ref not in expired

    def test_find_expired_excludes_null_refs(self, landscape_db: LandscapeDB) -> None:
        """Does not include rows with null source_data_ref."""
        from elspeth.core.landscape.schema import nodes_table, rows_table, runs_table
        from elspeth.core.retention.purge import PurgeManager

        db = landscape_db
        store = MockPayloadStore()
        manager = PurgeManager(db, store)

        run_id = str(uuid4())
        node_id = str(uuid4())
        old_completed_at = datetime.now(UTC) - timedelta(days=60)

        with db.connection() as conn:
            _create_run(
                conn,
                runs_table,
                run_id,
                completed_at=old_completed_at,
                status=RunStatus.COMPLETED,
            )
            _create_node(conn, nodes_table, node_id, run_id)
            _create_row(
                conn,
                rows_table,
                row_id=str(uuid4()),
                run_id=run_id,
                node_id=node_id,
                row_index=0,
                source_data_ref=None,  # No ref - payload was inline
                source_data_hash="hash_inline",
            )

        expired = manager.find_expired_row_payloads(retention_days=30)

        # Verify no null refs are returned
        assert None not in expired
        assert "" not in expired
        # All returned refs should be non-empty strings
        for ref in expired:
            assert ref and isinstance(ref, str)

    def test_find_expired_with_as_of_date(self, landscape_db: LandscapeDB) -> None:
        """Uses as_of date for cutoff calculation."""
        from elspeth.core.landscape.schema import nodes_table, rows_table, runs_table
        from elspeth.core.retention.purge import PurgeManager

        db = landscape_db
        store = MockPayloadStore()
        manager = PurgeManager(db, store)

        run_id = str(uuid4())
        node_id = str(uuid4())
        # Run completed 45 days ago
        completed_at = datetime.now(UTC) - timedelta(days=45)
        test_ref = f"ref_45_days_old_{uuid4()}"

        with db.connection() as conn:
            _create_run(conn, runs_table, run_id, completed_at=completed_at, status=RunStatus.COMPLETED)
            _create_node(conn, nodes_table, node_id, run_id)
            _create_row(
                conn,
                rows_table,
                row_id=str(uuid4()),
                run_id=run_id,
                node_id=node_id,
                row_index=0,
                source_data_ref=test_ref,
                source_data_hash="hash_45",
            )

        # With as_of=now, 30 day retention - 45 days old is expired
        expired_now = manager.find_expired_row_payloads(retention_days=30)
        assert test_ref in expired_now

        # With as_of=60 days ago, 30 day retention - 45 days old was not expired yet
        as_of = datetime.now(UTC) - timedelta(days=60)
        expired_past = manager.find_expired_row_payloads(retention_days=30, as_of=as_of)
        assert test_ref not in expired_past

    def test_find_expired_deduplicates_shared_refs(self, landscape_db: LandscapeDB) -> None:
        """Multiple rows referencing the same payload return only one ref.

        Content-addressed storage means identical content shares one blob,
        so multiple rows can have the same source_data_ref. The query must
        deduplicate to avoid returning the same ref multiple times.
        """
        from elspeth.core.landscape.schema import nodes_table, rows_table, runs_table
        from elspeth.core.retention.purge import PurgeManager

        db = landscape_db
        store = MockPayloadStore()
        manager = PurgeManager(db, store)

        run_id = str(uuid4())
        node_id = str(uuid4())
        old_completed_at = datetime.now(UTC) - timedelta(days=60)

        # The shared ref that multiple rows point to (content-addressed)
        shared_ref = f"shared_content_hash_{uuid4()}"

        with db.connection() as conn:
            _create_run(
                conn,
                runs_table,
                run_id,
                completed_at=old_completed_at,
                status=RunStatus.COMPLETED,
            )
            _create_node(conn, nodes_table, node_id, run_id)

            # Create 3 rows that all reference the same payload
            for i in range(3):
                _create_row(
                    conn,
                    rows_table,
                    row_id=str(uuid4()),
                    run_id=run_id,
                    node_id=node_id,
                    row_index=i,
                    source_data_ref=shared_ref,
                    source_data_hash=f"hash_{i}",  # Different hashes, same ref
                )

        expired = manager.find_expired_row_payloads(retention_days=30)

        # Should return exactly one instance of the shared ref, not three
        assert expired.count(shared_ref) == 1


class TestPurgePayloads:
    """Tests for purge_payloads method."""

    def test_purge_payloads_deletes_content(self, landscape_db: LandscapeDB) -> None:
        """Purge actually deletes from PayloadStore."""
        from elspeth.core.retention.purge import PurgeManager

        db = landscape_db
        store = MockPayloadStore()

        # Store some content
        ref1 = store.store(b"payload content 1")
        ref2 = store.store(b"payload content 2")

        manager = PurgeManager(db, store)
        result = manager.purge_payloads([ref1, ref2])

        assert result.deleted_count == 2
        assert ref1 not in store._storage
        assert ref2 not in store._storage
        assert store.delete_calls == [ref1, ref2]

    def test_purge_preserves_landscape_hashes(self, landscape_db: LandscapeDB) -> None:
        """Purge deletes blobs but keeps hashes in Landscape."""
        from elspeth.core.landscape.schema import nodes_table, rows_table, runs_table
        from elspeth.core.retention.purge import PurgeManager

        db = landscape_db
        store = MockPayloadStore()

        # Create run with row
        run_id = str(uuid4())
        node_id = str(uuid4())
        row_id = str(uuid4())
        old_completed_at = datetime.now(UTC) - timedelta(days=60)

        # Store payload and get ref
        payload_ref = store.store(b"source row content")

        with db.connection() as conn:
            _create_run(
                conn,
                runs_table,
                run_id,
                completed_at=old_completed_at,
                status=RunStatus.COMPLETED,
            )
            _create_node(conn, nodes_table, node_id, run_id)
            _create_row(
                conn,
                rows_table,
                row_id=row_id,
                run_id=run_id,
                node_id=node_id,
                row_index=0,
                source_data_ref=payload_ref,
                source_data_hash="original_hash_kept",
            )

        manager = PurgeManager(db, store)
        manager.purge_payloads([payload_ref])

        # Payload deleted
        assert not store.exists(payload_ref)

        # But hash still in Landscape
        with db.connection() as conn:
            from sqlalchemy import select

            result = conn.execute(select(rows_table.c.source_data_hash).where(rows_table.c.row_id == row_id))
            saved_hash = result.scalar()
            assert saved_hash == "original_hash_kept"

    def test_purge_tracks_skipped_refs(self, landscape_db: LandscapeDB) -> None:
        """Purge tracks refs that don't exist as skipped (not failed)."""
        from elspeth.core.retention.purge import PurgeManager

        db = landscape_db
        store = MockPayloadStore()

        # Store one payload, leave another ref nonexistent
        existing_ref = store.store(b"existing content")
        nonexistent_ref = "nonexistent_ref_abc123"

        manager = PurgeManager(db, store)
        result = manager.purge_payloads([existing_ref, nonexistent_ref])

        assert result.deleted_count == 1
        assert result.skipped_count == 1
        # Non-existent refs are skipped, not failed
        assert nonexistent_ref not in result.failed_refs
        assert result.failed_refs == []

    def test_purge_tracks_failed_refs(self, landscape_db: LandscapeDB) -> None:
        """Purge tracks refs that exist but fail to delete.

        This tests the failure path where exists() returns True but
        delete() returns False (e.g., permission error, I/O failure).
        """
        from elspeth.core.retention.purge import PurgeManager

        db = landscape_db

        # Create a mock store that simulates deletion failure for a specific ref
        class FailingPayloadStore:
            """Mock that fails to delete specific refs."""

            def __init__(self, fail_refs: set[str]) -> None:
                self._storage: dict[str, bytes] = {}
                self._fail_refs = fail_refs

            def store(self, content: bytes) -> str:
                import hashlib

                content_hash = hashlib.sha256(content).hexdigest()
                self._storage[content_hash] = content
                return content_hash

            def exists(self, content_hash: str) -> bool:
                return content_hash in self._storage

            def delete(self, content_hash: str) -> bool:
                if content_hash in self._fail_refs:
                    # Simulate deletion failure (e.g., I/O error)
                    return False
                if content_hash in self._storage:
                    del self._storage[content_hash]
                    return True
                return False

        # Store two payloads, mark one for failure
        store = FailingPayloadStore(fail_refs=set())
        success_ref = store.store(b"will succeed")
        fail_ref = store.store(b"will fail")
        store._fail_refs.add(fail_ref)

        manager = PurgeManager(db, store)
        result = manager.purge_payloads([success_ref, fail_ref])

        # One succeeded, one failed
        assert result.deleted_count == 1
        assert result.skipped_count == 0
        assert result.failed_refs == [fail_ref]

        # Successful one is gone, failed one still exists
        assert not store.exists(success_ref)
        assert store.exists(fail_ref)

    def test_purge_measures_duration(self, landscape_db: LandscapeDB, monkeypatch: pytest.MonkeyPatch) -> None:
        """Purge measures operation duration using deterministic time."""
        from elspeth.core.retention import purge as purge_module
        from elspeth.core.retention.purge import PurgeManager

        db = landscape_db
        store = MockPayloadStore()

        ref = store.store(b"content")

        # Monkeypatch perf_counter to return deterministic values
        call_count = 0

        def fake_perf_counter() -> float:
            nonlocal call_count
            call_count += 1
            # First call (start): 10.0, Second call (end): 12.5
            return 10.0 if call_count == 1 else 12.5

        monkeypatch.setattr(purge_module, "perf_counter", fake_perf_counter)

        manager = PurgeManager(db, store)
        result = manager.purge_payloads([ref])

        assert result.duration_seconds == 2.5

    def test_purge_empty_list(self, landscape_db: LandscapeDB) -> None:
        """Purge with empty list returns empty result."""
        from elspeth.core.retention.purge import PurgeManager

        db = landscape_db
        store = MockPayloadStore()

        manager = PurgeManager(db, store)
        result = manager.purge_payloads([])

        assert result.deleted_count == 0
        assert result.bytes_freed == 0
        assert result.skipped_count == 0
        assert result.failed_refs == []


class TestFindExpiredCallPayloads:
    """Tests for finding expired call payloads (request/response refs).

    Per P2-2026-01-19-retention-purge-ignores-call-and-reason-payload-refs:
    Call payloads (request_ref, response_ref) should be subject to the same
    retention policy as row payloads.
    """

    def test_find_expired_includes_call_request_refs(self, landscape_db: LandscapeDB) -> None:
        """Expired call request payloads should be found for purge."""
        from elspeth.core.landscape.schema import (
            calls_table,
            node_states_table,
            nodes_table,
            rows_table,
            runs_table,
            tokens_table,
        )
        from elspeth.core.retention.purge import PurgeManager

        db = landscape_db
        store = MockPayloadStore()
        manager = PurgeManager(db, store)

        # Create a run completed 60 days ago
        run_id = str(uuid4())
        node_id = str(uuid4())
        row_id = str(uuid4())
        token_id = str(uuid4())
        state_id = str(uuid4())
        call_id = str(uuid4())
        old_completed_at = datetime.now(UTC) - timedelta(days=60)

        with db.connection() as conn:
            _create_run(conn, runs_table, run_id, completed_at=old_completed_at, status=RunStatus.COMPLETED)
            _create_node(conn, nodes_table, node_id, run_id)
            _create_row(conn, rows_table, row_id, run_id, node_id, row_index=0)
            _create_token(conn, tokens_table, token_id, row_id)
            _create_state(conn, node_states_table, state_id, token_id, node_id, run_id)
            _create_call(
                conn,
                calls_table,
                call_id,
                state_id,
                request_ref="call_request_payload_ref",
                response_ref="call_response_payload_ref",
            )

        # Find payloads older than 30 days - should include call payloads
        expired = manager.find_expired_payload_refs(retention_days=30)

        assert "call_request_payload_ref" in expired, "Call request_ref should be found"
        assert "call_response_payload_ref" in expired, "Call response_ref should be found"

    def test_find_expired_includes_call_response_refs(self, landscape_db: LandscapeDB) -> None:
        """Expired call response payloads should be found for purge."""
        from elspeth.core.landscape.schema import (
            calls_table,
            node_states_table,
            nodes_table,
            rows_table,
            runs_table,
            tokens_table,
        )
        from elspeth.core.retention.purge import PurgeManager

        db = landscape_db
        store = MockPayloadStore()
        manager = PurgeManager(db, store)

        # Create a run completed 60 days ago with only response_ref
        run_id = str(uuid4())
        node_id = str(uuid4())
        row_id = str(uuid4())
        token_id = str(uuid4())
        state_id = str(uuid4())
        call_id = str(uuid4())
        old_completed_at = datetime.now(UTC) - timedelta(days=60)

        with db.connection() as conn:
            _create_run(conn, runs_table, run_id, completed_at=old_completed_at, status=RunStatus.COMPLETED)
            _create_node(conn, nodes_table, node_id, run_id)
            _create_row(conn, rows_table, row_id, run_id, node_id, row_index=0)
            _create_token(conn, tokens_table, token_id, row_id)
            _create_state(conn, node_states_table, state_id, token_id, node_id, run_id)
            _create_call(
                conn,
                calls_table,
                call_id,
                state_id,
                request_ref=None,  # No request ref
                response_ref="only_response_ref",
            )

        expired = manager.find_expired_payload_refs(retention_days=30)

        assert "only_response_ref" in expired


class TestFindExpiredRoutingPayloads:
    """Tests for finding expired routing event payloads (reason_ref).

    Per P2-2026-01-19-retention-purge-ignores-call-and-reason-payload-refs:
    Routing reason payloads should be subject to the same retention policy.
    """

    def test_find_expired_includes_routing_reason_refs(self, landscape_db: LandscapeDB) -> None:
        """Expired routing reason payloads should be found for purge."""
        from elspeth.core.landscape.schema import (
            edges_table,
            node_states_table,
            nodes_table,
            routing_events_table,
            rows_table,
            runs_table,
            tokens_table,
        )
        from elspeth.core.retention.purge import PurgeManager

        db = landscape_db
        store = MockPayloadStore()
        manager = PurgeManager(db, store)

        # Create a run completed 60 days ago
        run_id = str(uuid4())
        source_node_id = str(uuid4())
        sink_node_id = str(uuid4())
        row_id = str(uuid4())
        token_id = str(uuid4())
        state_id = str(uuid4())
        edge_id = str(uuid4())
        event_id = str(uuid4())
        old_completed_at = datetime.now(UTC) - timedelta(days=60)

        with db.connection() as conn:
            _create_run(conn, runs_table, run_id, completed_at=old_completed_at, status=RunStatus.COMPLETED)
            _create_node(conn, nodes_table, source_node_id, run_id)
            _create_node(conn, nodes_table, sink_node_id, run_id)
            _create_row(conn, rows_table, row_id, run_id, source_node_id, row_index=0)
            _create_token(conn, tokens_table, token_id, row_id)
            _create_state(conn, node_states_table, state_id, token_id, source_node_id, run_id)
            _create_edge(conn, edges_table, edge_id, run_id, source_node_id, sink_node_id)
            _create_routing_event(
                conn,
                routing_events_table,
                event_id,
                state_id,
                edge_id,
                reason_ref="routing_reason_payload_ref",
            )

        # Find payloads older than 30 days - should include routing reason
        expired = manager.find_expired_payload_refs(retention_days=30)

        assert "routing_reason_payload_ref" in expired, "Routing reason_ref should be found"


class TestFindExpiredAllPayloadRefs:
    """Tests for the unified find_expired_payload_refs method."""

    def test_find_expired_payload_refs_returns_deduplicated_union(self, landscape_db: LandscapeDB) -> None:
        """All payload types should be returned, deduplicated."""
        from elspeth.core.landscape.schema import (
            calls_table,
            edges_table,
            node_states_table,
            nodes_table,
            routing_events_table,
            rows_table,
            runs_table,
            tokens_table,
        )
        from elspeth.core.retention.purge import PurgeManager

        db = landscape_db
        store = MockPayloadStore()
        manager = PurgeManager(db, store)

        # Create a run completed 60 days ago with all payload types
        run_id = str(uuid4())
        source_node_id = str(uuid4())
        sink_node_id = str(uuid4())
        row_id = str(uuid4())
        token_id = str(uuid4())
        state_id = str(uuid4())
        call_id = str(uuid4())
        edge_id = str(uuid4())
        event_id = str(uuid4())
        old_completed_at = datetime.now(UTC) - timedelta(days=60)
        row_ref = f"row_payload_ref_{uuid4()}"
        call_req_ref = f"call_request_ref_{uuid4()}"
        call_resp_ref = f"call_response_ref_{uuid4()}"
        routing_ref = f"routing_reason_ref_{uuid4()}"

        with db.connection() as conn:
            _create_run(conn, runs_table, run_id, completed_at=old_completed_at, status=RunStatus.COMPLETED)
            _create_node(conn, nodes_table, source_node_id, run_id)
            _create_node(conn, nodes_table, sink_node_id, run_id)
            _create_row(
                conn,
                rows_table,
                row_id,
                run_id,
                source_node_id,
                row_index=0,
                source_data_ref=row_ref,
            )
            _create_token(conn, tokens_table, token_id, row_id)
            _create_state(conn, node_states_table, state_id, token_id, source_node_id, run_id)
            _create_call(
                conn,
                calls_table,
                call_id,
                state_id,
                request_ref=call_req_ref,
                response_ref=call_resp_ref,
            )
            _create_edge(conn, edges_table, edge_id, run_id, source_node_id, sink_node_id)
            _create_routing_event(
                conn,
                routing_events_table,
                event_id,
                state_id,
                edge_id,
                reason_ref=routing_ref,
            )

        # Find all expired payload refs
        expired = manager.find_expired_payload_refs(retention_days=30)

        # Should include all 4 payload refs
        assert row_ref in expired
        assert call_req_ref in expired
        assert call_resp_ref in expired
        assert routing_ref in expired

        # Each ref should appear exactly once (deduplicated)
        assert expired.count(row_ref) == 1
        assert expired.count(call_req_ref) == 1
        assert expired.count(call_resp_ref) == 1
        assert expired.count(routing_ref) == 1

    def test_find_expired_payload_refs_respects_retention(self, landscape_db: LandscapeDB) -> None:
        """Recent call/routing payloads should not be found."""
        from elspeth.core.landscape.schema import (
            calls_table,
            node_states_table,
            nodes_table,
            rows_table,
            runs_table,
            tokens_table,
        )
        from elspeth.core.retention.purge import PurgeManager

        db = landscape_db
        store = MockPayloadStore()
        manager = PurgeManager(db, store)

        # Create a run completed 10 days ago (within retention)
        run_id = str(uuid4())
        node_id = str(uuid4())
        row_id = str(uuid4())
        token_id = str(uuid4())
        state_id = str(uuid4())
        call_id = str(uuid4())
        recent_completed_at = datetime.now(UTC) - timedelta(days=10)

        with db.connection() as conn:
            _create_run(conn, runs_table, run_id, completed_at=recent_completed_at, status=RunStatus.COMPLETED)
            _create_node(conn, nodes_table, node_id, run_id)
            _create_row(conn, rows_table, row_id, run_id, node_id, row_index=0)
            _create_token(conn, tokens_table, token_id, row_id)
            _create_state(conn, node_states_table, state_id, token_id, node_id, run_id)
            _create_call(
                conn,
                calls_table,
                call_id,
                state_id,
                request_ref="recent_call_ref",
            )

        # Find payloads older than 30 days - should NOT include recent
        expired = manager.find_expired_payload_refs(retention_days=30)

        assert "recent_call_ref" not in expired


class TestContentAddressableSharedRefs:
    """Tests for content-addressable storage shared refs across runs.

    Regression tests for P2-payload-refs-shared-across-runs bug:
    Because payloads are content-addressable, the same hash can appear in
    multiple runs. Purge must exclude refs still used by non-expired runs.
    """

    def test_shared_row_ref_excluded_when_used_by_recent_run(self, landscape_db: LandscapeDB) -> None:
        """Shared row payload ref should NOT be purged if a recent run uses it.

        Scenario:
        - Run A (60 days old, expired) has row with source_data_ref=H
        - Run B (10 days old, not expired) has row with same source_data_ref=H
        - Purge with 30-day retention should NOT return H (Run B needs it)
        """
        from elspeth.core.landscape.schema import nodes_table, rows_table, runs_table
        from elspeth.core.retention.purge import PurgeManager

        db = landscape_db
        store = MockPayloadStore()
        manager = PurgeManager(db, store)

        # The shared ref (content-addressable: same content = same hash)
        shared_ref = "shared_row_payload_hash"

        # Run A: 60 days old (expired with 30-day retention)
        run_a_id = str(uuid4())
        node_a_id = str(uuid4())
        old_completed = datetime.now(UTC) - timedelta(days=60)

        # Run B: 10 days old (NOT expired with 30-day retention)
        run_b_id = str(uuid4())
        node_b_id = str(uuid4())
        recent_completed = datetime.now(UTC) - timedelta(days=10)

        with db.connection() as conn:
            # Create Run A (expired) with shared ref
            _create_run(conn, runs_table, run_a_id, completed_at=old_completed, status=RunStatus.COMPLETED)
            _create_node(conn, nodes_table, node_a_id, run_a_id)
            _create_row(
                conn,
                rows_table,
                str(uuid4()),
                run_a_id,
                node_a_id,
                row_index=0,
                source_data_ref=shared_ref,
                source_data_hash="hash_a",
            )

            # Create Run B (NOT expired) with SAME shared ref
            _create_run(conn, runs_table, run_b_id, completed_at=recent_completed, status=RunStatus.COMPLETED)
            _create_node(conn, nodes_table, node_b_id, run_b_id)
            _create_row(
                conn,
                rows_table,
                str(uuid4()),
                run_b_id,
                node_b_id,
                row_index=0,
                source_data_ref=shared_ref,
                source_data_hash="hash_b",
            )

        # Find expired refs - shared ref should be EXCLUDED
        expired = manager.find_expired_payload_refs(retention_days=30)

        assert shared_ref not in expired, (
            f"Shared ref {shared_ref} should NOT be returned for deletion because Run B (10 days old) still needs it"
        )

    def test_shared_call_ref_excluded_when_used_by_recent_run(self, landscape_db: LandscapeDB) -> None:
        """Shared call payload ref should NOT be purged if a recent run uses it."""
        from elspeth.core.landscape.schema import (
            calls_table,
            node_states_table,
            nodes_table,
            rows_table,
            runs_table,
            tokens_table,
        )
        from elspeth.core.retention.purge import PurgeManager

        db = landscape_db
        store = MockPayloadStore()
        manager = PurgeManager(db, store)

        # Shared call response ref (e.g., identical LLM response)
        shared_ref = "shared_call_response_hash"

        # Run A: 60 days old (expired)
        run_a_id = str(uuid4())
        old_completed = datetime.now(UTC) - timedelta(days=60)

        # Run B: 10 days old (NOT expired)
        run_b_id = str(uuid4())
        recent_completed = datetime.now(UTC) - timedelta(days=10)

        with db.connection() as conn:
            # Run A with call using shared ref
            node_a_id = str(uuid4())
            row_a_id = str(uuid4())
            token_a_id = str(uuid4())
            state_a_id = str(uuid4())
            _create_run(conn, runs_table, run_a_id, completed_at=old_completed, status=RunStatus.COMPLETED)
            _create_node(conn, nodes_table, node_a_id, run_a_id)
            _create_row(conn, rows_table, row_a_id, run_a_id, node_a_id, row_index=0)
            _create_token(conn, tokens_table, token_a_id, row_a_id)
            _create_state(conn, node_states_table, state_a_id, token_a_id, node_a_id, run_a_id)
            _create_call(conn, calls_table, str(uuid4()), state_a_id, response_ref=shared_ref)

            # Run B with call using SAME shared ref
            node_b_id = str(uuid4())
            row_b_id = str(uuid4())
            token_b_id = str(uuid4())
            state_b_id = str(uuid4())
            _create_run(conn, runs_table, run_b_id, completed_at=recent_completed, status=RunStatus.COMPLETED)
            _create_node(conn, nodes_table, node_b_id, run_b_id)
            _create_row(conn, rows_table, row_b_id, run_b_id, node_b_id, row_index=0)
            _create_token(conn, tokens_table, token_b_id, row_b_id)
            _create_state(conn, node_states_table, state_b_id, token_b_id, node_b_id, run_b_id)
            _create_call(conn, calls_table, str(uuid4()), state_b_id, response_ref=shared_ref)

        expired = manager.find_expired_payload_refs(retention_days=30)

        assert shared_ref not in expired, (
            f"Shared call ref {shared_ref} should NOT be returned for deletion because Run B (10 days old) still needs it"
        )

    def test_exclusive_expired_ref_is_returned(self, landscape_db: LandscapeDB) -> None:
        """Ref used ONLY by expired runs should be returned for deletion."""
        from elspeth.core.landscape.schema import nodes_table, rows_table, runs_table
        from elspeth.core.retention.purge import PurgeManager

        db = landscape_db
        store = MockPayloadStore()
        manager = PurgeManager(db, store)

        # Ref used only by expired run
        exclusive_ref = "exclusive_to_expired_run"

        # Run A: 60 days old (expired)
        run_a_id = str(uuid4())
        node_a_id = str(uuid4())
        old_completed = datetime.now(UTC) - timedelta(days=60)

        with db.connection() as conn:
            _create_run(conn, runs_table, run_a_id, completed_at=old_completed, status=RunStatus.COMPLETED)
            _create_node(conn, nodes_table, node_a_id, run_a_id)
            _create_row(
                conn,
                rows_table,
                str(uuid4()),
                run_a_id,
                node_a_id,
                row_index=0,
                source_data_ref=exclusive_ref,
                source_data_hash="hash_excl",
            )

        expired = manager.find_expired_payload_refs(retention_days=30)

        assert exclusive_ref in expired, f"Exclusive ref {exclusive_ref} SHOULD be returned for deletion because no active run needs it"

    def test_shared_ref_excluded_when_used_by_running_run(self, landscape_db: LandscapeDB) -> None:
        """Shared ref should NOT be purged if an incomplete (running) run uses it."""
        from elspeth.core.landscape.schema import nodes_table, rows_table, runs_table
        from elspeth.core.retention.purge import PurgeManager

        db = landscape_db
        store = MockPayloadStore()
        manager = PurgeManager(db, store)

        shared_ref = "shared_with_running_run"

        # Run A: 60 days old (expired)
        run_a_id = str(uuid4())
        node_a_id = str(uuid4())
        old_completed = datetime.now(UTC) - timedelta(days=60)

        # Run B: still running (no completed_at)
        run_b_id = str(uuid4())
        node_b_id = str(uuid4())

        with db.connection() as conn:
            _create_run(conn, runs_table, run_a_id, completed_at=old_completed, status=RunStatus.COMPLETED)
            _create_node(conn, nodes_table, node_a_id, run_a_id)
            _create_row(
                conn,
                rows_table,
                str(uuid4()),
                run_a_id,
                node_a_id,
                row_index=0,
                source_data_ref=shared_ref,
            )

            # Running run (completed_at=None, status=running)
            _create_run(conn, runs_table, run_b_id, completed_at=None, status=RunStatus.RUNNING)
            _create_node(conn, nodes_table, node_b_id, run_b_id)
            _create_row(
                conn,
                rows_table,
                str(uuid4()),
                run_b_id,
                node_b_id,
                row_index=0,
                source_data_ref=shared_ref,
            )

        expired = manager.find_expired_payload_refs(retention_days=30)

        assert shared_ref not in expired, (
            f"Shared ref {shared_ref} should NOT be returned for deletion because a running run still needs it"
        )


class TestCallJoinRunIsolation:
    """Tests for cross-run isolation in call_join.

    BUG: The call_join uses node_id alone to join node_states to nodes.
    When the same node_id is used in multiple runs, this creates ambiguous
    joins that can cause refs from expired runs to incorrectly appear in
    the active_refs set, preventing their purge.

    The fix: Use node_states.run_id directly instead of joining through
    nodes table (the run_id is denormalized on node_states for this purpose).
    """

    def test_expired_call_ref_returned_when_same_node_id_exists_in_recent_run(self, landscape_db: LandscapeDB) -> None:
        """BUG TEST: Expired run's call ref should be returned even when recent run has same node_id.

        Scenario:
        - Run A (60 days old, expired) has node_id="shared-node-id" with call response_ref="ref-A"
        - Run B (10 days old, recent) has the SAME node_id="shared-node-id" with call response_ref="ref-B"

        Expected (correct behavior):
        - ref-A should be returned for purge (Run A is expired)
        - ref-B should NOT be returned (Run B is recent, within retention)

        Bug behavior:
        The ambiguous join on node_id causes Run A's call to also match Run B's
        nodes row, which joins to Run B (active). This incorrectly puts ref-A in
        both expired_refs AND active_refs. Set difference removes it, so ref-A
        is NOT returned - causing DATA LOSS (refs that should be purged aren't).
        """
        from elspeth.core.landscape.schema import (
            calls_table,
            node_states_table,
            nodes_table,
            rows_table,
            runs_table,
            tokens_table,
        )
        from elspeth.core.retention.purge import PurgeManager

        db = landscape_db
        store = MockPayloadStore()
        manager = PurgeManager(db, store)

        # Critical: Both runs use the SAME node_id (simulating pipeline reuse)
        shared_node_id = "shared-node-id-for-isolation-test"

        # Run A: 60 days old (EXPIRED with 30-day retention)
        run_a_id = "run-A-expired"
        row_a_id = "row-A"
        token_a_id = "token-A"
        state_a_id = "state-A"
        call_a_id = "call-A"
        old_completed = datetime.now(UTC) - timedelta(days=60)

        # Run B: 10 days old (RECENT, within 30-day retention)
        run_b_id = "run-B-recent"
        row_b_id = "row-B"
        token_b_id = "token-B"
        state_b_id = "state-B"
        call_b_id = "call-B"
        recent_completed = datetime.now(UTC) - timedelta(days=10)

        with db.connection() as conn:
            # === Run A (expired) ===
            _create_run(conn, runs_table, run_a_id, completed_at=old_completed, status=RunStatus.COMPLETED)
            _create_node(conn, nodes_table, shared_node_id, run_a_id)  # Same node_id!
            _create_row(conn, rows_table, row_a_id, run_a_id, shared_node_id, row_index=0)
            _create_token(conn, tokens_table, token_a_id, row_a_id)
            _create_state(conn, node_states_table, state_a_id, token_a_id, shared_node_id, run_a_id)
            _create_call(conn, calls_table, call_a_id, state_a_id, response_ref="ref-A-should-be-purged")

            # === Run B (recent) ===
            _create_run(conn, runs_table, run_b_id, completed_at=recent_completed, status=RunStatus.COMPLETED)
            _create_node(conn, nodes_table, shared_node_id, run_b_id)  # Same node_id!
            _create_row(conn, rows_table, row_b_id, run_b_id, shared_node_id, row_index=0)
            _create_token(conn, tokens_table, token_b_id, row_b_id)
            _create_state(conn, node_states_table, state_b_id, token_b_id, shared_node_id, run_b_id)
            _create_call(conn, calls_table, call_b_id, state_b_id, response_ref="ref-B-keep")

        # Find expired refs with 30-day retention
        expired = manager.find_expired_payload_refs(retention_days=30)

        # ref-A SHOULD be returned (Run A is expired)
        # BUG: The ambiguous join causes ref-A to also appear in active_refs,
        # so set difference removes it and it's NOT returned
        assert "ref-A-should-be-purged" in expired, (
            f"BUG DETECTED: ref-A from expired Run A should be returned for purge, "
            f"but was incorrectly excluded due to ambiguous node_id join. "
            f"Got expired refs: {expired}"
        )

        # ref-B should NOT be returned (Run B is recent)
        assert "ref-B-keep" not in expired, f"ref-B from recent Run B should NOT be returned for purge. Got expired refs: {expired}"

    def test_recent_call_ref_not_returned_when_expired_run_has_same_node_id(self, landscape_db: LandscapeDB) -> None:
        """Verify recent run's call ref is protected even when expired run has same node_id.

        This is the inverse scenario - verifying that we don't accidentally purge
        recent data due to the join ambiguity.
        """
        from elspeth.core.landscape.schema import (
            calls_table,
            node_states_table,
            nodes_table,
            rows_table,
            runs_table,
            tokens_table,
        )
        from elspeth.core.retention.purge import PurgeManager

        db = landscape_db
        store = MockPayloadStore()
        manager = PurgeManager(db, store)

        # Same node_id in both runs
        shared_node_id = "shared-node-for-inverse-test"

        # Run A: expired
        run_a_id = "run-A-old"
        old_completed = datetime.now(UTC) - timedelta(days=60)

        # Run B: recent
        run_b_id = "run-B-new"
        recent_completed = datetime.now(UTC) - timedelta(days=10)

        with db.connection() as conn:
            # Run A (expired)
            _create_run(conn, runs_table, run_a_id, completed_at=old_completed, status=RunStatus.COMPLETED)
            _create_node(conn, nodes_table, shared_node_id, run_a_id)
            row_a_id = str(uuid4())
            token_a_id = str(uuid4())
            state_a_id = str(uuid4())
            _create_row(conn, rows_table, row_a_id, run_a_id, shared_node_id, row_index=0)
            _create_token(conn, tokens_table, token_a_id, row_a_id)
            _create_state(conn, node_states_table, state_a_id, token_a_id, shared_node_id, run_a_id)
            _create_call(conn, calls_table, str(uuid4()), state_a_id, response_ref="ref-old")

            # Run B (recent)
            _create_run(conn, runs_table, run_b_id, completed_at=recent_completed, status=RunStatus.COMPLETED)
            _create_node(conn, nodes_table, shared_node_id, run_b_id)
            row_b_id = str(uuid4())
            token_b_id = str(uuid4())
            state_b_id = str(uuid4())
            _create_row(conn, rows_table, row_b_id, run_b_id, shared_node_id, row_index=0)
            _create_token(conn, tokens_table, token_b_id, row_b_id)
            _create_state(conn, node_states_table, state_b_id, token_b_id, shared_node_id, run_b_id)
            _create_call(conn, calls_table, str(uuid4()), state_b_id, response_ref="ref-new-protect")

        expired = manager.find_expired_payload_refs(retention_days=30)

        # ref-new-protect should NEVER be returned (Run B is within retention)
        assert "ref-new-protect" not in expired, f"ref-new-protect from recent Run B should NOT be returned for purge. Got: {expired}"


class TestRoutingJoinRunIsolation:
    """Tests for cross-run isolation in routing_join.

    BUG: The routing_join uses node_id alone to join node_states to nodes.
    Same bug pattern as call_join - ambiguous joins when node_id is reused.

    The fix: Use node_states.run_id directly instead of joining through nodes.
    """

    def test_expired_routing_ref_returned_when_same_node_id_exists_in_recent_run(self, landscape_db: LandscapeDB) -> None:
        """BUG TEST: Expired run's routing ref should be returned even when recent run has same node_id.

        Same pattern as call_join test - the ambiguous join causes routing refs
        from expired runs to incorrectly appear in active_refs, preventing purge.
        """
        from elspeth.core.landscape.schema import (
            edges_table,
            node_states_table,
            nodes_table,
            routing_events_table,
            rows_table,
            runs_table,
            tokens_table,
        )
        from elspeth.core.retention.purge import PurgeManager

        db = landscape_db
        store = MockPayloadStore()
        manager = PurgeManager(db, store)

        # Both runs use the same node_id
        shared_node_id = "shared-node-routing-test"
        sink_node_id_a = "sink-A"  # Different per run (sinks can't be shared)
        sink_node_id_b = "sink-B"

        # Run A: 60 days old (EXPIRED)
        run_a_id = "run-routing-A-expired"
        old_completed = datetime.now(UTC) - timedelta(days=60)

        # Run B: 10 days old (RECENT)
        run_b_id = "run-routing-B-recent"
        recent_completed = datetime.now(UTC) - timedelta(days=10)

        with db.connection() as conn:
            # === Run A (expired) ===
            _create_run(conn, runs_table, run_a_id, completed_at=old_completed, status=RunStatus.COMPLETED)
            _create_node(conn, nodes_table, shared_node_id, run_a_id)  # Same node_id!
            _create_node(conn, nodes_table, sink_node_id_a, run_a_id)
            row_a_id = str(uuid4())
            token_a_id = str(uuid4())
            state_a_id = str(uuid4())
            edge_a_id = str(uuid4())
            event_a_id = str(uuid4())
            _create_row(conn, rows_table, row_a_id, run_a_id, shared_node_id, row_index=0)
            _create_token(conn, tokens_table, token_a_id, row_a_id)
            _create_state(conn, node_states_table, state_a_id, token_a_id, shared_node_id, run_a_id)
            _create_edge(conn, edges_table, edge_a_id, run_a_id, shared_node_id, sink_node_id_a)
            _create_routing_event(
                conn, routing_events_table, event_a_id, state_a_id, edge_a_id, reason_ref="routing-ref-A-should-be-purged"
            )

            # === Run B (recent) ===
            _create_run(conn, runs_table, run_b_id, completed_at=recent_completed, status=RunStatus.COMPLETED)
            _create_node(conn, nodes_table, shared_node_id, run_b_id)  # Same node_id!
            _create_node(conn, nodes_table, sink_node_id_b, run_b_id)
            row_b_id = str(uuid4())
            token_b_id = str(uuid4())
            state_b_id = str(uuid4())
            edge_b_id = str(uuid4())
            event_b_id = str(uuid4())
            _create_row(conn, rows_table, row_b_id, run_b_id, shared_node_id, row_index=0)
            _create_token(conn, tokens_table, token_b_id, row_b_id)
            _create_state(conn, node_states_table, state_b_id, token_b_id, shared_node_id, run_b_id)
            _create_edge(conn, edges_table, edge_b_id, run_b_id, shared_node_id, sink_node_id_b)
            _create_routing_event(conn, routing_events_table, event_b_id, state_b_id, edge_b_id, reason_ref="routing-ref-B-keep")

        # Find expired refs
        expired = manager.find_expired_payload_refs(retention_days=30)

        # routing-ref-A SHOULD be returned (Run A is expired)
        # BUG: Ambiguous join puts it in both expired_refs and active_refs
        assert "routing-ref-A-should-be-purged" in expired, (
            f"BUG DETECTED: routing-ref-A from expired Run A should be returned for purge, "
            f"but was incorrectly excluded due to ambiguous node_id join. "
            f"Got expired refs: {expired}"
        )

        # routing-ref-B should NOT be returned (Run B is recent)
        assert "routing-ref-B-keep" not in expired, f"routing-ref-B from recent Run B should NOT be returned for purge. Got: {expired}"


class TestPurgeUpdatesReproducibilityGrade:
    """Tests for BUG P1-2026-01-22-reproducibility-grade-not-updated-after-purge.

    After purging payloads, nondeterministic runs can no longer be replayed
    (we don't have the recorded responses). The reproducibility_grade must
    degrade from REPLAY_REPRODUCIBLE to ATTRIBUTABLE_ONLY.
    """

    def test_purge_degrades_replay_reproducible_to_attributable_only(self, landscape_db: LandscapeDB) -> None:
        """Purge of REPLAY_REPRODUCIBLE run's payloads degrades grade to ATTRIBUTABLE_ONLY.

        This is the core bug fix test: when a run with nondeterministic calls
        (REPLAY_REPRODUCIBLE) has its payloads purged, it can no longer be
        replayed, so the grade must degrade to ATTRIBUTABLE_ONLY.
        """
        from sqlalchemy import select

        from elspeth.core.landscape.reproducibility import ReproducibilityGrade, set_run_grade
        from elspeth.core.landscape.schema import nodes_table, rows_table, runs_table
        from elspeth.core.retention.purge import PurgeManager

        db = landscape_db
        store = MockPayloadStore()
        manager = PurgeManager(db, store)

        # Create a run completed 60 days ago (expired with 30-day retention)
        run_id = str(uuid4())
        node_id = str(uuid4())
        old_completed_at = datetime.now(UTC) - timedelta(days=60)

        # Store a payload and get ref
        payload_ref = store.store(b"source row content for replay")

        with db.connection() as conn:
            _create_run(conn, runs_table, run_id, completed_at=old_completed_at, status=RunStatus.COMPLETED)
            _create_node(conn, nodes_table, node_id, run_id)
            _create_row(
                conn,
                rows_table,
                row_id=str(uuid4()),
                run_id=run_id,
                node_id=node_id,
                row_index=0,
                source_data_ref=payload_ref,
                source_data_hash="hash_for_grade_test",
            )

        # Set reproducibility grade to REPLAY_REPRODUCIBLE
        # (simulating a run with nondeterministic calls like LLM)
        set_run_grade(db, run_id, ReproducibilityGrade.REPLAY_REPRODUCIBLE)

        # Verify initial grade
        with db.connection() as conn:
            result = conn.execute(select(runs_table.c.reproducibility_grade).where(runs_table.c.run_id == run_id))
            initial_grade = result.scalar()
        assert initial_grade == ReproducibilityGrade.REPLAY_REPRODUCIBLE.value

        # Purge the payload
        result = manager.purge_payloads([payload_ref])
        assert result.deleted_count == 1

        # Verify grade was degraded to ATTRIBUTABLE_ONLY
        with db.connection() as conn:
            result = conn.execute(select(runs_table.c.reproducibility_grade).where(runs_table.c.run_id == run_id))
            updated_grade = result.scalar()

        assert updated_grade == ReproducibilityGrade.ATTRIBUTABLE_ONLY.value, (
            f"Expected grade to degrade from REPLAY_REPRODUCIBLE to ATTRIBUTABLE_ONLY after purge, but got {updated_grade}"
        )

    def test_purge_keeps_full_reproducible_unchanged(self, landscape_db: LandscapeDB) -> None:
        """Purge of FULL_REPRODUCIBLE run's payloads does not degrade grade.

        FULL_REPRODUCIBLE runs are fully deterministic and don't depend on
        payloads for replay - they can be re-executed from scratch.
        """
        from sqlalchemy import select

        from elspeth.core.landscape.reproducibility import ReproducibilityGrade, set_run_grade
        from elspeth.core.landscape.schema import nodes_table, rows_table, runs_table
        from elspeth.core.retention.purge import PurgeManager

        db = landscape_db
        store = MockPayloadStore()
        manager = PurgeManager(db, store)

        run_id = str(uuid4())
        node_id = str(uuid4())
        old_completed_at = datetime.now(UTC) - timedelta(days=60)
        payload_ref = store.store(b"deterministic row content")

        with db.connection() as conn:
            _create_run(conn, runs_table, run_id, completed_at=old_completed_at, status=RunStatus.COMPLETED)
            _create_node(conn, nodes_table, node_id, run_id)
            _create_row(
                conn,
                rows_table,
                row_id=str(uuid4()),
                run_id=run_id,
                node_id=node_id,
                row_index=0,
                source_data_ref=payload_ref,
            )

        # Set grade to FULL_REPRODUCIBLE (deterministic run)
        set_run_grade(db, run_id, ReproducibilityGrade.FULL_REPRODUCIBLE)

        # Purge the payload
        manager.purge_payloads([payload_ref])

        # Verify grade remains FULL_REPRODUCIBLE
        with db.connection() as conn:
            result = conn.execute(select(runs_table.c.reproducibility_grade).where(runs_table.c.run_id == run_id))
            updated_grade = result.scalar()

        assert updated_grade == ReproducibilityGrade.FULL_REPRODUCIBLE.value

    def test_purge_keeps_attributable_only_unchanged(self, landscape_db: LandscapeDB) -> None:
        """Purge of ATTRIBUTABLE_ONLY run's payloads does not degrade grade further.

        ATTRIBUTABLE_ONLY is already the lowest grade.
        """
        from sqlalchemy import select

        from elspeth.core.landscape.reproducibility import ReproducibilityGrade, set_run_grade
        from elspeth.core.landscape.schema import nodes_table, rows_table, runs_table
        from elspeth.core.retention.purge import PurgeManager

        db = landscape_db
        store = MockPayloadStore()
        manager = PurgeManager(db, store)

        run_id = str(uuid4())
        node_id = str(uuid4())
        old_completed_at = datetime.now(UTC) - timedelta(days=60)
        payload_ref = store.store(b"already degraded content")

        with db.connection() as conn:
            _create_run(conn, runs_table, run_id, completed_at=old_completed_at, status=RunStatus.COMPLETED)
            _create_node(conn, nodes_table, node_id, run_id)
            _create_row(
                conn,
                rows_table,
                row_id=str(uuid4()),
                run_id=run_id,
                node_id=node_id,
                row_index=0,
                source_data_ref=payload_ref,
            )

        # Set grade to ATTRIBUTABLE_ONLY (already degraded)
        set_run_grade(db, run_id, ReproducibilityGrade.ATTRIBUTABLE_ONLY)

        # Purge the payload
        manager.purge_payloads([payload_ref])

        # Verify grade remains ATTRIBUTABLE_ONLY
        with db.connection() as conn:
            result = conn.execute(select(runs_table.c.reproducibility_grade).where(runs_table.c.run_id == run_id))
            updated_grade = result.scalar()

        assert updated_grade == ReproducibilityGrade.ATTRIBUTABLE_ONLY.value

    def test_purge_updates_multiple_affected_runs(self, landscape_db: LandscapeDB) -> None:
        """Purge updates grades for ALL runs affected by the refs.

        When a payload is shared across multiple expired runs (content-addressable
        storage), purging it should update grades for all affected runs.
        """
        from sqlalchemy import select

        from elspeth.core.landscape.reproducibility import ReproducibilityGrade, set_run_grade
        from elspeth.core.landscape.schema import nodes_table, rows_table, runs_table
        from elspeth.core.retention.purge import PurgeManager

        db = landscape_db
        store = MockPayloadStore()
        manager = PurgeManager(db, store)

        # Shared payload ref (content-addressable: same content = same hash)
        shared_payload_ref = store.store(b"shared content across runs")

        old_completed_at = datetime.now(UTC) - timedelta(days=60)

        # Create two runs that share the same payload ref
        run_a_id = str(uuid4())
        run_b_id = str(uuid4())
        node_a_id = str(uuid4())
        node_b_id = str(uuid4())

        with db.connection() as conn:
            # Run A
            _create_run(conn, runs_table, run_a_id, completed_at=old_completed_at, status=RunStatus.COMPLETED)
            _create_node(conn, nodes_table, node_a_id, run_a_id)
            _create_row(
                conn,
                rows_table,
                row_id=str(uuid4()),
                run_id=run_a_id,
                node_id=node_a_id,
                row_index=0,
                source_data_ref=shared_payload_ref,
            )

            # Run B
            _create_run(conn, runs_table, run_b_id, completed_at=old_completed_at, status=RunStatus.COMPLETED)
            _create_node(conn, nodes_table, node_b_id, run_b_id)
            _create_row(
                conn,
                rows_table,
                row_id=str(uuid4()),
                run_id=run_b_id,
                node_id=node_b_id,
                row_index=0,
                source_data_ref=shared_payload_ref,
            )

        # Set both runs to REPLAY_REPRODUCIBLE
        set_run_grade(db, run_a_id, ReproducibilityGrade.REPLAY_REPRODUCIBLE)
        set_run_grade(db, run_b_id, ReproducibilityGrade.REPLAY_REPRODUCIBLE)

        # Purge the shared payload
        manager.purge_payloads([shared_payload_ref])

        # Verify BOTH runs had their grades degraded
        with db.connection() as conn:
            result_a = conn.execute(select(runs_table.c.reproducibility_grade).where(runs_table.c.run_id == run_a_id))
            grade_a = result_a.scalar()

            result_b = conn.execute(select(runs_table.c.reproducibility_grade).where(runs_table.c.run_id == run_b_id))
            grade_b = result_b.scalar()

        assert grade_a == ReproducibilityGrade.ATTRIBUTABLE_ONLY.value, f"Run A grade not degraded: {grade_a}"
        assert grade_b == ReproducibilityGrade.ATTRIBUTABLE_ONLY.value, f"Run B grade not degraded: {grade_b}"

    def test_purge_empty_refs_does_not_update_any_grades(self, landscape_db: LandscapeDB) -> None:
        """Purge with empty refs list does not modify any run grades."""
        from sqlalchemy import select

        from elspeth.core.landscape.reproducibility import ReproducibilityGrade, set_run_grade
        from elspeth.core.landscape.schema import nodes_table, rows_table, runs_table
        from elspeth.core.retention.purge import PurgeManager

        db = landscape_db
        store = MockPayloadStore()
        manager = PurgeManager(db, store)

        run_id = str(uuid4())
        node_id = str(uuid4())
        old_completed_at = datetime.now(UTC) - timedelta(days=60)
        payload_ref = store.store(b"content that won't be purged")

        with db.connection() as conn:
            _create_run(conn, runs_table, run_id, completed_at=old_completed_at, status=RunStatus.COMPLETED)
            _create_node(conn, nodes_table, node_id, run_id)
            _create_row(
                conn,
                rows_table,
                row_id=str(uuid4()),
                run_id=run_id,
                node_id=node_id,
                row_index=0,
                source_data_ref=payload_ref,
            )

        set_run_grade(db, run_id, ReproducibilityGrade.REPLAY_REPRODUCIBLE)

        # Purge with empty list
        manager.purge_payloads([])

        # Verify grade unchanged
        with db.connection() as conn:
            result = conn.execute(select(runs_table.c.reproducibility_grade).where(runs_table.c.run_id == run_id))
            grade = result.scalar()

        assert grade == ReproducibilityGrade.REPLAY_REPRODUCIBLE.value

    def test_purge_call_payloads_also_degrades_grade(self, landscape_db: LandscapeDB) -> None:
        """Purging call payloads (LLM responses) also degrades reproducibility grade.

        This verifies that the grade degradation works for all payload types,
        not just row payloads.
        """
        from sqlalchemy import select

        from elspeth.core.landscape.reproducibility import ReproducibilityGrade, set_run_grade
        from elspeth.core.landscape.schema import (
            calls_table,
            node_states_table,
            nodes_table,
            rows_table,
            runs_table,
            tokens_table,
        )
        from elspeth.core.retention.purge import PurgeManager

        db = landscape_db
        store = MockPayloadStore()
        manager = PurgeManager(db, store)

        run_id = str(uuid4())
        node_id = str(uuid4())
        row_id = str(uuid4())
        token_id = str(uuid4())
        state_id = str(uuid4())
        call_id = str(uuid4())
        old_completed_at = datetime.now(UTC) - timedelta(days=60)

        # Store call response payload (simulating LLM response)
        call_response_ref = store.store(b"LLM response content")

        with db.connection() as conn:
            _create_run(conn, runs_table, run_id, completed_at=old_completed_at, status=RunStatus.COMPLETED)
            _create_node(conn, nodes_table, node_id, run_id)
            _create_row(conn, rows_table, row_id, run_id, node_id, row_index=0)
            _create_token(conn, tokens_table, token_id, row_id)
            _create_state(conn, node_states_table, state_id, token_id, node_id, run_id)
            _create_call(conn, calls_table, call_id, state_id, response_ref=call_response_ref)

        # Set to REPLAY_REPRODUCIBLE (has nondeterministic LLM calls)
        set_run_grade(db, run_id, ReproducibilityGrade.REPLAY_REPRODUCIBLE)

        # Purge the call response payload
        manager.purge_payloads([call_response_ref])

        # Verify grade degraded
        with db.connection() as conn:
            result = conn.execute(select(runs_table.c.reproducibility_grade).where(runs_table.c.run_id == run_id))
            updated_grade = result.scalar()

        assert updated_grade == ReproducibilityGrade.ATTRIBUTABLE_ONLY.value

    def test_purge_does_not_degrade_grade_when_deletion_fails(self, landscape_db: LandscapeDB) -> None:
        """Grade should NOT degrade when payload deletion fails.

        BUG P2-2026-01-28-grade-update-on-failed-deletion:
        If payload_store.delete(ref) returns False, the run still has its
        payloads and IS replayable. The grade should NOT be downgraded.

        This tests the scenario where:
        1. Run A has payload ref-A
        2. ref-A deletion FAILS (store returns False)
        3. Run A's payloads still exist
        4. Run A's grade should remain REPLAY_REPRODUCIBLE (not ATTRIBUTABLE_ONLY)
        """
        from sqlalchemy import select

        from elspeth.core.landscape.reproducibility import ReproducibilityGrade, set_run_grade
        from elspeth.core.landscape.schema import nodes_table, rows_table, runs_table
        from elspeth.core.retention.purge import PurgeManager

        db = landscape_db

        # Create a store that fails to delete specific refs
        class FailingPayloadStore:
            """Mock that fails to delete all refs."""

            def __init__(self) -> None:
                self._storage: dict[str, bytes] = {}

            def store(self, content: bytes) -> str:
                import hashlib

                content_hash = hashlib.sha256(content).hexdigest()
                self._storage[content_hash] = content
                return content_hash

            def exists(self, content_hash: str) -> bool:
                return content_hash in self._storage

            def delete(self, content_hash: str) -> bool:
                # Always fail - simulates I/O error or permission issue
                return False

        store = FailingPayloadStore()

        # Create a run completed 60 days ago (eligible for purge)
        run_id = str(uuid4())
        node_id = str(uuid4())
        row_id = str(uuid4())
        old_completed_at = datetime.now(UTC) - timedelta(days=60)

        # Store payload and get ref
        payload_ref = store.store(b"source row content")

        with db.connection() as conn:
            _create_run(
                conn,
                runs_table,
                run_id,
                completed_at=old_completed_at,
                status=RunStatus.COMPLETED,
            )
            _create_node(conn, nodes_table, node_id, run_id)
            _create_row(
                conn,
                rows_table,
                row_id=row_id,
                run_id=run_id,
                node_id=node_id,
                row_index=0,
                source_data_ref=payload_ref,
                source_data_hash="hash_for_run",
            )

        # Set to REPLAY_REPRODUCIBLE (has nondeterministic calls)
        set_run_grade(db, run_id, ReproducibilityGrade.REPLAY_REPRODUCIBLE)

        # Attempt purge - deletion will FAIL
        manager = PurgeManager(db, store)
        result = manager.purge_payloads([payload_ref])

        # Verify deletion failed
        assert result.deleted_count == 0
        assert result.failed_refs == [payload_ref]
        assert store.exists(payload_ref), "Payload should still exist after failed deletion"

        # CRITICAL: Grade should NOT be downgraded because payloads still exist
        with db.connection() as conn:
            query = select(runs_table.c.reproducibility_grade).where(runs_table.c.run_id == run_id)
            result_row = conn.execute(query)
            updated_grade = result_row.scalar()

        assert updated_grade == ReproducibilityGrade.REPLAY_REPRODUCIBLE.value, (
            f"Grade should remain REPLAY_REPRODUCIBLE when deletion fails (payloads still exist), but got {updated_grade}"
        )

    def test_purge_degrades_grade_when_some_deletions_succeed(self, landscape_db: LandscapeDB) -> None:
        """Grade should degrade if ANY payload for the run is deleted, even if others fail.

        This tests the scenario where a run has multiple payloads and some
        deletions succeed while others fail. The run is affected by the
        successful deletions, so its grade should be downgraded.

        This is the correct behavior because:
        - Payloads can include source data, LLM responses, routing reasons
        - If ANY required payload for replay is deleted, replay is incomplete
        - Conservative downgrade ensures we don't claim replayability we can't deliver
        """
        from sqlalchemy import select

        from elspeth.core.landscape.reproducibility import ReproducibilityGrade, set_run_grade
        from elspeth.core.landscape.schema import nodes_table, rows_table, runs_table
        from elspeth.core.retention.purge import PurgeManager

        db = landscape_db

        # Create a store that fails to delete specific refs
        class PartialFailingPayloadStore:
            """Mock that fails to delete specific refs but succeeds for others."""

            def __init__(self, fail_refs: set[str]) -> None:
                self._storage: dict[str, bytes] = {}
                self._fail_refs = fail_refs

            def store(self, content: bytes) -> str:
                import hashlib

                content_hash = hashlib.sha256(content).hexdigest()
                self._storage[content_hash] = content
                return content_hash

            def exists(self, content_hash: str) -> bool:
                return content_hash in self._storage

            def delete(self, content_hash: str) -> bool:
                if content_hash in self._fail_refs:
                    return False
                if content_hash in self._storage:
                    del self._storage[content_hash]
                    return True
                return False

        store = PartialFailingPayloadStore(fail_refs=set())

        # Create a run completed 60 days ago (eligible for purge)
        run_id = str(uuid4())
        node_id = str(uuid4())
        old_completed_at = datetime.now(UTC) - timedelta(days=60)

        # Store two payloads - one will succeed, one will fail
        success_ref = store.store(b"will be deleted")
        fail_ref = store.store(b"will fail to delete")
        store._fail_refs.add(fail_ref)

        with db.connection() as conn:
            _create_run(
                conn,
                runs_table,
                run_id,
                completed_at=old_completed_at,
                status=RunStatus.COMPLETED,
            )
            _create_node(conn, nodes_table, node_id, run_id)
            # Create two rows with different payload refs
            _create_row(
                conn,
                rows_table,
                row_id=str(uuid4()),
                run_id=run_id,
                node_id=node_id,
                row_index=0,
                source_data_ref=success_ref,
                source_data_hash="hash_success",
            )
            _create_row(
                conn,
                rows_table,
                row_id=str(uuid4()),
                run_id=run_id,
                node_id=node_id,
                row_index=1,
                source_data_ref=fail_ref,
                source_data_hash="hash_fail",
            )

        # Set to REPLAY_REPRODUCIBLE
        set_run_grade(db, run_id, ReproducibilityGrade.REPLAY_REPRODUCIBLE)

        # Attempt purge - one succeeds, one fails
        manager = PurgeManager(db, store)
        result = manager.purge_payloads([success_ref, fail_ref])

        # Verify partial success
        assert result.deleted_count == 1
        assert result.failed_refs == [fail_ref]

        # Grade SHOULD be downgraded because SOME payloads were deleted
        # The run can no longer be fully replayed
        with db.connection() as conn:
            query = select(runs_table.c.reproducibility_grade).where(runs_table.c.run_id == run_id)
            result_row = conn.execute(query)
            updated_grade = result_row.scalar()

        assert updated_grade == ReproducibilityGrade.ATTRIBUTABLE_ONLY.value, (
            f"Grade should degrade to ATTRIBUTABLE_ONLY when some payloads are deleted, but got {updated_grade}"
        )


class TestFailedRunsIncludedInPurge:
    """Tests for P2-2026-01-31-failed-runs-excluded-from-purge.

    BUG: Purge uses status == "completed" for expired condition, so failed runs
    are treated as "active" regardless of age. Failed runs older than retention
    cutoff should be eligible for purge.
    """

    def test_failed_run_payloads_are_eligible_for_purge(self, landscape_db: LandscapeDB) -> None:
        """Failed runs older than retention period should have their payloads purged.

        BUG: Currently, failed runs are excluded from purge because the code
        checks `status == "completed"`. A failed run from 60 days ago should
        be eligible for purge with 30-day retention.
        """
        from elspeth.core.landscape.schema import nodes_table, rows_table, runs_table
        from elspeth.core.retention.purge import PurgeManager

        db = landscape_db
        store = MockPayloadStore()
        manager = PurgeManager(db, store)

        # Create a FAILED run completed 60 days ago
        run_id = str(uuid4())
        node_id = str(uuid4())
        old_completed_at = datetime.now(UTC) - timedelta(days=60)
        test_ref = f"ref_for_failed_run_payload_{uuid4()}"

        with db.connection() as conn:
            _create_run(
                conn,
                runs_table,
                run_id,
                completed_at=old_completed_at,
                status=RunStatus.FAILED,  # <-- FAILED, not COMPLETED
            )
            _create_node(conn, nodes_table, node_id, run_id)
            _create_row(
                conn,
                rows_table,
                row_id=str(uuid4()),
                run_id=run_id,
                node_id=node_id,
                row_index=0,
                source_data_ref=test_ref,
                source_data_hash="hash_failed_run",
            )

        # Find payloads older than 30 days - should include FAILED run's payloads
        expired = manager.find_expired_payload_refs(retention_days=30)

        assert test_ref in expired, (
            f"BUG: Payload {test_ref} from FAILED run (60 days old) should be eligible for purge "
            f"with 30-day retention, but was excluded. Got expired refs: {expired}"
        )

    def test_failed_run_does_not_protect_shared_refs(self, landscape_db: LandscapeDB) -> None:
        """Failed runs should NOT protect shared payload refs from purge.

        BUG: Currently, failed runs are treated as "active" which incorrectly
        protects shared refs. If Run A (completed, expired) shares a ref with
        Run B (failed, expired), the ref should be purged.
        """
        from elspeth.core.landscape.schema import nodes_table, rows_table, runs_table
        from elspeth.core.retention.purge import PurgeManager

        db = landscape_db
        store = MockPayloadStore()
        manager = PurgeManager(db, store)

        # Shared ref between two old runs
        shared_ref = f"shared_ref_completed_and_failed_{uuid4()}"
        old_completed_at = datetime.now(UTC) - timedelta(days=60)

        # Run A: COMPLETED and expired
        run_a_id = str(uuid4())
        node_a_id = str(uuid4())

        # Run B: FAILED and expired (same age)
        run_b_id = str(uuid4())
        node_b_id = str(uuid4())

        with db.connection() as conn:
            # Run A - completed, 60 days old
            _create_run(conn, runs_table, run_a_id, completed_at=old_completed_at, status=RunStatus.COMPLETED)
            _create_node(conn, nodes_table, node_a_id, run_a_id)
            _create_row(
                conn,
                rows_table,
                str(uuid4()),
                run_a_id,
                node_a_id,
                row_index=0,
                source_data_ref=shared_ref,
            )

            # Run B - FAILED, 60 days old
            _create_run(conn, runs_table, run_b_id, completed_at=old_completed_at, status=RunStatus.FAILED)
            _create_node(conn, nodes_table, node_b_id, run_b_id)
            _create_row(
                conn,
                rows_table,
                str(uuid4()),
                run_b_id,
                node_b_id,
                row_index=0,
                source_data_ref=shared_ref,
            )

        expired = manager.find_expired_payload_refs(retention_days=30)

        assert shared_ref in expired, (
            f"BUG: Shared ref {shared_ref} should be eligible for purge because BOTH runs are expired "
            f"(one completed, one failed, both 60 days old). But failed run incorrectly protects it. "
            f"Got expired refs: {expired}"
        )

    def test_running_run_still_protects_refs(self, landscape_db: LandscapeDB) -> None:
        """Running (incomplete) runs should still protect their payload refs.

        This is a regression test - we must NOT accidentally include running runs
        in the expired set when fixing the failed run bug.
        """
        from elspeth.core.landscape.schema import nodes_table, rows_table, runs_table
        from elspeth.core.retention.purge import PurgeManager

        db = landscape_db
        store = MockPayloadStore()
        manager = PurgeManager(db, store)

        # Create a run that is still RUNNING (no completed_at)
        run_id = str(uuid4())
        node_id = str(uuid4())
        test_ref = f"ref_for_running_run_{uuid4()}"

        with db.connection() as conn:
            _create_run(
                conn,
                runs_table,
                run_id,
                completed_at=None,  # Still running
                status=RunStatus.RUNNING,
            )
            _create_node(conn, nodes_table, node_id, run_id)
            _create_row(
                conn,
                rows_table,
                row_id=str(uuid4()),
                run_id=run_id,
                node_id=node_id,
                row_index=0,
                source_data_ref=test_ref,
            )

        # Find payloads - running run's refs should NOT be included
        expired = manager.find_expired_payload_refs(retention_days=30)

        assert test_ref not in expired, f"Running run's payload {test_ref} should NOT be eligible for purge. Got expired refs: {expired}"


class TestPurgeIOErrorHandling:
    """Tests for P2-2026-01-31-purge-aborts-on-io-error.

    BUG: Calls to payload_store.exists() and delete() are not wrapped in
    try/except. An OSError or PermissionError aborts the purge loop, leaving
    the system in an inconsistent state.
    """

    def test_purge_continues_after_exists_raises_exception(self, landscape_db: LandscapeDB) -> None:
        """Purge should continue processing refs after exists() throws exception.

        BUG: Currently, an exception from exists() aborts the entire loop.
        The purge should catch the exception, record it as a failure, and
        continue with remaining refs.
        """
        from elspeth.core.retention.purge import PurgeManager

        db = landscape_db

        class ExistsRaisingPayloadStore:
            """Mock that raises on exists() for specific refs."""

            def __init__(self, raise_on: set[str]) -> None:
                self._storage: dict[str, bytes] = {}
                self._raise_on = raise_on
                self.delete_calls: list[str] = []

            def store(self, content: bytes) -> str:
                import hashlib

                content_hash = hashlib.sha256(content).hexdigest()
                self._storage[content_hash] = content
                return content_hash

            def exists(self, content_hash: str) -> bool:
                if content_hash in self._raise_on:
                    raise OSError(f"I/O error checking {content_hash}")
                return content_hash in self._storage

            def delete(self, content_hash: str) -> bool:
                self.delete_calls.append(content_hash)
                if content_hash in self._storage:
                    del self._storage[content_hash]
                    return True
                return False

        store = ExistsRaisingPayloadStore(raise_on=set())

        # Store three payloads - middle one will raise on exists()
        ref1 = store.store(b"content 1")
        ref2 = store.store(b"content 2 - will raise")
        ref3 = store.store(b"content 3")

        # Mark ref2 to raise on exists()
        store._raise_on.add(ref2)

        manager = PurgeManager(db, store)
        result = manager.purge_payloads([ref1, ref2, ref3])

        # ref1 and ref3 should be deleted despite ref2 raising exception
        assert result.deleted_count == 2, f"Expected 2 deletions, got {result.deleted_count}"
        assert ref2 in result.failed_refs, f"ref2 should be in failed_refs: {result.failed_refs}"
        assert ref1 not in store._storage, "ref1 should be deleted"
        assert ref3 not in store._storage, "ref3 should be deleted"

    def test_purge_continues_after_delete_raises_exception(self, landscape_db: LandscapeDB) -> None:
        """Purge should continue processing refs after delete() throws exception.

        BUG: Currently, an exception from delete() aborts the entire loop.
        """
        from elspeth.core.retention.purge import PurgeManager

        db = landscape_db

        class DeleteRaisingPayloadStore:
            """Mock that raises on delete() for specific refs."""

            def __init__(self, raise_on: set[str]) -> None:
                self._storage: dict[str, bytes] = {}
                self._raise_on = raise_on

            def store(self, content: bytes) -> str:
                import hashlib

                content_hash = hashlib.sha256(content).hexdigest()
                self._storage[content_hash] = content
                return content_hash

            def exists(self, content_hash: str) -> bool:
                return content_hash in self._storage

            def delete(self, content_hash: str) -> bool:
                if content_hash in self._raise_on:
                    raise PermissionError(f"Permission denied deleting {content_hash}")
                if content_hash in self._storage:
                    del self._storage[content_hash]
                    return True
                return False

        store = DeleteRaisingPayloadStore(raise_on=set())

        ref1 = store.store(b"content 1")
        ref2 = store.store(b"content 2 - will raise on delete")
        ref3 = store.store(b"content 3")

        store._raise_on.add(ref2)

        manager = PurgeManager(db, store)
        result = manager.purge_payloads([ref1, ref2, ref3])

        # ref1 and ref3 should be deleted despite ref2 raising exception
        assert result.deleted_count == 2, f"Expected 2 deletions, got {result.deleted_count}"
        assert ref2 in result.failed_refs, f"ref2 should be in failed_refs: {result.failed_refs}"
        assert ref1 not in store._storage, "ref1 should be deleted"
        assert ref2 in store._storage, "ref2 should still exist (delete failed)"
        assert ref3 not in store._storage, "ref3 should be deleted"

    def test_purge_updates_grades_despite_io_errors(self, landscape_db: LandscapeDB) -> None:
        """Grade updates should happen for successful deletions even when some fail.

        BUG: Currently, an I/O error aborts the loop before grade updates.
        Even if some refs fail, runs whose payloads WERE deleted should have
        their grades updated.
        """
        from sqlalchemy import select

        from elspeth.core.landscape.reproducibility import ReproducibilityGrade, set_run_grade
        from elspeth.core.landscape.schema import nodes_table, rows_table, runs_table
        from elspeth.core.retention.purge import PurgeManager

        db = landscape_db

        class PartiallyFailingPayloadStore:
            """Mock that raises on delete() for specific refs."""

            def __init__(self, raise_on: set[str]) -> None:
                self._storage: dict[str, bytes] = {}
                self._raise_on = raise_on

            def store(self, content: bytes) -> str:
                import hashlib

                content_hash = hashlib.sha256(content).hexdigest()
                self._storage[content_hash] = content
                return content_hash

            def exists(self, content_hash: str) -> bool:
                return content_hash in self._storage

            def delete(self, content_hash: str) -> bool:
                if content_hash in self._raise_on:
                    raise OSError(f"I/O error deleting {content_hash}")
                if content_hash in self._storage:
                    del self._storage[content_hash]
                    return True
                return False

        store = PartiallyFailingPayloadStore(raise_on=set())

        # Create two runs with different payloads
        old_completed_at = datetime.now(UTC) - timedelta(days=60)

        run_success_id = str(uuid4())
        node_success_id = str(uuid4())
        success_ref = store.store(b"will be deleted successfully")

        run_fail_id = str(uuid4())
        node_fail_id = str(uuid4())
        fail_ref = store.store(b"will fail to delete")
        store._raise_on.add(fail_ref)

        with db.connection() as conn:
            # Run with successful deletion
            _create_run(conn, runs_table, run_success_id, completed_at=old_completed_at, status=RunStatus.COMPLETED)
            _create_node(conn, nodes_table, node_success_id, run_success_id)
            _create_row(
                conn,
                rows_table,
                str(uuid4()),
                run_success_id,
                node_success_id,
                row_index=0,
                source_data_ref=success_ref,
            )

            # Run with failed deletion
            _create_run(conn, runs_table, run_fail_id, completed_at=old_completed_at, status=RunStatus.COMPLETED)
            _create_node(conn, nodes_table, node_fail_id, run_fail_id)
            _create_row(
                conn,
                rows_table,
                str(uuid4()),
                run_fail_id,
                node_fail_id,
                row_index=0,
                source_data_ref=fail_ref,
            )

        # Set both to REPLAY_REPRODUCIBLE
        set_run_grade(db, run_success_id, ReproducibilityGrade.REPLAY_REPRODUCIBLE)
        set_run_grade(db, run_fail_id, ReproducibilityGrade.REPLAY_REPRODUCIBLE)

        # Purge both refs - one will succeed, one will raise exception
        manager = PurgeManager(db, store)
        result = manager.purge_payloads([success_ref, fail_ref])

        assert result.deleted_count == 1
        assert fail_ref in result.failed_refs

        # Run with successful deletion should have grade updated
        with db.connection() as conn:
            result_success = conn.execute(select(runs_table.c.reproducibility_grade).where(runs_table.c.run_id == run_success_id))
            grade_success = result_success.scalar()

            result_fail = conn.execute(select(runs_table.c.reproducibility_grade).where(runs_table.c.run_id == run_fail_id))
            grade_fail = result_fail.scalar()

        # Success run's grade should be degraded (payload deleted)
        assert grade_success == ReproducibilityGrade.ATTRIBUTABLE_ONLY.value, (
            f"Run with successful deletion should have grade degraded, but got {grade_success}"
        )

        # Failed run's grade should remain unchanged (payload still exists)
        assert grade_fail == ReproducibilityGrade.REPLAY_REPRODUCIBLE.value, (
            f"Run with failed deletion should keep its grade (payload still exists), but got {grade_fail}"
        )
