# tests/core/retention/test_purge.py
"""Tests for PurgeManager - PayloadStore retention management."""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

from sqlalchemy import Connection, Table


def _create_state(
    conn: Connection,
    states_table: Table,
    state_id: str,
    token_id: str,
    node_id: str,
) -> None:
    """Helper to create a node_state record."""
    conn.execute(
        states_table.insert().values(
            state_id=state_id,
            token_id=token_id,
            node_id=node_id,
            step_index=0,
            attempt=0,
            status="completed",
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
            call_type="llm",
            status="completed",
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
            default_mode="move",
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
            mode="move",
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
    status: str = "completed",
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
            node_type="source",
            plugin_version="1.0.0",
            determinism="deterministic",
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

    def test_find_expired_row_payloads(self) -> None:
        """Finds row payloads older than retention period."""
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.schema import nodes_table, rows_table, runs_table
        from elspeth.core.retention.purge import PurgeManager

        db = LandscapeDB.in_memory()
        store = MockPayloadStore()
        manager = PurgeManager(db, store)

        # Create a run completed 60 days ago
        run_id = str(uuid4())
        node_id = str(uuid4())
        old_completed_at = datetime.now(UTC) - timedelta(days=60)

        with db.connection() as conn:
            _create_run(
                conn,
                runs_table,
                run_id,
                completed_at=old_completed_at,
                status="completed",
            )
            _create_node(conn, nodes_table, node_id, run_id)
            _create_row(
                conn,
                rows_table,
                row_id=str(uuid4()),
                run_id=run_id,
                node_id=node_id,
                row_index=0,
                source_data_ref="ref_for_old_payload",
                source_data_hash="hash_old",
            )

        # Find payloads older than 30 days
        expired = manager.find_expired_row_payloads(retention_days=30)

        assert "ref_for_old_payload" in expired

    def test_find_expired_respects_retention(self) -> None:
        """Does not flag recent payloads."""
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.schema import nodes_table, rows_table, runs_table
        from elspeth.core.retention.purge import PurgeManager

        db = LandscapeDB.in_memory()
        store = MockPayloadStore()
        manager = PurgeManager(db, store)

        # Create a run completed 10 days ago (within retention)
        run_id = str(uuid4())
        node_id = str(uuid4())
        recent_completed_at = datetime.now(UTC) - timedelta(days=10)

        with db.connection() as conn:
            _create_run(
                conn,
                runs_table,
                run_id,
                completed_at=recent_completed_at,
                status="completed",
            )
            _create_node(conn, nodes_table, node_id, run_id)
            _create_row(
                conn,
                rows_table,
                row_id=str(uuid4()),
                run_id=run_id,
                node_id=node_id,
                row_index=0,
                source_data_ref="ref_for_recent_payload",
                source_data_hash="hash_recent",
            )

        # Find payloads older than 30 days - should NOT include recent
        expired = manager.find_expired_row_payloads(retention_days=30)

        assert "ref_for_recent_payload" not in expired

    def test_find_expired_ignores_incomplete_runs(self) -> None:
        """Does not flag payloads from incomplete runs."""
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.schema import nodes_table, rows_table, runs_table
        from elspeth.core.retention.purge import PurgeManager

        db = LandscapeDB.in_memory()
        store = MockPayloadStore()
        manager = PurgeManager(db, store)

        # Create a run from 60 days ago that is still running
        run_id = str(uuid4())
        node_id = str(uuid4())

        with db.connection() as conn:
            _create_run(
                conn,
                runs_table,
                run_id,
                completed_at=None,  # Not completed
                status="running",
            )
            _create_node(conn, nodes_table, node_id, run_id)
            _create_row(
                conn,
                rows_table,
                row_id=str(uuid4()),
                run_id=run_id,
                node_id=node_id,
                row_index=0,
                source_data_ref="ref_for_running_payload",
                source_data_hash="hash_running",
            )

        # Find payloads older than 30 days - should NOT include running
        expired = manager.find_expired_row_payloads(retention_days=30)

        assert "ref_for_running_payload" not in expired

    def test_find_expired_excludes_null_refs(self) -> None:
        """Does not include rows with null source_data_ref."""
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.schema import nodes_table, rows_table, runs_table
        from elspeth.core.retention.purge import PurgeManager

        db = LandscapeDB.in_memory()
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
                status="completed",
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

        assert len(expired) == 0

    def test_find_expired_with_as_of_date(self) -> None:
        """Uses as_of date for cutoff calculation."""
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.schema import nodes_table, rows_table, runs_table
        from elspeth.core.retention.purge import PurgeManager

        db = LandscapeDB.in_memory()
        store = MockPayloadStore()
        manager = PurgeManager(db, store)

        run_id = str(uuid4())
        node_id = str(uuid4())
        # Run completed 45 days ago
        completed_at = datetime.now(UTC) - timedelta(days=45)

        with db.connection() as conn:
            _create_run(conn, runs_table, run_id, completed_at=completed_at, status="completed")
            _create_node(conn, nodes_table, node_id, run_id)
            _create_row(
                conn,
                rows_table,
                row_id=str(uuid4()),
                run_id=run_id,
                node_id=node_id,
                row_index=0,
                source_data_ref="ref_45_days_old",
                source_data_hash="hash_45",
            )

        # With as_of=now, 30 day retention - 45 days old is expired
        expired_now = manager.find_expired_row_payloads(retention_days=30)
        assert "ref_45_days_old" in expired_now

        # With as_of=60 days ago, 30 day retention - 45 days old was not expired yet
        as_of = datetime.now(UTC) - timedelta(days=60)
        expired_past = manager.find_expired_row_payloads(retention_days=30, as_of=as_of)
        assert "ref_45_days_old" not in expired_past

    def test_find_expired_deduplicates_shared_refs(self) -> None:
        """Multiple rows referencing the same payload return only one ref.

        Content-addressed storage means identical content shares one blob,
        so multiple rows can have the same source_data_ref. The query must
        deduplicate to avoid returning the same ref multiple times.
        """
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.schema import nodes_table, rows_table, runs_table
        from elspeth.core.retention.purge import PurgeManager

        db = LandscapeDB.in_memory()
        store = MockPayloadStore()
        manager = PurgeManager(db, store)

        run_id = str(uuid4())
        node_id = str(uuid4())
        old_completed_at = datetime.now(UTC) - timedelta(days=60)

        # The shared ref that multiple rows point to (content-addressed)
        shared_ref = "shared_content_hash_abc123"

        with db.connection() as conn:
            _create_run(
                conn,
                runs_table,
                run_id,
                completed_at=old_completed_at,
                status="completed",
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
        assert len(expired) == 1


class TestPurgePayloads:
    """Tests for purge_payloads method."""

    def test_purge_payloads_deletes_content(self) -> None:
        """Purge actually deletes from PayloadStore."""
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.retention.purge import PurgeManager

        db = LandscapeDB.in_memory()
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

    def test_purge_preserves_landscape_hashes(self) -> None:
        """Purge deletes blobs but keeps hashes in Landscape."""
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.schema import nodes_table, rows_table, runs_table
        from elspeth.core.retention.purge import PurgeManager

        db = LandscapeDB.in_memory()
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
                status="completed",
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

    def test_purge_tracks_skipped_refs(self) -> None:
        """Purge tracks refs that don't exist as skipped (not failed)."""
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.retention.purge import PurgeManager

        db = LandscapeDB.in_memory()
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

    def test_purge_measures_duration(self) -> None:
        """Purge measures operation duration."""
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.retention.purge import PurgeManager

        db = LandscapeDB.in_memory()
        store = MockPayloadStore()

        ref = store.store(b"content")

        manager = PurgeManager(db, store)
        result = manager.purge_payloads([ref])

        assert result.duration_seconds >= 0

    def test_purge_empty_list(self) -> None:
        """Purge with empty list returns empty result."""
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.retention.purge import PurgeManager

        db = LandscapeDB.in_memory()
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

    def test_find_expired_includes_call_request_refs(self) -> None:
        """Expired call request payloads should be found for purge."""
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.schema import (
            calls_table,
            node_states_table,
            nodes_table,
            rows_table,
            runs_table,
            tokens_table,
        )
        from elspeth.core.retention.purge import PurgeManager

        db = LandscapeDB.in_memory()
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
            _create_run(conn, runs_table, run_id, completed_at=old_completed_at, status="completed")
            _create_node(conn, nodes_table, node_id, run_id)
            _create_row(conn, rows_table, row_id, run_id, node_id, row_index=0)
            _create_token(conn, tokens_table, token_id, row_id)
            _create_state(conn, node_states_table, state_id, token_id, node_id)
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

    def test_find_expired_includes_call_response_refs(self) -> None:
        """Expired call response payloads should be found for purge."""
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.schema import (
            calls_table,
            node_states_table,
            nodes_table,
            rows_table,
            runs_table,
            tokens_table,
        )
        from elspeth.core.retention.purge import PurgeManager

        db = LandscapeDB.in_memory()
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
            _create_run(conn, runs_table, run_id, completed_at=old_completed_at, status="completed")
            _create_node(conn, nodes_table, node_id, run_id)
            _create_row(conn, rows_table, row_id, run_id, node_id, row_index=0)
            _create_token(conn, tokens_table, token_id, row_id)
            _create_state(conn, node_states_table, state_id, token_id, node_id)
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

    def test_find_expired_includes_routing_reason_refs(self) -> None:
        """Expired routing reason payloads should be found for purge."""
        from elspeth.core.landscape.database import LandscapeDB
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

        db = LandscapeDB.in_memory()
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
            _create_run(conn, runs_table, run_id, completed_at=old_completed_at, status="completed")
            _create_node(conn, nodes_table, source_node_id, run_id)
            _create_node(conn, nodes_table, sink_node_id, run_id)
            _create_row(conn, rows_table, row_id, run_id, source_node_id, row_index=0)
            _create_token(conn, tokens_table, token_id, row_id)
            _create_state(conn, node_states_table, state_id, token_id, source_node_id)
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

    def test_find_expired_payload_refs_returns_deduplicated_union(self) -> None:
        """All payload types should be returned, deduplicated."""
        from elspeth.core.landscape.database import LandscapeDB
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

        db = LandscapeDB.in_memory()
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

        with db.connection() as conn:
            _create_run(conn, runs_table, run_id, completed_at=old_completed_at, status="completed")
            _create_node(conn, nodes_table, source_node_id, run_id)
            _create_node(conn, nodes_table, sink_node_id, run_id)
            _create_row(
                conn,
                rows_table,
                row_id,
                run_id,
                source_node_id,
                row_index=0,
                source_data_ref="row_payload_ref",
            )
            _create_token(conn, tokens_table, token_id, row_id)
            _create_state(conn, node_states_table, state_id, token_id, source_node_id)
            _create_call(
                conn,
                calls_table,
                call_id,
                state_id,
                request_ref="call_request_ref",
                response_ref="call_response_ref",
            )
            _create_edge(conn, edges_table, edge_id, run_id, source_node_id, sink_node_id)
            _create_routing_event(
                conn,
                routing_events_table,
                event_id,
                state_id,
                edge_id,
                reason_ref="routing_reason_ref",
            )

        # Find all expired payload refs
        expired = manager.find_expired_payload_refs(retention_days=30)

        # Should include all 4 payload refs
        assert "row_payload_ref" in expired
        assert "call_request_ref" in expired
        assert "call_response_ref" in expired
        assert "routing_reason_ref" in expired

        # Should be exactly 4 (no duplicates)
        assert len(expired) == 4

    def test_find_expired_payload_refs_respects_retention(self) -> None:
        """Recent call/routing payloads should not be found."""
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.schema import (
            calls_table,
            node_states_table,
            nodes_table,
            rows_table,
            runs_table,
            tokens_table,
        )
        from elspeth.core.retention.purge import PurgeManager

        db = LandscapeDB.in_memory()
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
            _create_run(conn, runs_table, run_id, completed_at=recent_completed_at, status="completed")
            _create_node(conn, nodes_table, node_id, run_id)
            _create_row(conn, rows_table, row_id, run_id, node_id, row_index=0)
            _create_token(conn, tokens_table, token_id, row_id)
            _create_state(conn, node_states_table, state_id, token_id, node_id)
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

    def test_shared_row_ref_excluded_when_used_by_recent_run(self) -> None:
        """Shared row payload ref should NOT be purged if a recent run uses it.

        Scenario:
        - Run A (60 days old, expired) has row with source_data_ref=H
        - Run B (10 days old, not expired) has row with same source_data_ref=H
        - Purge with 30-day retention should NOT return H (Run B needs it)
        """
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.schema import nodes_table, rows_table, runs_table
        from elspeth.core.retention.purge import PurgeManager

        db = LandscapeDB.in_memory()
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
            _create_run(conn, runs_table, run_a_id, completed_at=old_completed, status="completed")
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
            _create_run(conn, runs_table, run_b_id, completed_at=recent_completed, status="completed")
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

    def test_shared_call_ref_excluded_when_used_by_recent_run(self) -> None:
        """Shared call payload ref should NOT be purged if a recent run uses it."""
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.schema import (
            calls_table,
            node_states_table,
            nodes_table,
            rows_table,
            runs_table,
            tokens_table,
        )
        from elspeth.core.retention.purge import PurgeManager

        db = LandscapeDB.in_memory()
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
            _create_run(conn, runs_table, run_a_id, completed_at=old_completed, status="completed")
            _create_node(conn, nodes_table, node_a_id, run_a_id)
            _create_row(conn, rows_table, row_a_id, run_a_id, node_a_id, row_index=0)
            _create_token(conn, tokens_table, token_a_id, row_a_id)
            _create_state(conn, node_states_table, state_a_id, token_a_id, node_a_id)
            _create_call(conn, calls_table, str(uuid4()), state_a_id, response_ref=shared_ref)

            # Run B with call using SAME shared ref
            node_b_id = str(uuid4())
            row_b_id = str(uuid4())
            token_b_id = str(uuid4())
            state_b_id = str(uuid4())
            _create_run(conn, runs_table, run_b_id, completed_at=recent_completed, status="completed")
            _create_node(conn, nodes_table, node_b_id, run_b_id)
            _create_row(conn, rows_table, row_b_id, run_b_id, node_b_id, row_index=0)
            _create_token(conn, tokens_table, token_b_id, row_b_id)
            _create_state(conn, node_states_table, state_b_id, token_b_id, node_b_id)
            _create_call(conn, calls_table, str(uuid4()), state_b_id, response_ref=shared_ref)

        expired = manager.find_expired_payload_refs(retention_days=30)

        assert shared_ref not in expired, (
            f"Shared call ref {shared_ref} should NOT be returned for deletion because Run B (10 days old) still needs it"
        )

    def test_exclusive_expired_ref_is_returned(self) -> None:
        """Ref used ONLY by expired runs should be returned for deletion."""
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.schema import nodes_table, rows_table, runs_table
        from elspeth.core.retention.purge import PurgeManager

        db = LandscapeDB.in_memory()
        store = MockPayloadStore()
        manager = PurgeManager(db, store)

        # Ref used only by expired run
        exclusive_ref = "exclusive_to_expired_run"

        # Run A: 60 days old (expired)
        run_a_id = str(uuid4())
        node_a_id = str(uuid4())
        old_completed = datetime.now(UTC) - timedelta(days=60)

        with db.connection() as conn:
            _create_run(conn, runs_table, run_a_id, completed_at=old_completed, status="completed")
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

    def test_shared_ref_excluded_when_used_by_running_run(self) -> None:
        """Shared ref should NOT be purged if an incomplete (running) run uses it."""
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.schema import nodes_table, rows_table, runs_table
        from elspeth.core.retention.purge import PurgeManager

        db = LandscapeDB.in_memory()
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
            _create_run(conn, runs_table, run_a_id, completed_at=old_completed, status="completed")
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
            _create_run(conn, runs_table, run_b_id, completed_at=None, status="running")
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
