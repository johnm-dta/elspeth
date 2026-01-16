# tests/core/retention/test_purge.py
"""Tests for PurgeManager - PayloadStore retention management."""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

from sqlalchemy import Connection, Table


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
            _create_run(
                conn, runs_table, run_id, completed_at=completed_at, status="completed"
            )
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

            result = conn.execute(
                select(rows_table.c.source_data_hash).where(
                    rows_table.c.row_id == row_id
                )
            )
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
