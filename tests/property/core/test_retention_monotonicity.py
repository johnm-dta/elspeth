# tests/property/core/test_retention_monotonicity.py
"""Property-based tests for retention age monotonicity.

These tests verify that ELSPETH's retention/purge system maintains
monotonicity invariants:

Age Monotonicity Properties:
- Increasing retention_days produces equal or fewer expired refs
- Zero retention days produces a superset of any positive retention days
- Sufficiently large retention days produces empty expired list

Cutoff Monotonicity Properties:
- Moving the cutoff earlier (as_of further in the past) reduces expired refs
- Moving the cutoff later (as_of further in the future) increases expired refs

PurgeResult Invariants:
- deleted_count + skipped_count + len(failed_refs) == total refs processed
- deleted_count is non-negative
- skipped_count is non-negative
- duration_seconds is non-negative

These properties ensure the retention system behaves predictably:
longer retention periods always preserve more data, and the purge
operation correctly accounts for all refs.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from itertools import count
from unittest.mock import MagicMock

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from elspeth.contracts.enums import Determinism, NodeType, RunStatus
from elspeth.core.canonical import stable_hash
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.reproducibility import ReproducibilityGrade
from elspeth.core.landscape.schema import nodes_table, rows_table, runs_table
from elspeth.core.retention import PurgeManager, PurgeResult
from tests.fixtures.landscape import make_landscape_db

# =============================================================================
# Strategies for retention testing
# =============================================================================

# Retention days (positive)
retention_days_strategy = st.integers(min_value=1, max_value=365)

# Number of rows to create
row_counts = st.integers(min_value=1, max_value=10)


# =============================================================================
# Helpers for DB-backed tests
# =============================================================================


SOURCE_NODE_ID = "source_node"
_REFERENCE_TIME = datetime(2025, 1, 1, tzinfo=UTC)
_RUN_COUNTER = count()
_ROW_COUNTER = count()
_REF_COUNTER = count()


def _create_completed_run(
    db: LandscapeDB,
    completed_at: datetime,
) -> str:
    """Create a completed run with a source node in the database."""
    run_id = f"run-{next(_RUN_COUNTER):06d}"
    now = _REFERENCE_TIME
    config = {"node_id": SOURCE_NODE_ID}
    with db.connection() as conn:
        conn.execute(
            runs_table.insert().values(
                run_id=run_id,
                started_at=now,
                completed_at=completed_at,
                config_hash=stable_hash({"run_id": run_id}),
                settings_json="{}",
                canonical_version="sha256-rfc8785-v1",
                status=RunStatus.COMPLETED,
            )
        )
        # Create a source node (required FK for rows.source_node_id)
        conn.execute(
            nodes_table.insert().values(
                node_id=SOURCE_NODE_ID,
                run_id=run_id,
                plugin_name="test_source",
                node_type=NodeType.SOURCE,
                plugin_version="1.0",
                determinism=Determinism.DETERMINISTIC,
                config_hash=stable_hash(config),
                config_json="{}",
                registered_at=now,
            )
        )
    # Set reproducibility grade (required by purge_payloads -> update_grade_after_purge)
    with db.connection() as conn:
        conn.execute(
            runs_table.update()
            .where(runs_table.c.run_id == run_id)
            .values(
                reproducibility_grade=ReproducibilityGrade.FULL_REPRODUCIBLE.value,
            )
        )
    return run_id


def _insert_rows_with_refs(
    db: LandscapeDB,
    run_id: str,
    num_rows: int,
) -> list[str]:
    """Insert rows with source_data_ref into the database. Returns refs."""
    refs = []
    with db.connection() as conn:
        for i in range(num_rows):
            row_id = f"row-{next(_ROW_COUNTER):06d}"
            ref = f"sha256-{next(_REF_COUNTER):064x}"
            refs.append(ref)
            conn.execute(
                rows_table.insert().values(
                    row_id=row_id,
                    run_id=run_id,
                    source_node_id=SOURCE_NODE_ID,
                    row_index=i,
                    source_data_ref=ref,
                    source_data_hash=stable_hash({"row": i}),
                    created_at=_REFERENCE_TIME,
                )
            )
    return refs


# =============================================================================
# Age Monotonicity Property Tests
# =============================================================================


class TestRetentionAgeMonotonicityProperties:
    """Property tests for retention age monotonicity.

    Core property: increasing retention_days should produce
    equal or fewer expired refs. More retention = less purging.
    """

    @given(
        days_short=retention_days_strategy,
        days_long=retention_days_strategy,
        num_rows=row_counts,
    )
    @settings(max_examples=50)
    def test_longer_retention_fewer_expired(
        self,
        days_short: int,
        days_long: int,
        num_rows: int,
    ) -> None:
        """Property: Longer retention produces equal or fewer expired refs.

        If we keep data for 30 days instead of 7, we should never
        expire MORE refs - only the same or fewer.
        """
        assume(days_short < days_long)

        with make_landscape_db() as db:
            # Create a run that completed 20 days ago
            completed_at = _REFERENCE_TIME - timedelta(days=20)
            run_id = _create_completed_run(db, completed_at)
            _insert_rows_with_refs(db, run_id, num_rows)

            mock_store = MagicMock()
            manager = PurgeManager(db, mock_store)

            refs_short = manager.find_expired_payload_refs(days_short)
            refs_long = manager.find_expired_payload_refs(days_long)

            assert len(refs_long) <= len(refs_short), (
                f"Longer retention ({days_long}d) expired MORE refs ({len(refs_long)}) "
                f"than shorter retention ({days_short}d) ({len(refs_short)})"
            )

    @given(num_rows=row_counts)
    @settings(max_examples=20)
    def test_very_large_retention_no_expiry(self, num_rows: int) -> None:
        """Property: Sufficiently large retention days produces no expired refs.

        If we retain for 9999 days, a run completed yesterday should
        never have expired refs.
        """
        with make_landscape_db() as db:
            completed_at = _REFERENCE_TIME - timedelta(days=1)
            run_id = _create_completed_run(db, completed_at)
            _insert_rows_with_refs(db, run_id, num_rows)

            mock_store = MagicMock()
            manager = PurgeManager(db, mock_store)

            refs = manager.find_expired_payload_refs(retention_days=9999)

            assert len(refs) == 0, f"Expected 0 expired refs with 9999-day retention, got {len(refs)}"

    @given(num_rows=row_counts)
    @settings(max_examples=20)
    def test_zero_retention_expires_completed(self, num_rows: int) -> None:
        """Property: Zero-day retention expires all completed runs.

        With 0-day retention, any completed run should have its
        payloads eligible for purge.
        """
        with make_landscape_db() as db:
            # Run completed 1 second ago
            completed_at = _REFERENCE_TIME - timedelta(seconds=1)
            run_id = _create_completed_run(db, completed_at)
            refs = _insert_rows_with_refs(db, run_id, num_rows)

            mock_store = MagicMock()
            manager = PurgeManager(db, mock_store)

            expired = manager.find_expired_payload_refs(retention_days=0)

            assert set(expired) == set(refs), f"Zero-day retention should expire all {len(refs)} refs, but expired {len(expired)}"


# =============================================================================
# Cutoff Monotonicity Property Tests
# =============================================================================


class TestCutoffMonotonicityProperties:
    """Property tests for cutoff time monotonicity.

    Core property: moving as_of forward in time should produce
    equal or more expired refs.
    """

    @given(
        days_offset_early=st.integers(min_value=1, max_value=30),
        days_offset_late=st.integers(min_value=31, max_value=60),
        num_rows=row_counts,
    )
    @settings(max_examples=30)
    def test_later_cutoff_more_expired(
        self,
        days_offset_early: int,
        days_offset_late: int,
        num_rows: int,
    ) -> None:
        """Property: Later as_of produces equal or more expired refs.

        If we check "what expired as of 30 days from now" vs "as of 60
        days from now", the later check should find at least as many.
        """
        with make_landscape_db() as db:
            # Create a run completed 15 days ago
            completed_at = _REFERENCE_TIME - timedelta(days=15)
            run_id = _create_completed_run(db, completed_at)
            _insert_rows_with_refs(db, run_id, num_rows)

            mock_store = MagicMock()
            manager = PurgeManager(db, mock_store)

            now = _REFERENCE_TIME
            as_of_early = now + timedelta(days=days_offset_early)
            as_of_late = now + timedelta(days=days_offset_late)

            retention = 10  # 10-day retention

            refs_early = manager.find_expired_payload_refs(retention, as_of=as_of_early)
            refs_late = manager.find_expired_payload_refs(retention, as_of=as_of_late)

            assert len(refs_late) >= len(refs_early), (
                f"Later as_of ({as_of_late}) found fewer refs ({len(refs_late)}) than earlier as_of ({as_of_early}) ({len(refs_early)})"
            )


# =============================================================================
# PurgeResult Invariant Property Tests
# =============================================================================


class TestPurgeResultInvariantProperties:
    """Property tests for PurgeResult accounting invariants."""

    @given(
        deleted=st.integers(min_value=0, max_value=100),
        skipped=st.integers(min_value=0, max_value=100),
        num_failed=st.integers(min_value=0, max_value=10),
        duration=st.floats(min_value=0.0, max_value=1000.0, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_purge_result_fields_non_negative(
        self,
        deleted: int,
        skipped: int,
        num_failed: int,
        duration: float,
    ) -> None:
        """Property: PurgeResult fields are non-negative."""
        failed_refs = tuple(f"ref_{i}" for i in range(num_failed))

        result = PurgeResult(
            deleted_count=deleted,
            skipped_count=skipped,
            failed_refs=failed_refs,
            grade_update_failures=(),
            duration_seconds=duration,
        )

        assert result.deleted_count >= 0
        assert result.skipped_count >= 0
        assert result.duration_seconds >= 0
        assert len(result.failed_refs) >= 0

    @given(num_rows=row_counts)
    @settings(max_examples=20)
    def test_purge_accounting_complete(self, num_rows: int) -> None:
        """Property: Purge accounts for every ref (deleted + skipped + failed = total).

        No refs should be silently dropped during purging.
        """
        # PayloadStore.delete(False) means the ref was not found (skipped).
        mock_store = MagicMock()
        mock_store.delete.return_value = False

        with make_landscape_db() as db:
            completed_at = _REFERENCE_TIME - timedelta(days=30)
            run_id = _create_completed_run(db, completed_at)
            refs = _insert_rows_with_refs(db, run_id, num_rows)

            manager = PurgeManager(db, mock_store)
            result = manager.purge_payloads(refs)

            total_accounted = result.deleted_count + result.skipped_count + len(result.failed_refs)
            assert total_accounted == len(refs), (
                f"Purge lost track of refs: {total_accounted} accounted vs {len(refs)} total "
                f"(deleted={result.deleted_count}, skipped={result.skipped_count}, "
                f"failed={len(result.failed_refs)})"
            )

    @given(num_rows=row_counts)
    @settings(max_examples=20)
    def test_purge_nonexistent_all_skipped(self, num_rows: int) -> None:
        """Property: Purging refs that don't exist in store counts as skipped."""
        mock_store = MagicMock()
        mock_store.delete.return_value = False

        with make_landscape_db() as db:
            completed_at = _REFERENCE_TIME - timedelta(days=30)
            run_id = _create_completed_run(db, completed_at)
            refs = _insert_rows_with_refs(db, run_id, num_rows)

            manager = PurgeManager(db, mock_store)
            result = manager.purge_payloads(refs)

            assert result.skipped_count == len(refs), (
                f"Expected all {len(refs)} refs to be skipped (non-existent), got {result.skipped_count} skipped"
            )
            assert result.deleted_count == 0
            assert len(result.failed_refs) == 0

    @given(num_rows=row_counts)
    @settings(max_examples=20)
    def test_purge_existing_all_deleted(self, num_rows: int) -> None:
        """Property: Purging refs that exist and succeed counts as deleted."""
        mock_store = MagicMock()
        mock_store.delete.return_value = True

        with make_landscape_db() as db:
            completed_at = _REFERENCE_TIME - timedelta(days=30)
            run_id = _create_completed_run(db, completed_at)
            refs = _insert_rows_with_refs(db, run_id, num_rows)

            manager = PurgeManager(db, mock_store)
            result = manager.purge_payloads(refs)

            assert result.deleted_count == len(refs), f"Expected all {len(refs)} refs to be deleted, got {result.deleted_count} deleted"
            assert result.skipped_count == 0
            assert len(result.failed_refs) == 0
