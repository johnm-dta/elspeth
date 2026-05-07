# tests/core/landscape/test_recorder_runs.py
"""Tests for RecorderFactory run lifecycle management and status validation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from elspeth.contracts.schema import SchemaConfig

if TYPE_CHECKING:
    pass

# Dynamic schema for tests that don't care about specific fields
DYNAMIC_SCHEMA = SchemaConfig.from_dict({"mode": "observed"})


class TestRecorderFactoryRuns:
    """Run lifecycle management."""

    def test_begin_run(self) -> None:
        from elspeth.contracts import RunStatus
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.factory import RecorderFactory

        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)

        run = factory.run_lifecycle.begin_run(
            config={"source": "test.csv"},
            canonical_version="sha256-rfc8785-v1",
        )

        assert run.run_id is not None
        assert run.status == RunStatus.RUNNING
        assert run.started_at is not None

    def test_complete_run_success(self) -> None:
        from elspeth.contracts import RunStatus
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.factory import RecorderFactory

        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)

        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        completed = factory.run_lifecycle.complete_run(run.run_id, status=RunStatus.COMPLETED)

        assert completed.status == RunStatus.COMPLETED
        assert completed.completed_at is not None

    def test_complete_run_failed(self) -> None:
        from elspeth.contracts import RunStatus
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.factory import RecorderFactory

        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)

        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        completed = factory.run_lifecycle.complete_run(run.run_id, status=RunStatus.FAILED)

        assert completed.status == RunStatus.FAILED

    def test_get_run(self) -> None:
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.factory import RecorderFactory

        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)

        run = factory.run_lifecycle.begin_run(config={"key": "value"}, canonical_version="v1")
        retrieved = factory.run_lifecycle.get_run(run.run_id)

        assert retrieved is not None
        assert retrieved.run_id == run.run_id


class TestRecorderFactoryRunStatusValidation:
    """Run status validation against RunStatus enum."""

    def test_begin_run_with_enum_status(self) -> None:
        """Test that RunStatus enum is accepted."""
        from elspeth.contracts import RunStatus
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.factory import RecorderFactory

        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)

        run = factory.run_lifecycle.begin_run(
            config={},
            canonical_version="v1",
            status=RunStatus.RUNNING,
        )

        assert run.status == RunStatus.RUNNING

    def test_begin_run_with_string_status_raises_typeerror(self) -> None:
        """Test that string status raises TypeError - enum required."""
        import pytest

        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.factory import RecorderFactory

        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)

        with pytest.raises(TypeError, match="status must be RunStatus, got str"):
            factory.run_lifecycle.begin_run(
                config={},
                canonical_version="v1",
                status="runnign",  # type: ignore[arg-type]  # Intentionally wrong type to test validation
            )

    def test_complete_run_with_enum_status(self) -> None:
        """Test that RunStatus enum is accepted for complete_run."""
        from elspeth.contracts import RunStatus
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.factory import RecorderFactory

        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)

        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        completed = factory.run_lifecycle.complete_run(run.run_id, status=RunStatus.COMPLETED)

        assert completed.status == RunStatus.COMPLETED

    def test_list_runs_with_enum_status_filter(self) -> None:
        """Test that RunStatus enum is accepted for list_runs filter."""
        from elspeth.contracts import RunStatus
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.factory import RecorderFactory

        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)

        # Create runs with different statuses
        run1 = factory.run_lifecycle.begin_run(config={"n": 1}, canonical_version="v1")
        run2 = factory.run_lifecycle.begin_run(config={"n": 2}, canonical_version="v1")
        factory.run_lifecycle.complete_run(run2.run_id, status=RunStatus.COMPLETED)

        # Filter by enum
        running_runs = factory.run_lifecycle.list_runs(status=RunStatus.RUNNING)
        assert len(running_runs) == 1
        assert running_runs[0].run_id == run1.run_id


class TestFieldResolutionTierOneIntegrity:
    """Tier-1 data integrity for field resolution storage.

    Per the Three-Tier Trust Model, Tier-1 (audit trail) data must be 100% pristine.
    If we detect corruption, we crash immediately - no silent recovery.

    The field resolution JSON always has this structure when recorded:
    {
        "resolution_mapping": {"Original Header": "normalized_header", ...},
        "normalization_version": "v1" | null
    }

    Corruption scenarios:
    - JSON present but missing resolution_mapping key -> crash
    - resolution_mapping is wrong type -> crash
    - resolution_mapping entry has wrong types -> crash

    Valid scenarios:
    - source_field_resolution_json is NULL in DB -> return None (never recorded)
    - resolution_mapping is an empty dict -> return {} (valid, no fields to map)
    """

    def test_get_field_resolution_returns_none_when_never_recorded(self) -> None:
        """NULL in DB means resolution was never recorded - return None."""
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.factory import RecorderFactory

        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)

        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        # Don't call record_source_field_resolution

        result = factory.run_lifecycle.get_source_field_resolution(run.run_id)
        assert result is None

    def test_get_field_resolution_returns_mapping_when_recorded(self) -> None:
        """Recorded mapping is returned correctly."""
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.factory import RecorderFactory

        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)

        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        mapping = {"User ID": "user_id", "Amount (USD)": "amount_usd"}
        factory.run_lifecycle.record_source_field_resolution(run.run_id, mapping, "v1")

        result = factory.run_lifecycle.get_source_field_resolution(run.run_id)
        assert result == mapping

    def test_get_field_resolution_returns_empty_dict_for_empty_mapping(self) -> None:
        """Empty mapping is valid (source had no fields to map)."""
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.factory import RecorderFactory

        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)

        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        factory.run_lifecycle.record_source_field_resolution(run.run_id, {}, None)

        result = factory.run_lifecycle.get_source_field_resolution(run.run_id)
        assert result == {}

    def test_get_field_resolution_crashes_on_missing_resolution_mapping_key(self) -> None:
        """Tier-1 corruption: JSON present but missing required key must crash.

        This tests the core Tier-1 principle: if our code always writes
        resolution_mapping, then finding JSON without it means corruption.
        We must crash, not silently return None.
        """
        import json

        import pytest
        from sqlalchemy import text

        from elspeth.contracts.errors import AuditIntegrityError
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.factory import RecorderFactory

        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)

        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")

        # Manually corrupt the database to simulate missing key
        corrupted_json = json.dumps({"normalization_version": "v1"})  # Missing resolution_mapping!
        with db.engine.begin() as conn:
            conn.execute(
                text("UPDATE runs SET source_field_resolution_json = :json WHERE run_id = :run_id"),
                {"json": corrupted_json, "run_id": run.run_id},
            )

        # Must crash on corruption - not return None
        with pytest.raises(AuditIntegrityError, match=r"missing required key.*resolution_mapping"):
            factory.run_lifecycle.get_source_field_resolution(run.run_id)

    def test_get_field_resolution_crashes_on_resolution_mapping_wrong_type(self) -> None:
        """Tier-1 corruption: resolution_mapping must be dict, not list/string/etc."""
        import json

        import pytest
        from sqlalchemy import text

        from elspeth.contracts.errors import AuditIntegrityError
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.factory import RecorderFactory

        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)

        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")

        # Manually corrupt: resolution_mapping is a list instead of dict
        corrupted_json = json.dumps({"resolution_mapping": ["a", "b"], "normalization_version": "v1"})
        with db.engine.begin() as conn:
            conn.execute(
                text("UPDATE runs SET source_field_resolution_json = :json WHERE run_id = :run_id"),
                {"json": corrupted_json, "run_id": run.run_id},
            )

        with pytest.raises(AuditIntegrityError, match=r"Corrupt resolution_mapping.*expected dict"):
            factory.run_lifecycle.get_source_field_resolution(run.run_id)

    def test_get_field_resolution_crashes_on_wrong_entry_types(self) -> None:
        """Tier-1 corruption: all mapping entries must be str->str."""
        import json

        import pytest
        from sqlalchemy import text

        from elspeth.contracts.errors import AuditIntegrityError
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.factory import RecorderFactory

        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)

        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")

        # Manually corrupt: value is int instead of str
        corrupted_json = json.dumps({"resolution_mapping": {"header": 123}, "normalization_version": "v1"})
        with db.engine.begin() as conn:
            conn.execute(
                text("UPDATE runs SET source_field_resolution_json = :json WHERE run_id = :run_id"),
                {"json": corrupted_json, "run_id": run.run_id},
            )

        with pytest.raises(AuditIntegrityError, match=r"Corrupt resolution_mapping entry.*str->str"):
            factory.run_lifecycle.get_source_field_resolution(run.run_id)
