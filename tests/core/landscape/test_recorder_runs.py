# tests/core/landscape/test_recorder_runs.py
"""Tests for LandscapeRecorder run lifecycle management and status validation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from elspeth.contracts.schema import SchemaConfig

if TYPE_CHECKING:
    pass

# Dynamic schema for tests that don't care about specific fields
DYNAMIC_SCHEMA = SchemaConfig.from_dict({"fields": "dynamic"})


class TestLandscapeRecorderRuns:
    """Run lifecycle management."""

    def test_begin_run(self) -> None:
        from elspeth.contracts import RunStatus
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(
            config={"source": "test.csv"},
            canonical_version="sha256-rfc8785-v1",
        )

        assert run.run_id is not None
        assert run.status == RunStatus.RUNNING
        assert run.started_at is not None

    def test_complete_run_success(self) -> None:
        from elspeth.contracts import RunStatus
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={}, canonical_version="v1")
        completed = recorder.complete_run(run.run_id, status=RunStatus.COMPLETED)

        assert completed.status == RunStatus.COMPLETED
        assert completed.completed_at is not None

    def test_complete_run_failed(self) -> None:
        from elspeth.contracts import RunStatus
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={}, canonical_version="v1")
        completed = recorder.complete_run(run.run_id, status=RunStatus.FAILED)

        assert completed.status == RunStatus.FAILED

    def test_get_run(self) -> None:
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={"key": "value"}, canonical_version="v1")
        retrieved = recorder.get_run(run.run_id)

        assert retrieved is not None
        assert retrieved.run_id == run.run_id


class TestLandscapeRecorderRunStatusValidation:
    """Run status validation against RunStatus enum."""

    def test_begin_run_with_enum_status(self) -> None:
        """Test that RunStatus enum is accepted."""
        from elspeth.contracts import RunStatus
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(
            config={},
            canonical_version="v1",
            status=RunStatus.RUNNING,
        )

        assert run.status == RunStatus.RUNNING

    def test_begin_run_with_string_status_raises_typeerror(self) -> None:
        """Test that string status raises TypeError - enum required."""
        import pytest

        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        with pytest.raises(TypeError, match="status must be RunStatus, got str"):
            recorder.begin_run(
                config={},
                canonical_version="v1",
                status="runnign",  # String not accepted - enum required
            )

    def test_complete_run_with_enum_status(self) -> None:
        """Test that RunStatus enum is accepted for complete_run."""
        from elspeth.contracts import RunStatus
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={}, canonical_version="v1")
        completed = recorder.complete_run(run.run_id, status=RunStatus.COMPLETED)

        assert completed.status == RunStatus.COMPLETED

    def test_list_runs_with_enum_status_filter(self) -> None:
        """Test that RunStatus enum is accepted for list_runs filter."""
        from elspeth.contracts import RunStatus
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        # Create runs with different statuses
        run1 = recorder.begin_run(config={"n": 1}, canonical_version="v1")
        run2 = recorder.begin_run(config={"n": 2}, canonical_version="v1")
        recorder.complete_run(run2.run_id, status=RunStatus.COMPLETED)

        # Filter by enum
        running_runs = recorder.list_runs(status=RunStatus.RUNNING)
        assert len(running_runs) == 1
        assert running_runs[0].run_id == run1.run_id
