"""Tests for LandscapeRecorder error recording and status enum coercion."""

from __future__ import annotations

from datetime import UTC
from typing import TYPE_CHECKING

from elspeth.contracts.schema import SchemaConfig

if TYPE_CHECKING:
    from elspeth.core.landscape.recorder import LandscapeRecorder

# Dynamic schema for tests that don't care about specific fields
DYNAMIC_SCHEMA = SchemaConfig.from_dict({"fields": "dynamic"})


class TestTransformErrorRecording:
    """Tests for transform error recording in landscape."""

    @staticmethod
    def _create_token_with_dependencies(recorder: LandscapeRecorder, run_id: str, token_id: str, transform_id: str) -> None:
        """Helper to create token with dependencies for FK constraints.

        Bug fix: P2-2026-01-19-error-tables-missing-foreign-keys
        FK constraints require token and node to exist before error recording.
        """
        from elspeth.contracts import NodeType
        from elspeth.contracts.schema import SchemaConfig

        # Create source node
        recorder.register_node(
            run_id=run_id,
            plugin_name="test_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=SchemaConfig.from_dict({"fields": "dynamic"}),
            node_id="source_test",
            sequence=0,
        )
        # Create row
        row = recorder.create_row(
            run_id=run_id,
            source_node_id="source_test",
            row_index=1,
            data={"id": "test"},
        )
        # Create token with specified ID
        from datetime import datetime

        from elspeth.core.landscape.schema import tokens_table

        with recorder._db.connection() as conn:
            conn.execute(
                tokens_table.insert().values(
                    token_id=token_id,
                    row_id=row.row_id,
                    step_in_pipeline=0,
                    created_at=datetime.now(UTC),
                )
            )
            conn.commit()
        # Create transform node for transform_id FK
        recorder.register_node(
            run_id=run_id,
            plugin_name="test_transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=SchemaConfig.from_dict({"fields": "dynamic"}),
            node_id=transform_id,
            sequence=1,
        )

    def test_record_transform_error_returns_error_id(self) -> None:
        """record_transform_error returns an error_id."""
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={}, canonical_version="v1")

        # Create token and node for FK constraints
        self._create_token_with_dependencies(recorder, run.run_id, "tok_123", "field_mapper")

        error_id = recorder.record_transform_error(
            run_id=run.run_id,
            token_id="tok_123",
            transform_id="field_mapper",
            row_data={"id": 42, "value": "bad"},
            error_details={"reason": "validation_failed", "error": "Division by zero"},
            destination="failed_rows",
        )

        assert error_id is not None
        assert error_id.startswith("terr_")

    def test_record_transform_error_stores_in_database(self) -> None:
        """record_transform_error stores error in transform_errors table."""
        from sqlalchemy import select

        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder
        from elspeth.core.landscape.schema import transform_errors_table

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={}, canonical_version="v1")

        # Create token and node for FK constraints
        self._create_token_with_dependencies(recorder, run.run_id, "tok_456", "field_mapper")

        error_id = recorder.record_transform_error(
            run_id=run.run_id,
            token_id="tok_456",
            transform_id="field_mapper",
            row_data={"id": 42, "value": "bad"},
            error_details={"reason": "validation_failed", "error": "Division by zero", "field": "divisor"},
            destination="error_sink",
        )

        # Verify stored in database
        with db.connection() as conn:
            result = conn.execute(select(transform_errors_table).where(transform_errors_table.c.error_id == error_id))
            row = result.fetchone()

        assert row is not None
        assert row.run_id == run.run_id
        assert row.token_id == "tok_456"
        assert row.transform_id == "field_mapper"
        assert row.row_hash is not None
        assert row.row_data_json is not None
        assert row.error_details_json is not None
        assert row.destination == "error_sink"
        assert row.created_at is not None

    def test_record_transform_error_stores_row_hash(self) -> None:
        """record_transform_error computes and stores row hash."""
        from sqlalchemy import select

        from elspeth.core.canonical import stable_hash
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder
        from elspeth.core.landscape.schema import transform_errors_table

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={}, canonical_version="v1")

        # Create token and node for FK constraints
        self._create_token_with_dependencies(recorder, run.run_id, "tok_789", "processor")

        row_data = {"id": 42, "value": "bad"}
        expected_hash = stable_hash(row_data)

        error_id = recorder.record_transform_error(
            run_id=run.run_id,
            token_id="tok_789",
            transform_id="processor",
            row_data=row_data,
            error_details={"reason": "validation_failed", "error": "Processing failed"},
            destination="discard",
        )

        with db.connection() as conn:
            result = conn.execute(select(transform_errors_table).where(transform_errors_table.c.error_id == error_id))
            row = result.fetchone()

        assert row is not None
        assert row.row_hash == expected_hash

    def test_record_transform_error_discard_destination(self) -> None:
        """record_transform_error handles 'discard' destination."""
        from sqlalchemy import select

        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder
        from elspeth.core.landscape.schema import transform_errors_table

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={}, canonical_version="v1")

        # Create token and node for FK constraints
        self._create_token_with_dependencies(recorder, run.run_id, "tok_999", "gate")

        error_id = recorder.record_transform_error(
            run_id=run.run_id,
            token_id="tok_999",
            transform_id="gate",
            row_data={"id": 1},
            error_details={"reason": "validation_failed", "error": "Gate evaluation failed"},
            destination="discard",
        )

        with db.connection() as conn:
            result = conn.execute(select(transform_errors_table).where(transform_errors_table.c.error_id == error_id))
            row = result.fetchone()

        assert row is not None
        assert row.destination == "discard"


class TestExportStatusEnumCoercion:
    """Tests that export status is properly coerced to ExportStatus enum.

    Regression tests for:
    - docs/bugs/closed/P2-2026-01-19-recorder-export-status-enum-mismatch.md
    """

    def test_get_run_returns_export_status_enum(self) -> None:
        """get_run() returns ExportStatus enum, not raw string."""
        from elspeth.contracts import ExportStatus
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={}, canonical_version="v1")
        recorder.set_export_status(run.run_id, ExportStatus.COMPLETED)

        loaded = recorder.get_run(run.run_id)

        assert loaded is not None
        assert loaded.export_status is not None
        assert isinstance(loaded.export_status, ExportStatus), (
            f"export_status should be ExportStatus enum, got {type(loaded.export_status).__name__}"
        )
        assert loaded.export_status == ExportStatus.COMPLETED

    def test_list_runs_returns_export_status_enum(self) -> None:
        """list_runs() returns ExportStatus enum, not raw string."""
        from elspeth.contracts import ExportStatus
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={}, canonical_version="v1")
        recorder.set_export_status(run.run_id, ExportStatus.PENDING)

        runs = recorder.list_runs()

        assert len(runs) == 1
        assert isinstance(runs[0].export_status, ExportStatus), (
            f"export_status should be ExportStatus enum, got {type(runs[0].export_status).__name__}"
        )

    def test_set_export_status_rejects_non_enum(self) -> None:
        """set_export_status() requires ExportStatus enum, not string.

        Strings passed where ExportStatus is expected will raise AttributeError
        because strings don't have .value attribute. This is the correct behavior
        per the strict enum enforcement policy.
        """
        import pytest

        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={}, canonical_version="v1")

        # Passing string instead of ExportStatus enum raises AttributeError
        with pytest.raises(AttributeError, match="'str' object has no attribute 'value'"):
            recorder.set_export_status(run.run_id, "invalid_status")  # type: ignore[arg-type]

    def test_set_export_status_clears_stale_error_on_completed(self) -> None:
        """Transitioning from failed to completed clears export_error."""
        from elspeth.contracts import ExportStatus
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={}, canonical_version="v1")

        # First fail with an error
        recorder.set_export_status(run.run_id, ExportStatus.FAILED, error="export failed")
        r1 = recorder.get_run(run.run_id)
        assert r1 is not None
        assert r1.export_error == "export failed"

        # Now complete - error should be cleared
        recorder.set_export_status(run.run_id, ExportStatus.COMPLETED)
        r2 = recorder.get_run(run.run_id)
        assert r2 is not None
        assert r2.export_error is None, f"export_error should be cleared on completed, got {r2.export_error!r}"

    def test_set_export_status_clears_stale_error_on_pending(self) -> None:
        """Transitioning from failed to pending clears export_error."""
        from elspeth.contracts import ExportStatus
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={}, canonical_version="v1")

        # First fail with an error
        recorder.set_export_status(run.run_id, ExportStatus.FAILED, error="export failed")

        # Now set to pending - error should be cleared
        recorder.set_export_status(run.run_id, ExportStatus.PENDING)
        r = recorder.get_run(run.run_id)
        assert r is not None
        assert r.export_error is None

    def test_set_export_status_accepts_enum_directly(self) -> None:
        """set_export_status() accepts ExportStatus enum as well as string."""
        from elspeth.contracts import ExportStatus
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={}, canonical_version="v1")

        # Pass enum directly
        recorder.set_export_status(run.run_id, ExportStatus.COMPLETED)

        r = recorder.get_run(run.run_id)
        assert r is not None
        assert r.export_status == ExportStatus.COMPLETED
