"""Tests for Landscape error recording and status enum coercion."""

from __future__ import annotations

from datetime import UTC
from typing import TYPE_CHECKING

from elspeth.contracts.audit import TokenRef
from elspeth.contracts.schema import SchemaConfig

if TYPE_CHECKING:
    from elspeth.core.landscape.data_flow_repository import DataFlowRepository
    from elspeth.core.landscape.database import LandscapeDB

# Dynamic schema for tests that don't care about specific fields
DYNAMIC_SCHEMA = SchemaConfig.from_dict({"mode": "observed"})


class TestTransformErrorRecording:
    """Tests for transform error recording in landscape."""

    @staticmethod
    def _create_token_with_dependencies(
        data_flow: DataFlowRepository, db: LandscapeDB, run_id: str, token_id: str, transform_id: str
    ) -> None:
        """Helper to create token with dependencies for FK constraints.

        Bug fix: P2-2026-01-19-error-tables-missing-foreign-keys
        FK constraints require token and node to exist before error recording.
        """
        from elspeth.contracts import NodeType
        from elspeth.contracts.schema import SchemaConfig

        # Create source node
        data_flow.register_node(
            run_id=run_id,
            plugin_name="test_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=SchemaConfig.from_dict({"mode": "observed"}),
            node_id="source_test",
            sequence=0,
        )
        # Create row
        row = data_flow.create_row(
            run_id=run_id,
            source_node_id="source_test",
            row_index=1,
            data={"id": "test"},
        )
        # Create token with specified ID
        from datetime import datetime

        from elspeth.core.landscape.schema import tokens_table

        with db.connection() as conn:
            conn.execute(
                tokens_table.insert().values(
                    token_id=token_id,
                    row_id=row.row_id,
                    run_id=run_id,
                    step_in_pipeline=0,
                    created_at=datetime.now(UTC),
                )
            )
            conn.commit()
        # Create transform node for transform_id FK
        data_flow.register_node(
            run_id=run_id,
            plugin_name="test_transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=SchemaConfig.from_dict({"mode": "observed"}),
            node_id=transform_id,
            sequence=1,
        )

    def test_record_transform_error_returns_error_id(self) -> None:
        """record_transform_error returns an error_id."""
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.factory import RecorderFactory

        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)

        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")

        # Create token and node for FK constraints
        self._create_token_with_dependencies(factory.data_flow, db, run.run_id, "tok_123", "field_mapper")

        error_id = factory.data_flow.record_transform_error(
            ref=TokenRef(token_id="tok_123", run_id=run.run_id),
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
        from elspeth.core.landscape.factory import RecorderFactory
        from elspeth.core.landscape.schema import transform_errors_table

        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)

        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")

        # Create token and node for FK constraints
        self._create_token_with_dependencies(factory.data_flow, db, run.run_id, "tok_456", "field_mapper")

        error_id = factory.data_flow.record_transform_error(
            ref=TokenRef(token_id="tok_456", run_id=run.run_id),
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
        from elspeth.core.landscape.factory import RecorderFactory
        from elspeth.core.landscape.schema import transform_errors_table

        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)

        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")

        # Create token and node for FK constraints
        self._create_token_with_dependencies(factory.data_flow, db, run.run_id, "tok_789", "processor")

        row_data = {"id": 42, "value": "bad"}
        expected_hash = stable_hash(row_data)

        error_id = factory.data_flow.record_transform_error(
            ref=TokenRef(token_id="tok_789", run_id=run.run_id),
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
        from elspeth.core.landscape.factory import RecorderFactory
        from elspeth.core.landscape.schema import transform_errors_table

        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)

        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")

        # Create token and node for FK constraints
        self._create_token_with_dependencies(factory.data_flow, db, run.run_id, "tok_999", "gate")

        error_id = factory.data_flow.record_transform_error(
            ref=TokenRef(token_id="tok_999", run_id=run.run_id),
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

    def test_record_transform_error_with_nan_row_data_uses_repr_fallback(self) -> None:
        """record_transform_error handles NaN in row_data without crashing.

        Regression: elspeth-80cd927867 — row_data serialization was unprotected,
        causing the error-recording path itself to crash when row_data contained
        NaN (a valid float that passes source validation). The actual transform
        error diagnostic was lost.
        """
        from sqlalchemy import select

        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.factory import RecorderFactory
        from elspeth.core.landscape.schema import transform_errors_table

        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)

        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")

        self._create_token_with_dependencies(factory.data_flow, db, run.run_id, "tok_nan", "processor")

        # Row with NaN — valid float, passes source validation, but
        # canonical_json rejects it (RFC 8785 forbids NaN/Infinity)
        row_data = {"id": 42, "score": float("nan"), "label": "test"}

        error_id = factory.data_flow.record_transform_error(
            ref=TokenRef(token_id="tok_nan", run_id=run.run_id),
            transform_id="processor",
            row_data=row_data,
            error_details={"reason": "float_overflow", "error": "Cannot divide by zero"},
            destination="error_sink",
        )

        assert error_id is not None
        assert error_id.startswith("terr_")

        # Verify stored in database with fallback representation
        with db.connection() as conn:
            result = conn.execute(select(transform_errors_table).where(transform_errors_table.c.error_id == error_id))
            row = result.fetchone()

        assert row is not None
        assert row.row_hash is not None  # repr_hash fallback
        assert row.row_data_json is not None
        assert "__repr__" in row.row_data_json  # NonCanonicalMetadata marker
        assert "__canonical_error__" in row.row_data_json
        assert row.error_details_json is not None
        assert "float_overflow" in row.error_details_json

    def test_record_transform_error_with_infinity_row_data_uses_repr_fallback(self) -> None:
        """record_transform_error handles Infinity in row_data without crashing."""
        from sqlalchemy import select

        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.factory import RecorderFactory
        from elspeth.core.landscape.schema import transform_errors_table

        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)

        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")

        self._create_token_with_dependencies(factory.data_flow, db, run.run_id, "tok_inf", "processor")

        row_data = {"id": 1, "value": float("inf")}

        error_id = factory.data_flow.record_transform_error(
            ref=TokenRef(token_id="tok_inf", run_id=run.run_id),
            transform_id="processor",
            row_data=row_data,
            error_details={"reason": "float_overflow", "error": "Value too large"},
            destination="discard",
        )

        assert error_id is not None

        with db.connection() as conn:
            result = conn.execute(select(transform_errors_table).where(transform_errors_table.c.error_id == error_id))
            row = result.fetchone()

        assert row is not None
        assert row.row_hash is not None
        assert "__repr__" in row.row_data_json


class TestExportStatusEnumCoercion:
    """Tests that export status is properly coerced to ExportStatus enum.

    Regression tests for:
    - docs/bugs/closed/P2-2026-01-19-recorder-export-status-enum-mismatch.md
    """

    def test_get_run_returns_export_status_enum(self) -> None:
        """get_run() returns ExportStatus enum, not raw string."""
        from elspeth.contracts import ExportStatus
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.factory import RecorderFactory

        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)

        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        factory.run_lifecycle.set_export_status(run.run_id, ExportStatus.COMPLETED)

        loaded = factory.run_lifecycle.get_run(run.run_id)

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
        from elspeth.core.landscape.factory import RecorderFactory

        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)

        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        factory.run_lifecycle.set_export_status(run.run_id, ExportStatus.PENDING)

        runs = factory.run_lifecycle.list_runs()

        assert len(runs) == 1
        assert isinstance(runs[0].export_status, ExportStatus), (
            f"export_status should be ExportStatus enum, got {type(runs[0].export_status).__name__}"
        )

    def test_set_export_status_clears_stale_error_on_completed(self) -> None:
        """Transitioning from failed to completed clears export_error."""
        from elspeth.contracts import ExportStatus
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.factory import RecorderFactory

        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)

        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")

        # First fail with an error
        factory.run_lifecycle.set_export_status(run.run_id, ExportStatus.FAILED, error="export failed")
        r1 = factory.run_lifecycle.get_run(run.run_id)
        assert r1 is not None
        assert r1.export_error == "export failed"

        # Now complete - error should be cleared
        factory.run_lifecycle.set_export_status(run.run_id, ExportStatus.COMPLETED)
        r2 = factory.run_lifecycle.get_run(run.run_id)
        assert r2 is not None
        assert r2.export_error is None, f"export_error should be cleared on completed, got {r2.export_error!r}"

    def test_set_export_status_clears_stale_error_on_pending(self) -> None:
        """Transitioning from failed to pending clears export_error."""
        from elspeth.contracts import ExportStatus
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.factory import RecorderFactory

        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)

        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")

        # First fail with an error
        factory.run_lifecycle.set_export_status(run.run_id, ExportStatus.FAILED, error="export failed")

        # Now set to pending - error should be cleared
        factory.run_lifecycle.set_export_status(run.run_id, ExportStatus.PENDING)
        r = factory.run_lifecycle.get_run(run.run_id)
        assert r is not None
        assert r.export_error is None

    def test_set_export_status_accepts_enum_directly(self) -> None:
        """set_export_status() accepts ExportStatus enum as well as string."""
        from elspeth.contracts import ExportStatus
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.factory import RecorderFactory

        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)

        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")

        # Pass enum directly
        factory.run_lifecycle.set_export_status(run.run_id, ExportStatus.COMPLETED)

        r = factory.run_lifecycle.get_run(run.run_id)
        assert r is not None
        assert r.export_status == ExportStatus.COMPLETED
