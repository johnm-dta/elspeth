# tests/integration/test_error_event_persistence.py
"""Integration tests for error event persistence in landscape.

Verifies that validation errors recorded through PluginContext are properly
persisted to the landscape database and queryable. This confirms the
SDA-029 implementation for validation error audit trail.
"""

from typing import Any

import pytest
from sqlalchemy import select

from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.recorder import LandscapeRecorder
from elspeth.core.landscape.schema import (
    transform_errors_table,
    validation_errors_table,
)
from elspeth.plugins.context import PluginContext


class TestValidationErrorPersistence:
    """Verify validation errors are persisted to landscape database."""

    @pytest.fixture
    def test_env(self) -> dict[str, Any]:
        """Set up test environment with in-memory database."""
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        return {
            "db": db,
            "recorder": recorder,
        }

    def test_validation_error_persisted_to_database(
        self, test_env: dict[str, Any]
    ) -> None:
        """Validation error from source should be queryable in database."""
        db: LandscapeDB = test_env["db"]
        recorder: LandscapeRecorder = test_env["recorder"]

        # Arrange: Create a run
        run = recorder.begin_run(
            config={"test": True},
            canonical_version="1.0",
        )
        run_id = run.run_id

        # Create context with landscape
        ctx = PluginContext(
            run_id=run_id,
            config={},
            landscape=recorder,
            node_id="source_node",
        )

        # Act: Record a validation error
        error_token = ctx.record_validation_error(
            row={"id": "row-1", "bad_field": "not_an_int"},
            error="Field 'bad_field' expected int, got str",
            schema_mode="strict",
            destination="quarantine_sink",
        )

        # Assert: Error is in database
        with db.connection() as conn:
            result = conn.execute(
                select(validation_errors_table).where(
                    validation_errors_table.c.error_id == error_token.error_id
                )
            ).fetchone()

        assert result is not None
        assert result.run_id == run_id
        assert result.node_id == "source_node"
        assert "bad_field" in result.error
        assert result.schema_mode == "strict"
        assert result.destination == "quarantine_sink"

    def test_validation_error_with_discard_still_recorded(
        self, test_env: dict[str, Any]
    ) -> None:
        """Even 'discard' destination records error for audit completeness."""
        db: LandscapeDB = test_env["db"]
        recorder: LandscapeRecorder = test_env["recorder"]

        # Arrange: Create a run
        run = recorder.begin_run(
            config={},
            canonical_version="1.0",
        )
        run_id = run.run_id

        ctx = PluginContext(
            run_id=run_id,
            config={},
            landscape=recorder,
            node_id="source_node",
        )

        # Act: Record with discard destination
        error_token = ctx.record_validation_error(
            row={"id": "discarded-row"},
            error="Missing required field",
            schema_mode="strict",
            destination="discard",
        )

        # Assert: Still recorded (audit completeness)
        with db.connection() as conn:
            result = conn.execute(
                select(validation_errors_table).where(
                    validation_errors_table.c.error_id == error_token.error_id
                )
            ).fetchone()

        assert result is not None
        assert result.destination == "discard"


class TestTransformErrorPersistence:
    """Verify transform errors are persisted to landscape database."""

    @pytest.fixture
    def test_env(self) -> dict[str, Any]:
        """Set up test environment with in-memory database."""
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        return {
            "db": db,
            "recorder": recorder,
        }

    def test_transform_error_persisted_to_database(
        self, test_env: dict[str, Any]
    ) -> None:
        """Transform error should be queryable in database."""
        db: LandscapeDB = test_env["db"]
        recorder: LandscapeRecorder = test_env["recorder"]

        # Arrange: Create a run
        run = recorder.begin_run(
            config={"test": True},
            canonical_version="1.0",
        )
        run_id = run.run_id

        # Create context with landscape
        ctx = PluginContext(
            run_id=run_id,
            config={},
            landscape=recorder,
            node_id="transform_node",
        )

        # Act: Record a transform error
        error_token = ctx.record_transform_error(
            token_id="token-123",
            transform_id="price_calculator",
            row={"quantity": 0, "total": 100},
            error_details={"reason": "division_by_zero", "field": "quantity"},
            destination="failed_calculations",
        )

        # Assert: Error is in database
        with db.connection() as conn:
            result = conn.execute(
                select(transform_errors_table).where(
                    transform_errors_table.c.error_id == error_token.error_id
                )
            ).fetchone()

        assert result is not None
        assert result.run_id == run_id
        assert result.token_id == "token-123"
        assert result.transform_id == "price_calculator"
        assert "division_by_zero" in result.error_details_json
        assert result.destination == "failed_calculations"

    def test_transform_error_with_discard_still_recorded(
        self, test_env: dict[str, Any]
    ) -> None:
        """Even 'discard' destination records TransformErrorEvent."""
        db: LandscapeDB = test_env["db"]
        recorder: LandscapeRecorder = test_env["recorder"]

        # Arrange: Create a run
        run = recorder.begin_run(
            config={},
            canonical_version="1.0",
        )
        run_id = run.run_id

        ctx = PluginContext(
            run_id=run_id,
            config={},
            landscape=recorder,
            node_id="transform_node",
        )

        # Act: Record with discard destination
        error_token = ctx.record_transform_error(
            token_id="token-456",
            transform_id="validator",
            row={"data": "invalid"},
            error_details={"reason": "validation_failed"},
            destination="discard",
        )

        # Assert: Still recorded (audit completeness)
        with db.connection() as conn:
            result = conn.execute(
                select(transform_errors_table).where(
                    transform_errors_table.c.error_id == error_token.error_id
                )
            ).fetchone()

        assert result is not None
        assert result.destination == "discard"
