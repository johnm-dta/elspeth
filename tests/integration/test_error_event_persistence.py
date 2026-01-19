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
from elspeth.core.landscape.schema import validation_errors_table
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
