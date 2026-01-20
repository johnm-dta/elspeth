# tests/integration/test_error_event_persistence.py
"""Integration tests for error event persistence in landscape.

Verifies that validation errors recorded through PluginContext are properly
persisted to the landscape database and queryable. This confirms the
SDA-029 implementation for validation error audit trail.
"""

from sqlalchemy import select

from elspeth.contracts.schema import SchemaConfig
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.lineage import explain
from elspeth.core.landscape.recorder import LandscapeRecorder
from elspeth.core.landscape.schema import (
    transform_errors_table,
    validation_errors_table,
)
from elspeth.plugins.context import PluginContext

# Shared schema config for tests
DYNAMIC_SCHEMA = SchemaConfig.from_dict({"fields": "dynamic"})


class TestValidationErrorPersistence:
    """Verify validation errors are persisted to landscape database."""

    def test_validation_error_persisted_to_database(self, landscape_db: LandscapeDB, recorder: LandscapeRecorder) -> None:
        """Validation error from source should be queryable in database."""
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
        with landscape_db.connection() as conn:
            result = conn.execute(
                select(validation_errors_table).where(validation_errors_table.c.error_id == error_token.error_id)
            ).fetchone()

        assert result is not None
        assert result.run_id == run_id
        assert result.node_id == "source_node"
        assert "bad_field" in result.error
        assert result.schema_mode == "strict"
        assert result.destination == "quarantine_sink"

    def test_validation_error_with_discard_still_recorded(self, landscape_db: LandscapeDB, recorder: LandscapeRecorder) -> None:
        """Even 'discard' destination records error for audit completeness."""
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
        with landscape_db.connection() as conn:
            result = conn.execute(
                select(validation_errors_table).where(validation_errors_table.c.error_id == error_token.error_id)
            ).fetchone()

        assert result is not None
        assert result.destination == "discard"


class TestTransformErrorPersistence:
    """Verify transform errors are persisted to landscape database."""

    def test_transform_error_persisted_to_database(self, landscape_db: LandscapeDB, recorder: LandscapeRecorder) -> None:
        """Transform error should be queryable in database."""
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
        with landscape_db.connection() as conn:
            result = conn.execute(
                select(transform_errors_table).where(transform_errors_table.c.error_id == error_token.error_id)
            ).fetchone()

        assert result is not None
        assert result.run_id == run_id
        assert result.token_id == "token-123"
        assert result.transform_id == "price_calculator"
        assert "division_by_zero" in result.error_details_json
        assert result.destination == "failed_calculations"

    def test_transform_error_with_discard_still_recorded(self, landscape_db: LandscapeDB, recorder: LandscapeRecorder) -> None:
        """Even 'discard' destination records TransformErrorEvent."""
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
        with landscape_db.connection() as conn:
            result = conn.execute(
                select(transform_errors_table).where(transform_errors_table.c.error_id == error_token.error_id)
            ).fetchone()

        assert result is not None
        assert result.destination == "discard"


class TestErrorEventExplainQuery:
    """Verify explain() includes error events in lineage."""

    def test_explain_includes_validation_errors(self, recorder: LandscapeRecorder) -> None:
        """explain() should return validation errors for queried row."""
        # Arrange: Create run and source node
        run = recorder.begin_run(
            config={"test": True},
            canonical_version="1.0",
        )
        run_id = run.run_id

        source_node = recorder.register_node(
            run_id=run_id,
            plugin_name="csv_source",
            node_type="source",
            plugin_version="1.0.0",
            config={"path": "data.csv"},
            sequence=0,
            schema_config=DYNAMIC_SCHEMA,
        )

        # Create a row that will have the same hash as the error
        row_data = {"id": "row-42", "value": "not_a_number"}
        row = recorder.create_row(
            run_id=run_id,
            source_node_id=source_node.node_id,
            row_index=0,
            data=row_data,
        )

        token = recorder.create_token(row_id=row.row_id)

        # Record validation error using same data (for matching hash)
        ctx = PluginContext(
            run_id=run_id,
            config={},
            landscape=recorder,
            node_id=source_node.node_id,
        )

        error_token = ctx.record_validation_error(
            row=row_data,
            error="Expected int for 'value'",
            schema_mode="strict",
            destination="quarantine",
        )

        # Act: Query lineage for this row
        lineage = explain(
            recorder=recorder,
            run_id=run_id,
            token_id=token.token_id,
        )

        # Assert: Lineage includes validation error
        assert lineage is not None
        assert len(lineage.validation_errors) == 1
        assert lineage.validation_errors[0].error_id == error_token.error_id
        assert "Expected int" in lineage.validation_errors[0].error

    def test_explain_includes_transform_errors(self, recorder: LandscapeRecorder) -> None:
        """explain() should return transform errors for queried token."""
        # Arrange: Create run, source node, and transform node
        run = recorder.begin_run(
            config={"test": True},
            canonical_version="1.0",
        )
        run_id = run.run_id

        source_node = recorder.register_node(
            run_id=run_id,
            plugin_name="csv_source",
            node_type="source",
            plugin_version="1.0.0",
            config={},
            sequence=0,
            schema_config=DYNAMIC_SCHEMA,
        )

        transform_node = recorder.register_node(
            run_id=run_id,
            plugin_name="divide_transform",
            node_type="transform",
            plugin_version="1.0.0",
            config={},
            sequence=1,
            schema_config=DYNAMIC_SCHEMA,
        )

        # Create row and token
        row_data = {"id": "row-99", "divisor": 0}
        row = recorder.create_row(
            run_id=run_id,
            source_node_id=source_node.node_id,
            row_index=0,
            data=row_data,
        )
        token = recorder.create_token(row_id=row.row_id)

        # Record transform error
        ctx = PluginContext(
            run_id=run_id,
            config={},
            landscape=recorder,
            node_id=transform_node.node_id,
        )

        error_token = ctx.record_transform_error(
            token_id=token.token_id,
            transform_id=transform_node.node_id,
            row=row_data,
            error_details={"reason": "division_by_zero"},
            destination="error_sink",
        )

        # Act: Query lineage
        lineage = explain(
            recorder=recorder,
            run_id=run_id,
            token_id=token.token_id,
        )

        # Assert: Lineage includes transform error
        assert lineage is not None
        assert len(lineage.transform_errors) == 1
        assert lineage.transform_errors[0].error_id == error_token.error_id
        assert lineage.transform_errors[0].token_id == token.token_id

    def test_explain_returns_empty_lists_when_no_errors(self, recorder: LandscapeRecorder) -> None:
        """explain() should return empty error lists for clean rows."""
        # Arrange: Create run, node, row, token (no errors)
        run = recorder.begin_run(
            config={},
            canonical_version="1.0",
        )
        run_id = run.run_id

        source_node = recorder.register_node(
            run_id=run_id,
            plugin_name="csv_source",
            node_type="source",
            plugin_version="1.0.0",
            config={},
            sequence=0,
            schema_config=DYNAMIC_SCHEMA,
        )

        row = recorder.create_row(
            run_id=run_id,
            source_node_id=source_node.node_id,
            row_index=0,
            data={"id": "clean-row", "value": 42},
        )
        token = recorder.create_token(row_id=row.row_id)

        # Act: Query lineage
        lineage = explain(
            recorder=recorder,
            run_id=run_id,
            token_id=token.token_id,
        )

        # Assert: No errors
        assert lineage is not None
        assert lineage.validation_errors == []
        assert lineage.transform_errors == []

    def test_explain_multiple_errors_for_same_token(self, recorder: LandscapeRecorder) -> None:
        """explain() should return multiple transform errors for same token."""
        # Arrange: Create run with multiple transforms
        run = recorder.begin_run(
            config={},
            canonical_version="1.0",
        )
        run_id = run.run_id

        source_node = recorder.register_node(
            run_id=run_id,
            plugin_name="csv_source",
            node_type="source",
            plugin_version="1.0.0",
            config={},
            sequence=0,
            schema_config=DYNAMIC_SCHEMA,
        )

        transform1 = recorder.register_node(
            run_id=run_id,
            plugin_name="transform1",
            node_type="transform",
            plugin_version="1.0.0",
            config={},
            sequence=1,
            schema_config=DYNAMIC_SCHEMA,
        )

        transform2 = recorder.register_node(
            run_id=run_id,
            plugin_name="transform2",
            node_type="transform",
            plugin_version="1.0.0",
            config={},
            sequence=2,
            schema_config=DYNAMIC_SCHEMA,
        )

        row_data = {"id": "multi-error", "value": "bad"}
        row = recorder.create_row(
            run_id=run_id,
            source_node_id=source_node.node_id,
            row_index=0,
            data=row_data,
        )
        token = recorder.create_token(row_id=row.row_id)

        ctx = PluginContext(
            run_id=run_id,
            config={},
            landscape=recorder,
            node_id="test",
        )

        # Record two transform errors for same token
        ctx.record_transform_error(
            token_id=token.token_id,
            transform_id=transform1.node_id,
            row=row_data,
            error_details={"reason": "error_1"},
            destination="error_sink",
        )

        ctx.record_transform_error(
            token_id=token.token_id,
            transform_id=transform2.node_id,
            row=row_data,
            error_details={"reason": "error_2"},
            destination="error_sink",
        )

        # Act: Query lineage
        lineage = explain(
            recorder=recorder,
            run_id=run_id,
            token_id=token.token_id,
        )

        # Assert: Both errors returned
        assert lineage is not None
        assert len(lineage.transform_errors) == 2
