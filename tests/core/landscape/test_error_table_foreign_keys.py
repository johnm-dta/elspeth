# tests/core/landscape/test_error_table_foreign_keys.py
"""Test foreign key enforcement for error tables.

Verifies that validation_errors and transform_errors tables enforce
referential integrity through FK constraints, preventing orphan error
records that would violate Tier 1 audit integrity.

Bug: P2-2026-01-19-error-tables-missing-foreign-keys.md
"""

from datetime import UTC, datetime

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from elspeth.contracts import NodeType
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.recorder import LandscapeRecorder
from elspeth.core.landscape.schema import (
    nodes_table,
    tokens_table,
    transform_errors_table,
    validation_errors_table,
)

# Shared schema config for tests
DYNAMIC_SCHEMA = SchemaConfig.from_dict({"fields": "dynamic"})


class TestTransformErrorsForeignKeys:
    """Verify transform_errors FK constraints prevent orphan records."""

    def test_rejects_orphan_token_id(self, landscape_db: LandscapeDB, recorder: LandscapeRecorder) -> None:
        """FK constraint should reject transform_errors with missing token_id."""
        # Arrange: Create a run and node (but NOT a token)
        run = recorder.begin_run(
            config={"test": True},
            canonical_version="1.0",
        )
        recorder.register_node(
            run_id=run.run_id,
            plugin_name="test_transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
            node_id="node_test",
            sequence=0,
        )

        # Act & Assert: Try to insert error with non-existent token_id
        with (
            pytest.raises(IntegrityError, match=r"(FOREIGN KEY constraint failed|violates foreign key)"),
            landscape_db.connection() as conn,
        ):
            conn.execute(
                transform_errors_table.insert().values(
                    error_id="terr_orphan001",
                    run_id=run.run_id,
                    token_id="nonexistent_token",  # ORPHAN - no such token
                    transform_id="node_test",
                    row_hash="abc123",
                    destination="discard",
                    created_at=datetime.now(UTC),
                )
            )
            conn.commit()

    def test_rejects_orphan_transform_id(self, landscape_db: LandscapeDB, recorder: LandscapeRecorder) -> None:
        """FK constraint should reject transform_errors with missing transform_id (node_id)."""
        # Arrange: Create a run, source node, and token (but NOT the transform node)
        run = recorder.begin_run(
            config={"test": True},
            canonical_version="1.0",
        )
        # Need source node for row creation
        recorder.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
            node_id="source_test",
            sequence=0,
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id="source_test",
            row_index=1,
            data={"id": "test-1"},
        )
        token = recorder.create_token(
            row_id=row.row_id,
        )

        # Act & Assert: Try to insert error with non-existent transform_id (node_id)
        with (
            pytest.raises(IntegrityError, match=r"(FOREIGN KEY constraint failed|violates foreign key)"),
            landscape_db.connection() as conn,
        ):
            conn.execute(
                transform_errors_table.insert().values(
                    error_id="terr_orphan002",
                    run_id=run.run_id,
                    token_id=token.token_id,
                    transform_id="nonexistent_node",  # ORPHAN - no such node
                    row_hash="abc123",
                    destination="discard",
                    created_at=datetime.now(UTC),
                )
            )
            conn.commit()

    def test_restrict_prevents_token_deletion(self, landscape_db: LandscapeDB, recorder: LandscapeRecorder) -> None:
        """Cannot delete token if transform_errors references it (RESTRICT)."""
        # Arrange: Create run, source node, token, transform node, and error record
        run = recorder.begin_run(
            config={"test": True},
            canonical_version="1.0",
        )
        # Create source node first for row creation
        recorder.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
            node_id="source_test",
            sequence=0,
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id="source_test",
            row_index=1,
            data={"id": "test-1"},
        )
        token = recorder.create_token(
            row_id=row.row_id,
        )
        recorder.register_node(
            run_id=run.run_id,
            plugin_name="test_transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
            node_id="node_test",
            sequence=0,
        )

        # Create error record referencing the token
        with landscape_db.connection() as conn:
            conn.execute(
                transform_errors_table.insert().values(
                    error_id="terr_test001",
                    run_id=run.run_id,
                    token_id=token.token_id,
                    transform_id="node_test",
                    row_hash="abc123",
                    destination="discard",
                    created_at=datetime.now(UTC),
                )
            )
            conn.commit()

        # Act & Assert: Attempt to delete token should fail (RESTRICT)
        with (
            pytest.raises(IntegrityError, match=r"(FOREIGN KEY constraint failed|violates foreign key)"),
            landscape_db.connection() as conn,
        ):
            conn.execute(tokens_table.delete().where(tokens_table.c.token_id == token.token_id))
            conn.commit()

    def test_restrict_prevents_node_deletion(self, landscape_db: LandscapeDB, recorder: LandscapeRecorder) -> None:
        """Cannot delete node if transform_errors.transform_id references it (RESTRICT)."""
        # Arrange: Create run, source node, token, transform node, and error record
        run = recorder.begin_run(
            config={"test": True},
            canonical_version="1.0",
        )
        # Create source node first for row creation
        recorder.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
            node_id="source_test",
            sequence=0,
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id="source_test",
            row_index=1,
            data={"id": "test-1"},
        )
        token = recorder.create_token(
            row_id=row.row_id,
        )
        recorder.register_node(
            run_id=run.run_id,
            plugin_name="test_transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
            node_id="node_test",
            sequence=0,
        )

        # Create error record referencing the node via transform_id
        with landscape_db.connection() as conn:
            conn.execute(
                transform_errors_table.insert().values(
                    error_id="terr_test002",
                    run_id=run.run_id,
                    token_id=token.token_id,
                    transform_id="node_test",  # References node_id
                    row_hash="abc123",
                    destination="discard",
                    created_at=datetime.now(UTC),
                )
            )
            conn.commit()

        # Act & Assert: Attempt to delete node should fail (RESTRICT)
        with (
            pytest.raises(IntegrityError, match=r"(FOREIGN KEY constraint failed|violates foreign key)"),
            landscape_db.connection() as conn,
        ):
            conn.execute(nodes_table.delete().where(nodes_table.c.node_id == "node_test"))
            conn.commit()

    def test_accepts_valid_foreign_keys(self, landscape_db: LandscapeDB, recorder: LandscapeRecorder) -> None:
        """Valid transform_error with existing token_id and transform_id should succeed."""
        # Arrange: Create run, source node, token, and transform node
        run = recorder.begin_run(
            config={"test": True},
            canonical_version="1.0",
        )
        # Create source node first for row creation
        recorder.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
            node_id="source_test",
            sequence=0,
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id="source_test",
            row_index=1,
            data={"id": "test-1"},
        )
        token = recorder.create_token(
            row_id=row.row_id,
        )
        recorder.register_node(
            run_id=run.run_id,
            plugin_name="test_transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
            node_id="node_test",
            sequence=0,
        )

        # Act: Insert error with valid FKs (should succeed)
        with landscape_db.connection() as conn:
            conn.execute(
                transform_errors_table.insert().values(
                    error_id="terr_valid001",
                    run_id=run.run_id,
                    token_id=token.token_id,  # Valid FK
                    transform_id="node_test",  # Valid FK
                    row_hash="abc123",
                    destination="discard",
                    created_at=datetime.now(UTC),
                )
            )
            conn.commit()

        # Assert: Record was created
        with landscape_db.connection() as conn:
            error_record = conn.execute(
                select(transform_errors_table).where(transform_errors_table.c.error_id == "terr_valid001")
            ).fetchone()

        assert error_record is not None
        assert error_record.token_id == token.token_id
        assert error_record.transform_id == "node_test"

    def test_rejects_orphan_run_id(self, landscape_db: LandscapeDB, recorder: LandscapeRecorder) -> None:
        """P1: FK constraint should reject transform_errors with non-existent run_id.

        Previous tests only exercised token/node FKs. run_id is core audit
        linkage - orphan error records without a run would violate Tier 1
        audit integrity.
        """
        # Arrange: Create a run with valid token and transform for all other FKs
        run = recorder.begin_run(
            config={"test": True},
            canonical_version="1.0",
        )
        recorder.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
            node_id="source_test",
            sequence=0,
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id="source_test",
            row_index=1,
            data={"id": "test-1"},
        )
        token = recorder.create_token(
            row_id=row.row_id,
        )
        recorder.register_node(
            run_id=run.run_id,
            plugin_name="test_transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
            node_id="node_test",
            sequence=0,
        )

        # Act & Assert: Try to insert error with non-existent run_id
        with (
            pytest.raises(IntegrityError, match=r"(FOREIGN KEY constraint failed|violates foreign key)"),
            landscape_db.connection() as conn,
        ):
            conn.execute(
                transform_errors_table.insert().values(
                    error_id="terr_orphan_run",
                    run_id="nonexistent_run",  # ORPHAN - no such run
                    token_id=token.token_id,
                    transform_id="node_test",
                    row_hash="abc123",
                    destination="discard",
                    created_at=datetime.now(UTC),
                )
            )
            conn.commit()


class TestValidationErrorsForeignKeys:
    """Verify validation_errors FK constraints."""

    def test_rejects_orphan_node_id(self, landscape_db: LandscapeDB, recorder: LandscapeRecorder) -> None:
        """FK constraint should reject validation_errors with non-NULL missing node_id."""
        # Arrange: Create a run (but NOT a node)
        run = recorder.begin_run(
            config={"test": True},
            canonical_version="1.0",
        )

        # Act & Assert: Try to insert error with non-existent node_id
        with (
            pytest.raises(IntegrityError, match=r"(FOREIGN KEY constraint failed|violates foreign key)"),
            landscape_db.connection() as conn,
        ):
            conn.execute(
                validation_errors_table.insert().values(
                    error_id="verr_orphan001",
                    run_id=run.run_id,
                    node_id="nonexistent_node",  # ORPHAN - no such node
                    row_hash="abc123",
                    error="Validation failed",
                    schema_mode="strict",
                    destination="discard",
                    created_at=datetime.now(UTC),
                )
            )
            conn.commit()

    def test_allows_null_node_id(self, landscape_db: LandscapeDB, recorder: LandscapeRecorder) -> None:
        """NULL node_id should be allowed (nullable FK for early validation failures)."""
        # Arrange: Create a run (no node needed for NULL FK)
        run = recorder.begin_run(
            config={"test": True},
            canonical_version="1.0",
        )

        # Act: Insert error with NULL node_id (should succeed)
        with landscape_db.connection() as conn:
            conn.execute(
                validation_errors_table.insert().values(
                    error_id="verr_null001",
                    run_id=run.run_id,
                    node_id=None,  # NULL is valid for nullable FK
                    row_hash="abc123",
                    error="Validation failed before node association",
                    schema_mode="strict",
                    destination="discard",
                    created_at=datetime.now(UTC),
                )
            )
            conn.commit()

        # Assert: Record was created with NULL node_id
        with landscape_db.connection() as conn:
            error_record = conn.execute(
                select(validation_errors_table).where(validation_errors_table.c.error_id == "verr_null001")
            ).fetchone()

        assert error_record is not None
        assert error_record.node_id is None  # NULL FK is valid

    def test_restrict_prevents_node_deletion(self, landscape_db: LandscapeDB, recorder: LandscapeRecorder) -> None:
        """Cannot delete node if validation_errors references it (RESTRICT)."""
        # Arrange: Create run, node, and error record
        run = recorder.begin_run(
            config={"test": True},
            canonical_version="1.0",
        )
        recorder.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
            node_id="source_node",
            sequence=0,
        )

        # Create error record referencing the node
        with landscape_db.connection() as conn:
            conn.execute(
                validation_errors_table.insert().values(
                    error_id="verr_test001",
                    run_id=run.run_id,
                    node_id="source_node",  # References node
                    row_hash="abc123",
                    error="Validation failed",
                    schema_mode="strict",
                    destination="discard",
                    created_at=datetime.now(UTC),
                )
            )
            conn.commit()

        # Act & Assert: Attempt to delete node should fail (RESTRICT)
        with (
            pytest.raises(IntegrityError, match=r"(FOREIGN KEY constraint failed|violates foreign key)"),
            landscape_db.connection() as conn,
        ):
            conn.execute(nodes_table.delete().where(nodes_table.c.node_id == "source_node"))
            conn.commit()

    def test_accepts_valid_node_id(self, landscape_db: LandscapeDB, recorder: LandscapeRecorder) -> None:
        """Valid validation_error with existing node_id should succeed."""
        # Arrange: Create run and node
        run = recorder.begin_run(
            config={"test": True},
            canonical_version="1.0",
        )
        recorder.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
            node_id="source_node",
            sequence=0,
        )

        # Act: Insert error with valid node_id (should succeed)
        with landscape_db.connection() as conn:
            conn.execute(
                validation_errors_table.insert().values(
                    error_id="verr_valid001",
                    run_id=run.run_id,
                    node_id="source_node",  # Valid FK
                    row_hash="abc123",
                    error="Validation failed",
                    schema_mode="strict",
                    destination="discard",
                    created_at=datetime.now(UTC),
                )
            )
            conn.commit()

        # Assert: Record was created
        with landscape_db.connection() as conn:
            error_record = conn.execute(
                select(validation_errors_table).where(validation_errors_table.c.error_id == "verr_valid001")
            ).fetchone()

        assert error_record is not None
        assert error_record.node_id == "source_node"
