"""Tests for validation error recording with non-canonical data.

Verifies that validation errors can be recorded even when row data is:
1. Non-dict (primitives, lists, etc.)
2. Contains non-finite values (NaN, Infinity)

This is critical for the Three-Tier Trust Model - Tier 3 (external data)
must be quarantined and recorded even when malformed.
"""

import pytest

from elspeth.contracts import NodeType
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.recorder import LandscapeRecorder
from elspeth.plugins.context import PluginContext


@pytest.fixture
def recorder() -> LandscapeRecorder:
    """Create in-memory recorder with a registered run."""
    db = LandscapeDB("sqlite:///:memory:")
    rec = LandscapeRecorder(db)

    # Create a run for foreign key constraint
    rec.begin_run(
        config={},
        canonical_version="v1",
        run_id="test-run",
    )

    # Register source_node to satisfy FK constraint on validation_errors
    rec.register_node(
        run_id="test-run",
        plugin_name="test_source",
        node_type=NodeType.SOURCE,
        plugin_version="1.0",
        config={},
        schema_config=SchemaConfig.from_dict({"fields": "dynamic"}),
        node_id="source_node",
        sequence=0,
    )

    return rec


class TestValidationErrorNonCanonical:
    """Test validation error recording for non-canonical data."""

    def test_record_primitive_int(self, recorder: LandscapeRecorder) -> None:
        """Primitive int should be quarantined without crash."""
        ctx = PluginContext(
            run_id="test-run",
            config={},
            node_id="source_node",
            landscape=recorder,
        )

        token = ctx.record_validation_error(
            row=42,
            error="Expected dict, got int",
            schema_mode="dynamic",
            destination="discard",
        )

        assert token.error_id is not None
        assert token.node_id == "source_node"
        assert token.destination == "discard"

    def test_primitive_int_audit_record_verified(self, recorder: LandscapeRecorder) -> None:
        """P1: Verify persisted audit record fields for primitive int.

        Tests must verify all audit trail fields, not just error_id.
        """
        import json

        from elspeth.core.canonical import stable_hash

        ctx = PluginContext(
            run_id="test-run",
            config={},
            node_id="source_node",
            landscape=recorder,
        )

        row_value = 42
        error_msg = "Expected dict, got int"

        token = ctx.record_validation_error(
            row=row_value,
            error=error_msg,
            schema_mode="dynamic",
            destination="discard",
        )

        # Query the persisted validation error record
        records = recorder.get_validation_errors_for_run("test-run")
        assert len(records) == 1
        record = records[0]

        # Verify all persisted fields
        assert record.error_id == token.error_id
        assert record.run_id == "test-run"
        assert record.node_id == "source_node"
        assert record.error == error_msg
        assert record.schema_mode == "dynamic"
        assert record.destination == "discard"

        # Verify row_hash matches stable_hash of the primitive
        expected_hash = stable_hash(row_value)
        assert record.row_hash == expected_hash, f"row_hash mismatch: expected {expected_hash}, got {record.row_hash}"

        # Verify row_data_json is canonical JSON
        assert record.row_data_json is not None, "row_data_json should be present"
        row_data = json.loads(record.row_data_json)
        assert row_data == row_value

    def test_record_primitive_string(self, recorder: LandscapeRecorder) -> None:
        """Primitive string should be quarantined without crash."""
        ctx = PluginContext(
            run_id="test-run",
            config={},
            node_id="source_node",
            landscape=recorder,
        )

        token = ctx.record_validation_error(
            row="invalid_string",
            error="Expected dict, got str",
            schema_mode="dynamic",
            destination="discard",
        )

        assert token.error_id is not None

    def test_record_list(self, recorder: LandscapeRecorder) -> None:
        """List should be quarantined without crash."""
        ctx = PluginContext(
            run_id="test-run",
            config={},
            node_id="source_node",
            landscape=recorder,
        )

        token = ctx.record_validation_error(
            row=[1, 2, 3],
            error="Expected dict, got list",
            schema_mode="dynamic",
            destination="discard",
        )

        assert token.error_id is not None

    def test_record_nan_value(self, recorder: LandscapeRecorder) -> None:
        """Row with NaN should be quarantined without crash."""
        ctx = PluginContext(
            run_id="test-run",
            config={},
            node_id="source_node",
            landscape=recorder,
        )

        token = ctx.record_validation_error(
            row={"value": float("nan")},
            error="Row contains NaN",
            schema_mode="dynamic",
            destination="discard",
        )

        assert token.error_id is not None
        assert token.node_id == "source_node"

    def test_nan_audit_record_uses_repr_fallback(self, recorder: LandscapeRecorder) -> None:
        """P1: Verify NaN uses repr_hash and NonCanonicalMetadata.

        Non-finite floats cannot be canonicalized, so:
        - row_hash should use repr_hash (hash of repr string)
        - row_data_json should contain NonCanonicalMetadata structure
        """
        import json

        from elspeth.core.canonical import repr_hash

        ctx = PluginContext(
            run_id="test-run",
            config={},
            node_id="source_node",
            landscape=recorder,
        )

        row_value = {"value": float("nan")}
        error_msg = "Row contains NaN"

        ctx.record_validation_error(
            row=row_value,
            error=error_msg,
            schema_mode="dynamic",
            destination="discard",
        )

        # Query the persisted validation error record
        records = recorder.get_validation_errors_for_run("test-run")
        assert len(records) == 1
        record = records[0]

        # Verify row_hash uses repr_hash (not stable_hash which would crash)
        expected_hash = repr_hash(row_value)
        assert record.row_hash == expected_hash, f"row_hash mismatch for NaN: expected repr_hash {expected_hash}, got {record.row_hash}"

        # Verify row_data_json contains NonCanonicalMetadata structure
        assert record.row_data_json is not None, "row_data_json should be present"
        row_data = json.loads(record.row_data_json)
        assert "__repr__" in row_data, "NaN should use repr fallback metadata"
        assert "__type__" in row_data
        assert "__canonical_error__" in row_data
        assert row_data["__type__"] == "dict"
        assert "nan" in row_data["__repr__"].lower()

    def test_record_infinity_value(self, recorder: LandscapeRecorder) -> None:
        """Row with Infinity should be quarantined without crash."""
        ctx = PluginContext(
            run_id="test-run",
            config={},
            node_id="source_node",
            landscape=recorder,
        )

        token = ctx.record_validation_error(
            row={"value": float("inf")},
            error="Row contains Infinity",
            schema_mode="dynamic",
            destination="discard",
        )

        assert token.error_id is not None

    def test_record_negative_infinity(self, recorder: LandscapeRecorder) -> None:
        """Row with -Infinity should be quarantined without crash."""
        ctx = PluginContext(
            run_id="test-run",
            config={},
            node_id="source_node",
            landscape=recorder,
        )

        token = ctx.record_validation_error(
            row={"value": float("-inf")},
            error="Row contains -Infinity",
            schema_mode="dynamic",
            destination="discard",
        )

        assert token.error_id is not None

    def test_audit_trail_contains_repr_fallback(self, recorder: LandscapeRecorder) -> None:
        """Verify audit trail stores repr() for non-canonical data."""
        ctx = PluginContext(
            run_id="test-run",
            config={},
            node_id="source_node",
            landscape=recorder,
        )

        # Use NaN which actually triggers the fallback (primitives are canonical)
        token = ctx.record_validation_error(
            row={"value": float("nan")},
            error="Row contains NaN",
            schema_mode="dynamic",
            destination="discard",
        )

        # Query the validation_errors table to verify repr storage
        from sqlalchemy import select

        from elspeth.core.landscape.schema import validation_errors_table

        with recorder._db.connection() as conn:
            result = conn.execute(select(validation_errors_table).where(validation_errors_table.c.error_id == token.error_id))
            row = result.fetchone()

            assert row is not None
            # row_data_json should contain repr fallback metadata
            import json

            row_data = json.loads(row.row_data_json)

            # Check that it's a repr fallback (has __repr__ and __type__ keys)
            assert "__repr__" in row_data
            assert "__type__" in row_data
            assert row_data["__type__"] == "dict"
            assert "nan" in row_data["__repr__"].lower()
            assert "__canonical_error__" in row_data

    def test_multiple_non_canonical_rows(self, recorder: LandscapeRecorder) -> None:
        """Multiple non-canonical rows should all be recorded."""
        ctx = PluginContext(
            run_id="test-run",
            config={},
            node_id="source_node",
            landscape=recorder,
        )

        tokens = []
        test_rows = [
            42,
            "string",
            [1, 2, 3],
            {"nan": float("nan")},
            {"inf": float("inf")},
        ]

        for i, row in enumerate(test_rows):
            token = ctx.record_validation_error(
                row=row,
                error=f"Invalid row {i}",
                schema_mode="dynamic",
                destination="discard",
            )
            tokens.append(token)

        # All should have error_ids
        assert all(t.error_id is not None for t in tokens)

        # All should have unique error_ids
        error_ids = [t.error_id for t in tokens]
        assert len(error_ids) == len(set(error_ids))


def test_repr_hash_helper():
    """Verify repr_hash() helper produces consistent hashes."""
    from elspeth.core.canonical import repr_hash

    # Same object should produce same hash
    hash1 = repr_hash(42)
    hash2 = repr_hash(42)
    assert hash1 == hash2

    # Different objects should produce different hashes
    hash_int = repr_hash(42)
    hash_str = repr_hash("42")
    assert hash_int != hash_str

    # Non-canonical data should hash successfully
    hash_nan = repr_hash({"value": float("nan")})
    assert len(hash_nan) == 64  # SHA-256 hex digest length


def test_noncanonical_metadata_structure():
    """Verify NonCanonicalMetadata dataclass works correctly."""
    from elspeth.contracts.audit import NonCanonicalMetadata

    # Test direct creation
    metadata = NonCanonicalMetadata(
        repr_value="{'value': nan}",
        type_name="dict",
        canonical_error="Cannot canonicalize non-finite float",
    )

    # Test to_dict() produces correct structure
    meta_dict = metadata.to_dict()
    assert meta_dict["__repr__"] == "{'value': nan}"
    assert meta_dict["__type__"] == "dict"
    assert meta_dict["__canonical_error__"] == "Cannot canonicalize non-finite float"

    # Test from_error() factory
    try:
        from elspeth.core.canonical import canonical_json

        canonical_json({"value": float("nan")})
    except ValueError as e:
        metadata2 = NonCanonicalMetadata.from_error({"value": float("nan")}, e)
        assert "nan" in metadata2.repr_value.lower()
        assert metadata2.type_name == "dict"
        assert "non-finite" in metadata2.canonical_error.lower()

    # Test immutability (frozen=True)
    with pytest.raises(AttributeError):
        metadata.repr_value = "changed"  # Should fail - frozen dataclass
