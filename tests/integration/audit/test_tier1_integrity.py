# tests/integration/audit/test_tier1_integrity.py
"""Tier 1 audit integrity tests for RecorderFactory.

Per the Three-Tier Trust Model, the audit database is FULL TRUST (Tier 1).
Bad data in the audit trail means corruption or tampering -- the response
must be to crash immediately.  Never silently coerce, default, or recover.

These tests verify two properties:
1. CRASH: Invalid enum values, NULL required fields, and wrong types crash
   the recorder (via TypeError, ValueError, AttributeError, or IntegrityError).
2. FIDELITY: Valid audit writes produce exact records -- every field round-trips
   without silent transformation, and hashes match canonical_json(data).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import select

from elspeth.contracts import (
    Determinism,
    Node,
    NodeType,
    Run,
    RunStatus,
)
from elspeth.contracts.audit import TokenRef
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.canonical import CANONICAL_VERSION, canonical_json, stable_hash
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.core.landscape.schema import nodes_table, rows_table

# Dynamic schema for tests that do not care about specific field definitions
DYNAMIC_SCHEMA = SchemaConfig.from_dict({"mode": "observed"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _begin_run(factory: RecorderFactory) -> Run:
    """Create a minimal run for tests that need a prerequisite run."""
    return factory.run_lifecycle.begin_run(config={"test": True}, canonical_version=CANONICAL_VERSION)


def _register_source(factory: RecorderFactory, run_id: str) -> Node:
    """Register a minimal source node for tests that need one."""
    return factory.data_flow.register_node(
        run_id=run_id,
        plugin_name="csv_source",
        node_type=NodeType.SOURCE,
        plugin_version="1.0",
        config={"path": "test.csv"},
        schema_config=DYNAMIC_SCHEMA,
    )


# ===================================================================
# TestRecorderCrashesOnInvalidEnums
# ===================================================================


class TestRecorderCrashesOnInvalidEnums:
    """Tier 1 invariant: invalid enum values in audit records must crash.

    The recorder stores enum values via `.value` on StrEnum instances.
    Passing a raw string or nonsense value instead of the expected enum
    must raise an error -- never silently store garbage.
    """

    def test_invalid_run_status_crashes(self, landscape_db: LandscapeDB, landscape_factory: RecorderFactory) -> None:
        """complete_run with a bogus status string must crash.

        complete_run calls `status.value` -- a plain string has no `.value`,
        so this raises AttributeError.
        """
        run = _begin_run(landscape_factory)

        with pytest.raises((AttributeError, TypeError, ValueError)):
            landscape_factory.run_lifecycle.complete_run(run.run_id, status="bogus_status")  # type: ignore[arg-type]

    def test_invalid_node_type_crashes(self, landscape_db: LandscapeDB, landscape_factory: RecorderFactory) -> None:
        """register_node with an invalid node_type must crash.

        The recorder calls `node_type.value` to store the string representation.
        A raw string has no `.value` attribute.
        """
        run = _begin_run(landscape_factory)

        with pytest.raises((AttributeError, TypeError, ValueError)):
            landscape_factory.data_flow.register_node(
                run_id=run.run_id,
                plugin_name="test_plugin",
                node_type="not_a_valid_type",  # type: ignore[arg-type]
                plugin_version="1.0",
                config={},
                schema_config=DYNAMIC_SCHEMA,
            )

    def test_invalid_determinism_crashes(self, landscape_db: LandscapeDB, landscape_factory: RecorderFactory) -> None:
        """register_node with invalid determinism must crash.

        Same pattern: determinism.value is called, raw string crashes.
        """
        run = _begin_run(landscape_factory)

        with pytest.raises((AttributeError, TypeError, ValueError)):
            landscape_factory.data_flow.register_node(
                run_id=run.run_id,
                plugin_name="test_plugin",
                node_type=NodeType.TRANSFORM,
                plugin_version="1.0",
                config={},
                determinism="totally_random",  # type: ignore[arg-type]
                schema_config=DYNAMIC_SCHEMA,
            )

    def test_invalid_terminal_state_crashes(self, landscape_db: LandscapeDB, landscape_factory: RecorderFactory) -> None:
        """record_token_outcome with invalid outcome must crash.

        The method calls outcome.value and outcome.is_terminal on the enum.
        A raw string has neither attribute.
        """
        run = _begin_run(landscape_factory)
        source: Node = _register_source(landscape_factory, run.run_id)
        row = landscape_factory.data_flow.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data={"x": 1},
        )
        token = landscape_factory.data_flow.create_token(row.row_id)

        with pytest.raises((AttributeError, TypeError, ValueError)):
            landscape_factory.data_flow.record_token_outcome(
                ref=TokenRef(token_id=token.token_id, run_id=run.run_id),
                outcome="bogus_terminal",  # type: ignore[arg-type]
                sink_name="output",
            )


# ===================================================================
# TestRecorderCrashesOnNullAuditFields
# ===================================================================


class TestRecorderCrashesOnNullAuditFields:
    """Tier 1 invariant: NULL in required audit fields must crash.

    Schema columns marked nullable=False enforce NOT NULL at the database
    level.  The recorder must not silently allow NULL through.
    """

    def test_null_plugin_name_in_register_node_crashes(self, landscape_db: LandscapeDB, landscape_factory: RecorderFactory) -> None:
        """register_node with None as plugin_name must crash.

        The nodes table has plugin_name NOT NULL.  Passing None should
        surface as LandscapeRecordError with the database IntegrityError as cause.
        """
        from sqlalchemy.exc import IntegrityError

        from elspeth.core.landscape.errors import LandscapeRecordError

        run = _begin_run(landscape_factory)

        with pytest.raises(LandscapeRecordError) as exc_info:
            landscape_factory.data_flow.register_node(
                run_id=run.run_id,
                plugin_name=None,  # type: ignore[arg-type]
                node_type=NodeType.SOURCE,
                plugin_version="1.0",
                config={},
                schema_config=DYNAMIC_SCHEMA,
            )
        assert isinstance(exc_info.value.__cause__, IntegrityError)

    def test_null_config_hash_prevented(self, landscape_db: LandscapeDB, landscape_factory: RecorderFactory) -> None:
        """register_node computes config_hash from config dict.

        Even with an empty config, the hash is computed (not NULL).
        Verify the stored config_hash is always non-NULL by checking
        the returned Node object.
        """
        run = _begin_run(landscape_factory)
        node = landscape_factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="test_plugin",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # config_hash must never be None -- the recorder computes it
        assert node.config_hash is not None
        assert len(node.config_hash) > 0

    def test_null_run_id_in_register_node_crashes(self, landscape_db: LandscapeDB, landscape_factory: RecorderFactory) -> None:
        """register_node with None as run_id must crash.

        The nodes table has run_id NOT NULL with a FK to runs; repository writes
        surface DB rejection as LandscapeRecordError.
        """
        from sqlalchemy.exc import IntegrityError

        from elspeth.core.landscape.errors import LandscapeRecordError

        with pytest.raises(LandscapeRecordError) as exc_info:
            landscape_factory.data_flow.register_node(
                run_id=None,  # type: ignore[arg-type]
                plugin_name="test_plugin",
                node_type=NodeType.SOURCE,
                plugin_version="1.0",
                config={},
                schema_config=DYNAMIC_SCHEMA,
            )
        assert isinstance(exc_info.value.__cause__, IntegrityError)


# ===================================================================
# TestRecorderCrashesOnWrongTypes
# ===================================================================


class TestRecorderCrashesOnWrongTypes:
    """Tier 1 invariant: wrong types in audit fields must crash.

    The recorder's dataclass contracts and enum validation provide
    the type enforcement layer.  These tests verify that the Python-level
    type contracts catch wrong types before or during construction.
    """

    def test_string_status_in_run_dataclass_crashes(self, landscape_db: LandscapeDB, landscape_factory: RecorderFactory) -> None:
        """Constructing a Run with a plain string status must crash.

        The Run dataclass __post_init__ calls _validate_enum() which
        raises TypeError if status is not a RunStatus instance.
        """
        with pytest.raises(TypeError, match="status must be RunStatus"):
            Run(
                run_id="test",
                started_at=datetime.now(UTC),
                config_hash="abc",
                settings_json="{}",
                canonical_version="v1",
                status="running",  # type: ignore[arg-type]
            )

    def test_non_hashable_data_in_create_row_crashes(self, landscape_db: LandscapeDB, landscape_factory: RecorderFactory) -> None:
        """create_row with un-serializable data must crash.

        stable_hash calls canonical_json which rejects non-serializable values.
        Passing data that contains non-JSON-serializable objects (e.g., a set)
        must raise TypeError or ValueError, not silently store garbage.
        """
        run = _begin_run(landscape_factory)
        source: Node = _register_source(landscape_factory, run.run_id)

        with pytest.raises((TypeError, ValueError)):
            landscape_factory.data_flow.create_row(
                run_id=run.run_id,
                source_node_id=source.node_id,
                row_index=0,
                data={"bad": {1, 2, 3}},  # Sets are not JSON-serializable
            )


# ===================================================================
# TestRecorderPositiveAuditIntegrity
# ===================================================================


class TestRecorderPositiveAuditIntegrity:
    """Tier 1 invariant: valid audit writes produce exact records.

    Every field written via the recorder must round-trip exactly.
    No silent transformation, truncation, or timezone loss.
    """

    def test_run_round_trips_exactly(self, landscape_db: LandscapeDB, landscape_factory: RecorderFactory) -> None:
        """begin_run -> get_run must return identical field values.

        Note: SQLite drops timezone info on DateTime round-trip, so
        timestamp comparison uses replace(tzinfo=None) to compare the
        actual datetime value independent of the tzinfo slot.  The
        test_timestamps_are_utc test separately verifies that the
        recorder's in-memory objects carry UTC timezone info.
        """
        config = {"source": "data.csv", "transforms": [{"plugin": "passthrough"}]}
        run = landscape_factory.run_lifecycle.begin_run(config=config, canonical_version=CANONICAL_VERSION)

        retrieved = landscape_factory.run_lifecycle.get_run(run.run_id)
        assert retrieved is not None

        assert retrieved.run_id == run.run_id
        assert retrieved.config_hash == run.config_hash
        assert retrieved.settings_json == run.settings_json
        assert retrieved.canonical_version == CANONICAL_VERSION
        assert retrieved.status == RunStatus.RUNNING
        # Compare timestamps stripped of tzinfo (SQLite drops it on round-trip)
        assert retrieved.started_at.replace(tzinfo=None) == run.started_at.replace(tzinfo=None)
        assert retrieved.completed_at is None

    def test_node_registration_round_trips(self, landscape_db: LandscapeDB, landscape_factory: RecorderFactory) -> None:
        """register_node -> get_node must return identical field values."""
        run = _begin_run(landscape_factory)
        config = {"path": "data.csv", "delimiter": ","}

        node = landscape_factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
            node_type=NodeType.SOURCE,
            plugin_version="2.1.0",
            config=config,
            determinism=Determinism.DETERMINISTIC,
            sequence=0,
            schema_config=DYNAMIC_SCHEMA,
        )

        retrieved = landscape_factory.data_flow.get_node(node.node_id, run.run_id)
        assert retrieved is not None

        assert retrieved.node_id == node.node_id
        assert retrieved.run_id == run.run_id
        assert retrieved.plugin_name == "csv_source"
        assert retrieved.node_type == NodeType.SOURCE
        assert retrieved.plugin_version == "2.1.0"
        assert retrieved.determinism == Determinism.DETERMINISTIC
        assert retrieved.config_hash == node.config_hash
        assert retrieved.config_json == node.config_json
        assert retrieved.sequence_in_pipeline == 0
        # Compare timestamps stripped of tzinfo (SQLite drops it on round-trip)
        assert retrieved.registered_at.replace(tzinfo=None) == node.registered_at.replace(tzinfo=None)

    def test_row_creation_round_trips(self, landscape_db: LandscapeDB, landscape_factory: RecorderFactory) -> None:
        """create_row -> query back must produce matching source_data_hash.

        The hash stored in the rows table must equal stable_hash(data).
        """
        run = _begin_run(landscape_factory)
        source: Node = _register_source(landscape_factory, run.run_id)
        data = {"customer_id": "C-123", "amount": 99.95, "currency": "USD"}

        row = landscape_factory.data_flow.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data=data,
        )

        # Verify returned object
        expected_hash = stable_hash(data)
        assert row.source_data_hash == expected_hash

        # Verify database record
        with landscape_db.engine.connect() as conn:
            db_row = conn.execute(select(rows_table).where(rows_table.c.row_id == row.row_id)).fetchone()

        assert db_row is not None
        assert db_row.source_data_hash == expected_hash
        assert db_row.row_index == 0
        assert db_row.run_id == run.run_id
        assert db_row.source_node_id == source.node_id

    def test_timestamps_are_utc(self, landscape_db: LandscapeDB, landscape_factory: RecorderFactory) -> None:
        """All recorded timestamps must have UTC timezone info.

        Naive timestamps in the audit trail would make cross-timezone
        comparison impossible -- a direct audit integrity failure.
        """
        run = _begin_run(landscape_factory)
        source: Node = _register_source(landscape_factory, run.run_id)
        row = landscape_factory.data_flow.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data={"x": 1},
        )

        # Run timestamp
        assert run.started_at.tzinfo is not None, "run.started_at must be timezone-aware"

        # Node timestamp
        assert source.registered_at.tzinfo is not None, "node.registered_at must be timezone-aware"

        # Row timestamp
        assert row.created_at.tzinfo is not None, "row.created_at must be timezone-aware"

        # Token timestamp
        token = landscape_factory.data_flow.create_token(row.row_id)
        assert token.created_at.tzinfo is not None, "token.created_at must be timezone-aware"


# ===================================================================
# TestRecorderHashIntegrity
# ===================================================================


class TestRecorderHashIntegrity:
    """Tier 1 invariant: stored hashes match canonical_json(data).

    Hashes survive payload deletion -- they are the permanent integrity
    proof.  If hash != canonical_json(data), audit integrity is broken.
    """

    def test_config_hash_matches_canonical(self, landscape_db: LandscapeDB, landscape_factory: RecorderFactory) -> None:
        """register_node must store config_hash == stable_hash(config).

        The config_hash is the integrity proof for the node configuration.
        It must be deterministically derived from canonical_json(config).
        """
        run = _begin_run(landscape_factory)
        config = {"model": "gpt-4", "temperature": 0.7, "max_tokens": 100}

        node = landscape_factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="llm_transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config=config,
            schema_config=DYNAMIC_SCHEMA,
        )

        expected_hash = stable_hash(config)
        assert node.config_hash == expected_hash

        # Also verify it matches canonical_json -> SHA256
        canonical = canonical_json(config)
        import hashlib

        manual_hash = hashlib.sha256(canonical.encode()).hexdigest()
        assert node.config_hash == manual_hash

        # Verify DB agrees
        with landscape_db.engine.connect() as conn:
            db_row = conn.execute(
                select(nodes_table).where((nodes_table.c.node_id == node.node_id) & (nodes_table.c.run_id == run.run_id))
            ).fetchone()

        assert db_row is not None
        assert db_row.config_hash == expected_hash

    def test_source_data_hash_matches_canonical(self, landscape_db: LandscapeDB, landscape_factory: RecorderFactory) -> None:
        """create_row must store source_data_hash == stable_hash(data).

        The source_data_hash is the permanent fingerprint of what the
        source loaded.  It must exactly match canonical_json(data) hashed.
        """
        run = _begin_run(landscape_factory)
        source: Node = _register_source(landscape_factory, run.run_id)
        data = {
            "id": 42,
            "name": "Alice",
            "score": 98.6,
            "active": True,
            "tags": ["premium", "verified"],
        }

        row = landscape_factory.data_flow.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data=data,
        )

        expected_hash = stable_hash(data)
        assert row.source_data_hash == expected_hash

        # Verify via manual hash of canonical_json
        canonical = canonical_json(data)
        import hashlib

        manual_hash = hashlib.sha256(canonical.encode()).hexdigest()
        assert row.source_data_hash == manual_hash

    def test_run_config_hash_matches_canonical(self, landscape_db: LandscapeDB, landscape_factory: RecorderFactory) -> None:
        """begin_run must store config_hash == stable_hash(config).

        The run-level config_hash is the integrity proof for the entire
        pipeline configuration used for this run.
        """
        config = {
            "source": {"plugin": "csv", "path": "data.csv"},
            "transforms": [{"plugin": "passthrough"}],
            "sinks": [{"plugin": "csv", "path": "output.csv"}],
        }

        run = landscape_factory.run_lifecycle.begin_run(config=config, canonical_version=CANONICAL_VERSION)

        expected_hash = stable_hash(config)
        assert run.config_hash == expected_hash

        # Verify settings_json matches canonical_json
        expected_json = canonical_json(config)
        assert run.settings_json == expected_json

    def test_settings_json_is_canonical(self, landscape_db: LandscapeDB, landscape_factory: RecorderFactory) -> None:
        """settings_json must be canonical JSON (RFC 8785 deterministic).

        If two runs have identical config, their settings_json must be
        byte-for-byte identical.  This is essential for config drift
        detection across runs.
        """
        config = {"z_last": 1, "a_first": 2, "m_middle": 3}

        run1 = landscape_factory.run_lifecycle.begin_run(
            config=config,
            canonical_version=CANONICAL_VERSION,
            run_id="run-canonical-1",
        )
        run2 = landscape_factory.run_lifecycle.begin_run(
            config=config,
            canonical_version=CANONICAL_VERSION,
            run_id="run-canonical-2",
        )

        # Canonical JSON is deterministic -- key order does not matter
        assert run1.settings_json == run2.settings_json
        assert run1.config_hash == run2.config_hash
