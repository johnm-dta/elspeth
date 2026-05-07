# tests/plugins/sinks/test_sink_protocol_compliance.py
"""Protocol compliance tests for sink plugins.

All sink plugins must implement SinkProtocol and satisfy its contract.
This test suite verifies protocol compliance for all built-in sinks.

Tests cover:
1. Required attributes (class and instance level)
2. write() behavior - data written correctly, returns ArtifactDescriptor
3. flush() behavior - buffered data persisted
4. close() behavior - resources released, idempotent
5. Lifecycle hooks (on_start, on_complete)
6. Resume support (configure_for_resume, validate_output_target)
"""

import csv
import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from elspeth.contracts import ArtifactDescriptor
from tests.fixtures.base_classes import inject_write_failure

# Schema configs for tests
# CSV and Database sinks require fixed columns (strict mode)
# JSON sink accepts dynamic schemas
STRICT_SCHEMA = {"mode": "fixed", "fields": ["id: int", "name: str"]}
DYNAMIC_SCHEMA = {"mode": "observed"}


def _import_sink_class(class_path: str) -> type:
    """Import a sink class from its fully qualified path."""
    module_path, class_name = class_path.rsplit(".", 1)
    import importlib

    module = importlib.import_module(module_path)
    cls: type = getattr(module, class_name)
    return cls


def _create_temp_path(suffix: str) -> Path:
    """Create a temporary file path for testing."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    import os

    os.close(fd)
    return Path(path)


# Sink configurations for parametrized testing
# Each tuple: (class_path, config_factory, expected_name, file_suffix)
# Using factories instead of static configs so each test gets a fresh temp file
def _csv_config() -> dict[str, Any]:
    return {"path": str(_create_temp_path(".csv")), "schema": STRICT_SCHEMA}


def _json_config() -> dict[str, Any]:
    return {"path": str(_create_temp_path(".json")), "schema": DYNAMIC_SCHEMA}


def _jsonl_config() -> dict[str, Any]:
    return {"path": str(_create_temp_path(".jsonl")), "schema": DYNAMIC_SCHEMA}


def _database_config() -> dict[str, Any]:
    return {"url": "sqlite:///:memory:", "table": "test", "schema": STRICT_SCHEMA}


# Parametrized test configs
SINK_CONFIGS = [
    pytest.param(
        "elspeth.plugins.sinks.csv_sink.CSVSink",
        _csv_config,
        "csv",
        id="csv",
    ),
    pytest.param(
        "elspeth.plugins.sinks.json_sink.JSONSink",
        _json_config,
        "json",
        id="json",
    ),
    pytest.param(
        "elspeth.plugins.sinks.database_sink.DatabaseSink",
        _database_config,
        "database",
        id="database",
    ),
]


def _create_mock_context() -> MagicMock:
    """Create a mock PluginContext for testing."""
    mock_ctx = MagicMock()
    mock_ctx.run_id = "test-run"
    mock_ctx.landscape = None  # Not needed for basic tests
    mock_ctx.contract = None  # Not needed for basic tests
    mock_ctx.record_call = MagicMock()  # Mock the record_call method
    return mock_ctx


class TestSinkProtocolCompliance:
    """Parametrized protocol compliance tests for all sink plugins."""

    @pytest.mark.parametrize("class_path,config_factory,expected_name", SINK_CONFIGS)
    def test_has_required_class_attributes(self, class_path: str, config_factory: Any, expected_name: str) -> None:
        """All sinks must have name class attribute."""
        sink_class = _import_sink_class(class_path)
        # Direct attribute access - crash on missing (our code, our bug)
        assert sink_class.name == expected_name  # type: ignore[attr-defined]

    @pytest.mark.parametrize("class_path,config_factory,expected_name", SINK_CONFIGS)
    def test_has_required_instance_attributes(self, class_path: str, config_factory: Any, expected_name: str) -> None:
        """All sinks must have input_schema, idempotent, supports_resume attributes after instantiation."""
        sink_class = _import_sink_class(class_path)
        sink = inject_write_failure(sink_class(config_factory()))

        # Direct attribute access - crash on missing (our code, our bug)
        _ = sink.input_schema  # Verify attribute exists
        _ = sink.idempotent  # Verify attribute exists
        _ = sink.supports_resume  # Verify attribute exists
        _ = sink.determinism  # Verify attribute exists
        _ = sink.plugin_version  # Verify attribute exists
        _ = sink.config  # Verify attribute exists

        # Clean up
        sink.close()

    @pytest.mark.parametrize("class_path,config_factory,expected_name", SINK_CONFIGS)
    def test_write_empty_batch_returns_descriptor(self, class_path: str, config_factory: Any, expected_name: str) -> None:
        """write() with empty list should return ArtifactDescriptor without error."""
        sink_class = _import_sink_class(class_path)
        sink = inject_write_failure(sink_class(config_factory()))
        mock_ctx = _create_mock_context()

        # Call write with empty list (should not crash)
        result = sink.write([], mock_ctx)

        # Verify return type - write() now returns SinkWriteResult wrapping ArtifactDescriptor
        from elspeth.contracts.diversion import SinkWriteResult

        assert isinstance(result, SinkWriteResult), f"write() must return SinkWriteResult, got {type(result)}"
        assert isinstance(result.artifact, ArtifactDescriptor), f"write().artifact must be ArtifactDescriptor, got {type(result.artifact)}"
        # Verify required fields exist (our code, crash on missing)
        _ = result.artifact.content_hash
        _ = result.artifact.size_bytes

        # Clean up
        sink.close()

    @pytest.mark.parametrize("class_path,config_factory,expected_name", SINK_CONFIGS)
    def test_write_with_data_returns_descriptor_with_valid_hash(self, class_path: str, config_factory: Any, expected_name: str) -> None:
        """write() with actual data should return ArtifactDescriptor with valid hash."""
        sink_class = _import_sink_class(class_path)
        sink = inject_write_failure(sink_class(config_factory()))
        mock_ctx = _create_mock_context()

        # Write actual data
        test_rows = [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"},
        ]
        result = sink.write(test_rows, mock_ctx)

        # Verify return type
        from elspeth.contracts.diversion import SinkWriteResult

        assert isinstance(result, SinkWriteResult)

        # Content hash should be non-empty hex string
        assert result.artifact.content_hash, "content_hash should not be empty"
        assert all(c in "0123456789abcdef" for c in result.artifact.content_hash), "content_hash should be hex string"

        # Size should be positive (we wrote data)
        assert result.artifact.size_bytes > 0 or expected_name == "database"  # DB uses payload_size

        # Clean up
        sink.close()

    @pytest.mark.parametrize("class_path,config_factory,expected_name", SINK_CONFIGS)
    def test_flush_method_callable(self, class_path: str, config_factory: Any, expected_name: str) -> None:
        """All sinks must have callable flush() method."""
        sink_class = _import_sink_class(class_path)
        sink = inject_write_failure(sink_class(config_factory()))

        # Direct method call - crash on missing (our code, our bug)
        sink.flush()  # Should not raise

        # Clean up
        sink.close()

    @pytest.mark.parametrize("class_path,config_factory,expected_name", SINK_CONFIGS)
    def test_close_method_callable_and_idempotent(self, class_path: str, config_factory: Any, expected_name: str) -> None:
        """All sinks must have callable close() method that is idempotent."""
        sink_class = _import_sink_class(class_path)
        sink = inject_write_failure(sink_class(config_factory()))

        # Direct method call - crash on missing (our code, our bug)
        sink.close()  # First close
        sink.close()  # Second close - should not raise (idempotency)

    @pytest.mark.parametrize("class_path,config_factory,expected_name", SINK_CONFIGS)
    def test_lifecycle_hooks_exist(self, class_path: str, config_factory: Any, expected_name: str) -> None:
        """All sinks must have on_start() and on_complete() lifecycle hooks."""
        sink_class = _import_sink_class(class_path)
        sink = inject_write_failure(sink_class(config_factory()))

        # Create mock context
        mock_ctx = _create_mock_context()

        # Direct method calls - crash on missing (our code, our bug)
        sink.on_start(mock_ctx)  # Should not raise
        sink.on_complete(mock_ctx)  # Should not raise

        # Clean up
        sink.close()

    @pytest.mark.parametrize("class_path,config_factory,expected_name", SINK_CONFIGS)
    def test_resume_methods_exist(self, class_path: str, config_factory: Any, expected_name: str) -> None:
        """All sinks must have configure_for_resume() and validate_output_target() methods."""
        sink_class = _import_sink_class(class_path)
        sink = inject_write_failure(sink_class(config_factory()))

        # Direct attribute access - crash on missing (our code, our bug)
        supports_resume = sink.supports_resume

        # configure_for_resume() should only be called if sink supports resume
        # Sinks that don't support resume may raise NotImplementedError
        if supports_resume:
            sink.configure_for_resume()  # Should not raise for resumable sinks

        # validate_output_target() should always be callable
        result = sink.validate_output_target()  # Should not raise

        # Verify return value has expected structure
        _ = result.valid  # Crash if missing field (our code, our bug)

        # Clean up
        sink.close()


class TestCSVSinkWriteBehavior:
    """CSV sink-specific write behavior tests."""

    def test_write_creates_file_with_headers(self) -> None:
        """CSV sink should create file with headers on first write."""
        from elspeth.plugins.sinks.csv_sink import CSVSink

        path = _create_temp_path(".csv")
        sink = inject_write_failure(CSVSink({"path": str(path), "schema": STRICT_SCHEMA}))
        mock_ctx = _create_mock_context()

        sink.write([{"id": 1, "name": "Alice"}], mock_ctx)
        sink.flush()
        sink.close()

        # Verify file contents
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["id"] == "1"  # CSV reads as strings
        assert rows[0]["name"] == "Alice"

        # Clean up temp file
        path.unlink()

    def test_write_multiple_batches_accumulates_data(self) -> None:
        """CSV sink should accumulate data across multiple write() calls."""
        from elspeth.plugins.sinks.csv_sink import CSVSink

        path = _create_temp_path(".csv")
        sink = inject_write_failure(CSVSink({"path": str(path), "schema": STRICT_SCHEMA}))
        mock_ctx = _create_mock_context()

        sink.write([{"id": 1, "name": "Alice"}], mock_ctx)
        sink.write([{"id": 2, "name": "Bob"}], mock_ctx)
        sink.flush()
        sink.close()

        # Verify file contents
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2
        assert rows[0]["name"] == "Alice"
        assert rows[1]["name"] == "Bob"

        # Clean up temp file
        path.unlink()

    def test_flush_persists_data_to_disk(self) -> None:
        """flush() should ensure data is persisted to disk."""
        from elspeth.plugins.sinks.csv_sink import CSVSink

        path = _create_temp_path(".csv")
        sink = inject_write_failure(CSVSink({"path": str(path), "schema": STRICT_SCHEMA}))
        mock_ctx = _create_mock_context()

        sink.write([{"id": 1, "name": "Test"}], mock_ctx)
        sink.flush()  # Force persistence

        # File should be readable immediately after flush (before close)
        with open(path) as f:
            content = f.read()
            assert "Test" in content

        sink.close()
        path.unlink()

    def test_close_releases_file_handle(self) -> None:
        """close() should release file handle allowing deletion."""
        from elspeth.plugins.sinks.csv_sink import CSVSink

        path = _create_temp_path(".csv")
        sink = inject_write_failure(CSVSink({"path": str(path), "schema": STRICT_SCHEMA}))
        mock_ctx = _create_mock_context()

        sink.write([{"id": 1, "name": "Test"}], mock_ctx)
        sink.close()

        # Should be able to delete file after close
        path.unlink()
        assert not path.exists()


class TestJSONSinkWriteBehavior:
    """JSON sink-specific write behavior tests."""

    def test_write_json_array_format(self) -> None:
        """JSON sink should write data as JSON array."""
        from elspeth.plugins.sinks.json_sink import JSONSink

        path = _create_temp_path(".json")
        sink = inject_write_failure(JSONSink({"path": str(path), "schema": DYNAMIC_SCHEMA, "format": "json"}))
        mock_ctx = _create_mock_context()

        sink.write([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}], mock_ctx)
        sink.flush()
        sink.close()

        # Verify file contents
        with open(path) as f:
            data = json.load(f)

        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0]["name"] == "Alice"
        assert data[1]["name"] == "Bob"

        path.unlink()

    def test_write_jsonl_format(self) -> None:
        """JSON sink should write data as JSONL (one object per line)."""
        from elspeth.plugins.sinks.json_sink import JSONSink

        path = _create_temp_path(".jsonl")
        sink = inject_write_failure(JSONSink({"path": str(path), "schema": DYNAMIC_SCHEMA, "format": "jsonl"}))
        mock_ctx = _create_mock_context()

        sink.write([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}], mock_ctx)
        sink.flush()
        sink.close()

        # Verify file contents (one JSON object per line)
        with open(path) as f:
            lines = f.readlines()

        assert len(lines) == 2
        assert json.loads(lines[0])["name"] == "Alice"
        assert json.loads(lines[1])["name"] == "Bob"

        path.unlink()

    def test_jsonl_multiple_writes_append(self) -> None:
        """JSONL format should append data across multiple writes."""
        from elspeth.plugins.sinks.json_sink import JSONSink

        path = _create_temp_path(".jsonl")
        sink = inject_write_failure(JSONSink({"path": str(path), "schema": DYNAMIC_SCHEMA, "format": "jsonl"}))
        mock_ctx = _create_mock_context()

        sink.write([{"id": 1, "name": "Alice"}], mock_ctx)
        sink.write([{"id": 2, "name": "Bob"}], mock_ctx)
        sink.flush()
        sink.close()

        with open(path) as f:
            lines = f.readlines()

        assert len(lines) == 2

        path.unlink()


class TestDatabaseSinkWriteBehavior:
    """Database sink-specific write behavior tests."""

    def test_write_creates_table_and_inserts(self) -> None:
        """Database sink should create table and insert rows."""
        from sqlalchemy import text

        from elspeth.plugins.sinks.database_sink import DatabaseSink

        # Use a unique in-memory database
        db_url = "sqlite:///:memory:"
        sink = inject_write_failure(DatabaseSink({"url": db_url, "table": "test_table", "schema": STRICT_SCHEMA}))
        mock_ctx = _create_mock_context()

        sink.write([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}], mock_ctx)
        sink.flush()

        # Query the database to verify data
        # Need to use sink's engine since :memory: doesn't persist between connections
        engine = sink._engine
        assert engine is not None  # Engine is created on first write
        with engine.connect() as conn:
            result = conn.execute(text("SELECT id, name FROM test_table ORDER BY id"))
            rows = list(result)

        assert len(rows) == 2
        assert rows[0][0] == 1
        assert rows[0][1] == "Alice"
        assert rows[1][0] == 2
        assert rows[1][1] == "Bob"

        sink.close()

    def test_write_multiple_batches_inserts_all(self) -> None:
        """Database sink should insert all rows from multiple write() calls."""
        from sqlalchemy import text

        from elspeth.plugins.sinks.database_sink import DatabaseSink

        db_url = "sqlite:///:memory:"
        sink = inject_write_failure(DatabaseSink({"url": db_url, "table": "test_table", "schema": STRICT_SCHEMA}))
        mock_ctx = _create_mock_context()

        sink.write([{"id": 1, "name": "Alice"}], mock_ctx)
        sink.write([{"id": 2, "name": "Bob"}], mock_ctx)
        sink.flush()

        engine = sink._engine
        assert engine is not None  # Engine is created on first write
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM test_table"))
            count = result.scalar()

        assert count == 2

        sink.close()

    def test_close_disposes_engine(self) -> None:
        """close() should dispose of database engine."""
        from elspeth.plugins.sinks.database_sink import DatabaseSink

        db_url = "sqlite:///:memory:"
        sink = inject_write_failure(DatabaseSink({"url": db_url, "table": "test_table", "schema": STRICT_SCHEMA}))
        mock_ctx = _create_mock_context()

        sink.write([{"id": 1, "name": "Test"}], mock_ctx)

        # Engine should exist before close
        assert sink._engine is not None

        sink.close()

        # Engine should be None after close
        assert sink._engine is None


class TestSinkContentHashConsistency:
    """Tests verifying content hash behavior across sinks."""

    def test_same_data_produces_consistent_hash(self) -> None:
        """Writing the same data should produce the same content hash."""
        from elspeth.plugins.sinks.csv_sink import CSVSink

        path1 = _create_temp_path(".csv")
        path2 = _create_temp_path(".csv")

        sink1 = inject_write_failure(CSVSink({"path": str(path1), "schema": STRICT_SCHEMA}))
        sink2 = inject_write_failure(CSVSink({"path": str(path2), "schema": STRICT_SCHEMA}))
        mock_ctx = _create_mock_context()

        test_data = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]

        result1 = sink1.write(test_data, mock_ctx)
        result2 = sink2.write(test_data, mock_ctx)

        sink1.close()
        sink2.close()

        # Same data should produce same hash
        assert result1.artifact.content_hash == result2.artifact.content_hash

        path1.unlink()
        path2.unlink()

    def test_different_data_produces_different_hash(self) -> None:
        """Writing different data should produce different content hashes."""
        from elspeth.plugins.sinks.csv_sink import CSVSink

        path1 = _create_temp_path(".csv")
        path2 = _create_temp_path(".csv")

        sink1 = inject_write_failure(CSVSink({"path": str(path1), "schema": STRICT_SCHEMA}))
        sink2 = inject_write_failure(CSVSink({"path": str(path2), "schema": STRICT_SCHEMA}))
        mock_ctx = _create_mock_context()

        result1 = sink1.write([{"id": 1, "name": "Alice"}], mock_ctx)
        result2 = sink2.write([{"id": 2, "name": "Bob"}], mock_ctx)

        sink1.close()
        sink2.close()

        # Different data should produce different hashes
        assert result1.artifact.content_hash != result2.artifact.content_hash

        path1.unlink()
        path2.unlink()

    def test_empty_write_has_consistent_empty_hash(self) -> None:
        """Empty writes should produce consistent hash (SHA-256 of empty string)."""
        from elspeth.plugins.sinks.csv_sink import CSVSink

        path1 = _create_temp_path(".csv")
        path2 = _create_temp_path(".csv")

        sink1 = inject_write_failure(CSVSink({"path": str(path1), "schema": STRICT_SCHEMA}))
        sink2 = inject_write_failure(CSVSink({"path": str(path2), "schema": STRICT_SCHEMA}))
        mock_ctx = _create_mock_context()

        result1 = sink1.write([], mock_ctx)
        result2 = sink2.write([], mock_ctx)

        sink1.close()
        sink2.close()

        # Empty writes should produce same hash
        assert result1.artifact.content_hash == result2.artifact.content_hash
        # Should be SHA-256 of empty bytes
        import hashlib

        expected = hashlib.sha256(b"").hexdigest()
        assert result1.artifact.content_hash == expected

        path1.unlink()
        path2.unlink()
