"""Consolidated tests for sink output target schema validation.

This module tests schema validation behavior that is common across all sink types:
- CSVSink
- JSONSink
- DatabaseSink

Each sink type has slightly different setup requirements, handled by factory fixtures.
"""

import csv
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import pytest
from sqlalchemy import Column, MetaData, String, Table, create_engine

from elspeth.plugins.sinks.csv_sink import CSVSink
from elspeth.plugins.sinks.database_sink import DatabaseSink
from elspeth.plugins.sinks.json_sink import JSONSink

# =============================================================================
# Type Definitions
# =============================================================================


class SinkProtocol(Protocol):
    """Protocol for sinks that support schema validation."""

    def validate_output_target(self) -> Any: ...

    def close(self) -> None: ...


SinkFactory = Callable[[list[str], dict[str, Any]], SinkProtocol]


# =============================================================================
# Sink Factory Fixtures
# =============================================================================


def _create_csv_with_headers(path: Path, headers: list[str]) -> None:
    """Create a CSV file with the given headers and no data rows."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()


def _create_jsonl_with_record(path: Path, fields: list[str]) -> None:
    """Create a JSONL file with a single record containing the given fields."""
    record = {field: f"value_{i}" for i, field in enumerate(fields)}
    with open(path, "w") as f:
        f.write(json.dumps(record) + "\n")


def _create_table_with_columns(url: str, table_name: str, columns: list[str]) -> None:
    """Create a database table with the given column names (all as String type)."""
    engine = create_engine(url)
    metadata = MetaData()
    Table(
        table_name,
        metadata,
        *[Column(col, String) for col in columns],
    )
    metadata.create_all(engine)
    engine.dispose()


@pytest.fixture
def csv_sink_factory(tmp_path: Path) -> SinkFactory:
    """Factory for creating CSVSink instances with pre-populated headers."""
    csv_path = tmp_path / "output.csv"

    def factory(target_fields: list[str], schema_config: dict[str, Any]) -> SinkProtocol:
        if target_fields:
            _create_csv_with_headers(csv_path, target_fields)
        config = {"path": str(csv_path), "schema": schema_config}
        return CSVSink(config)

    return factory


@pytest.fixture
def json_sink_factory(tmp_path: Path) -> SinkFactory:
    """Factory for creating JSONSink instances with pre-populated records."""
    jsonl_path = tmp_path / "output.jsonl"

    def factory(target_fields: list[str], schema_config: dict[str, Any]) -> SinkProtocol:
        if target_fields:
            _create_jsonl_with_record(jsonl_path, target_fields)
        config = {"path": str(jsonl_path), "schema": schema_config, "format": "jsonl"}
        return JSONSink(config)

    return factory


@pytest.fixture
def database_sink_factory(tmp_path: Path) -> SinkFactory:
    """Factory for creating DatabaseSink instances with pre-populated tables."""
    db_url = f"sqlite:///{tmp_path / 'test.db'}"
    table_name = "output_data"

    def factory(target_fields: list[str], schema_config: dict[str, Any]) -> SinkProtocol:
        if target_fields:
            _create_table_with_columns(db_url, table_name, target_fields)
        config = {"url": db_url, "table": table_name, "schema": schema_config}
        return DatabaseSink(config)

    return factory


@pytest.fixture(
    params=["csv", "json", "database"],
    ids=["CSVSink", "JSONSink", "DatabaseSink"],
)
def sink_factory(
    request: pytest.FixtureRequest,
    csv_sink_factory: SinkFactory,
    json_sink_factory: SinkFactory,
    database_sink_factory: SinkFactory,
) -> SinkFactory:
    """Parametrized fixture that provides factories for all sink types."""
    factories = {
        "csv": csv_sink_factory,
        "json": json_sink_factory,
        "database": database_sink_factory,
    }
    return factories[request.param]


@pytest.fixture
def json_only_sink_factory(json_sink_factory: SinkFactory) -> SinkFactory:
    """Fixture for tests that only apply to JSONSink (dynamic/free schemas).

    CSVSink and DatabaseSink reject dynamic/free schemas at initialization
    because they require fixed-column structure.
    """
    return json_sink_factory


# =============================================================================
# Common Schema Validation Tests
# =============================================================================


class TestSinkSchemaValidationCommon:
    """Tests for schema validation behavior common to all sink types."""

    # -------------------------------------------------------------------------
    # Nonexistent Target Tests
    # -------------------------------------------------------------------------

    def test_validate_nonexistent_target_returns_success(self, sink_factory: SinkFactory):
        """When target doesn't exist, validation should pass (will create)."""
        sink = sink_factory(
            target_fields=[],  # Empty = don't create target
            schema_config={"mode": "strict", "fields": ["id: int", "name: str"]},
        )
        try:
            result = sink.validate_output_target()
            assert result.valid is True
        finally:
            sink.close()

    # -------------------------------------------------------------------------
    # Dynamic Schema Tests (JSONSink only - CSV/Database reject dynamic schemas)
    # -------------------------------------------------------------------------

    def test_validate_dynamic_schema_skips_validation(self, json_only_sink_factory: SinkFactory):
        """Dynamic schema should always pass validation (JSONSink only).

        Note: CSVSink and DatabaseSink reject dynamic schemas at initialization
        because they require fixed-column structure.
        """
        sink = json_only_sink_factory(
            target_fields=["wrong", "fields", "entirely"],
            schema_config={"fields": "dynamic"},
        )
        try:
            result = sink.validate_output_target()
            assert result.valid is True
            # Dynamic schema should report actual target fields
            assert set(result.target_fields) == {"wrong", "fields", "entirely"}
        finally:
            sink.close()

    # -------------------------------------------------------------------------
    # Strict Mode Tests
    # -------------------------------------------------------------------------

    def test_validate_strict_mode_exact_match(self, sink_factory: SinkFactory):
        """Strict mode should pass when fields match exactly."""
        sink = sink_factory(
            target_fields=["id", "name"],
            schema_config={"mode": "strict", "fields": ["id: int", "name: str"]},
        )
        try:
            result = sink.validate_output_target()
            assert result.valid is True
        finally:
            sink.close()

    def test_validate_strict_mode_missing_field(self, sink_factory: SinkFactory):
        """Strict mode should fail when schema field is missing from target."""
        sink = sink_factory(
            target_fields=["id"],  # Missing 'name'
            schema_config={"mode": "strict", "fields": ["id: int", "name: str"]},
        )
        try:
            result = sink.validate_output_target()
            assert result.valid is False
            assert "strict mode" in result.error_message
            assert "name" in result.missing_fields
        finally:
            sink.close()

    def test_validate_strict_mode_extra_field(self, sink_factory: SinkFactory):
        """Strict mode should fail when target has extra field."""
        sink = sink_factory(
            target_fields=["id", "name", "extra"],
            schema_config={"mode": "strict", "fields": ["id: int", "name: str"]},
        )
        try:
            result = sink.validate_output_target()
            assert result.valid is False
            assert "strict mode" in result.error_message
            assert "extra" in result.extra_fields
        finally:
            sink.close()

    # -------------------------------------------------------------------------
    # Free Mode Tests (JSONSink only - CSV/Database reject free schemas)
    # -------------------------------------------------------------------------

    def test_validate_free_mode_exact_match(self, json_only_sink_factory: SinkFactory):
        """Free mode should pass when fields match exactly (JSONSink only).

        Note: CSVSink and DatabaseSink reject free schemas at initialization
        because they require fixed-column structure.
        """
        sink = json_only_sink_factory(
            target_fields=["id", "name"],
            schema_config={"mode": "free", "fields": ["id: int", "name: str"]},
        )
        try:
            result = sink.validate_output_target()
            assert result.valid is True
        finally:
            sink.close()

    def test_validate_free_mode_missing_field(self, json_only_sink_factory: SinkFactory):
        """Free mode should fail when required schema field is missing (JSONSink only)."""
        sink = json_only_sink_factory(
            target_fields=["id"],  # Missing 'name'
            schema_config={"mode": "free", "fields": ["id: int", "name: str"]},
        )
        try:
            result = sink.validate_output_target()
            assert result.valid is False
            assert "free mode" in result.error_message
            assert "name" in result.missing_fields
        finally:
            sink.close()

    def test_validate_free_mode_extra_field_allowed(self, json_only_sink_factory: SinkFactory):
        """Free mode should pass when target has extra fields (JSONSink only)."""
        sink = json_only_sink_factory(
            target_fields=["id", "name", "extra", "another"],
            schema_config={"mode": "free", "fields": ["id: int", "name: str"]},
        )
        try:
            result = sink.validate_output_target()
            assert result.valid is True
        finally:
            sink.close()


# =============================================================================
# Sink-Specific Tests (behavior differs between sink types)
# =============================================================================


class TestCSVSinkOrderValidation:
    """CSV-specific: strict mode validates column order."""

    @pytest.fixture
    def tmp_csv_path(self, tmp_path: Path) -> Path:
        """Return a temporary CSV file path."""
        return tmp_path / "output.csv"

    def test_validate_strict_mode_order_mismatch(self, tmp_csv_path: Path):
        """Strict mode should fail when same fields but different order."""
        _create_csv_with_headers(tmp_csv_path, ["name", "id"])  # Reversed order
        sink = CSVSink(
            {
                "path": str(tmp_csv_path),
                "schema": {"mode": "strict", "fields": ["id: int", "name: str"]},
            }
        )

        result = sink.validate_output_target()

        assert result.valid is False
        assert "strict mode" in result.error_message
        assert result.order_mismatch is True
        # No missing or extra fields - just order wrong
        assert len(result.missing_fields) == 0
        assert len(result.extra_fields) == 0

    def test_validate_empty_file_returns_success(self, tmp_csv_path: Path):
        """When file exists but is empty, validation should pass."""
        tmp_csv_path.write_text("")
        sink = CSVSink(
            {
                "path": str(tmp_csv_path),
                "schema": {"mode": "strict", "fields": ["id: int", "name: str"]},
            }
        )

        result = sink.validate_output_target()

        assert result.valid is True

    def test_validate_with_custom_delimiter(self, tmp_csv_path: Path):
        """Validation should respect custom delimiter."""
        # Create CSV with tab delimiter
        with open(tmp_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "name"], delimiter="\t")
            writer.writeheader()

        sink = CSVSink(
            {
                "path": str(tmp_csv_path),
                "schema": {"mode": "strict", "fields": ["id: int", "name: str"]},
                "delimiter": "\t",
            }
        )

        result = sink.validate_output_target()

        assert result.valid is True


class TestJSONSinkSpecific:
    """JSON-specific schema validation tests."""

    @pytest.fixture
    def tmp_jsonl_path(self, tmp_path: Path) -> Path:
        """Return a temporary JSONL file path."""
        return tmp_path / "output.jsonl"

    @pytest.fixture
    def tmp_json_path(self, tmp_path: Path) -> Path:
        """Return a temporary JSON file path."""
        return tmp_path / "output.json"

    def test_validate_empty_file_returns_success(self, tmp_jsonl_path: Path):
        """When file exists but is empty, validation should pass."""
        tmp_jsonl_path.write_text("")
        sink = JSONSink(
            {
                "path": str(tmp_jsonl_path),
                "schema": {"mode": "strict", "fields": ["id: int", "name: str"]},
                "format": "jsonl",
            }
        )

        result = sink.validate_output_target()

        assert result.valid is True

    def test_validate_json_array_format_skips_validation(self, tmp_json_path: Path):
        """JSON array format always passes validation (it rewrites entirely)."""
        # Create a JSON array file with wrong structure
        with open(tmp_json_path, "w") as f:
            json.dump([{"wrong": "fields"}], f)

        sink = JSONSink(
            {
                "path": str(tmp_json_path),
                "schema": {"mode": "strict", "fields": ["id: int", "name: str"]},
                "format": "json",
            }
        )

        result = sink.validate_output_target()

        # JSON array format doesn't need validation - it overwrites
        assert result.valid is True

    def test_validate_invalid_json_returns_failure(self, tmp_jsonl_path: Path):
        """Invalid JSON in file should return failure."""
        tmp_jsonl_path.write_text("not valid json\n")
        sink = JSONSink(
            {
                "path": str(tmp_jsonl_path),
                "schema": {"mode": "strict", "fields": ["id: int", "name: str"]},
                "format": "jsonl",
            }
        )

        result = sink.validate_output_target()

        assert result.valid is False
        assert "invalid JSON" in result.error_message

    def test_validate_non_object_record_returns_failure(self, tmp_jsonl_path: Path):
        """JSONL with non-object records should return failure."""
        tmp_jsonl_path.write_text("[1, 2, 3]\n")  # Array instead of object
        sink = JSONSink(
            {
                "path": str(tmp_jsonl_path),
                "schema": {"mode": "strict", "fields": ["id: int", "name: str"]},
                "format": "jsonl",
            }
        )

        result = sink.validate_output_target()

        assert result.valid is False
        assert "non-object" in result.error_message

    def test_validate_auto_detected_jsonl_format(self, tmp_path: Path):
        """Auto-detected JSONL format should validate correctly."""
        jsonl_path = tmp_path / "output.jsonl"
        with open(jsonl_path, "w") as f:
            f.write(json.dumps({"id": 1, "name": "test"}) + "\n")

        sink = JSONSink(
            {
                "path": str(jsonl_path),
                "schema": {"mode": "strict", "fields": ["id: int", "name: str"]},
                # format not specified - auto-detect from extension
            }
        )

        result = sink.validate_output_target()

        assert result.valid is True

    def test_validate_auto_detected_json_format_skips(self, tmp_path: Path):
        """Auto-detected JSON format should skip validation."""
        json_path = tmp_path / "output.json"
        with open(json_path, "w") as f:
            json.dump([{"wrong": "structure"}], f)

        sink = JSONSink(
            {
                "path": str(json_path),
                "schema": {"mode": "strict", "fields": ["id: int", "name: str"]},
                # format not specified - auto-detect from extension
            }
        )

        result = sink.validate_output_target()

        # JSON array format doesn't validate (it rewrites)
        assert result.valid is True


class TestDatabaseSinkSpecific:
    """Database-specific schema validation tests."""

    @pytest.fixture
    def sqlite_path(self, tmp_path: Path) -> str:
        """Return a SQLite database URL."""
        return f"sqlite:///{tmp_path / 'test.db'}"

    def test_validate_strict_mode_order_independent(self, sqlite_path: str):
        """Strict mode for databases is order-independent (set comparison)."""
        # Create with different order - should still pass for database
        _create_table_with_columns(sqlite_path, "output_data", ["name", "id"])
        sink = DatabaseSink(
            {
                "url": sqlite_path,
                "table": "output_data",
                "schema": {"mode": "strict", "fields": ["id: int", "name: str"]},
            }
        )

        result = sink.validate_output_target()

        # Database strict mode uses set comparison (unlike CSV)
        assert result.valid is True
        sink.close()
