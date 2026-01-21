# src/elspeth/plugins/sinks/database_sink.py
"""Database sink plugin for ELSPETH.

Writes rows to a database table using SQLAlchemy Core.

IMPORTANT: Sinks use allow_coercion=False to enforce that transforms
output correct types. Wrong types = upstream bug = crash.
"""

import hashlib
import json
from typing import Any, Literal

from sqlalchemy import Boolean, Column, Float, Integer, MetaData, String, Table, create_engine, insert
from sqlalchemy.engine import Engine
from sqlalchemy.types import TypeEngine

from elspeth.contracts import ArtifactDescriptor, PluginSchema
from elspeth.plugins.base import BaseSink
from elspeth.plugins.config_base import DataPluginConfig
from elspeth.plugins.context import PluginContext
from elspeth.plugins.schema_factory import create_schema_from_config

# Map schema field types to SQLAlchemy column types
SCHEMA_TYPE_TO_SQLALCHEMY: dict[str, type[TypeEngine[Any]]] = {
    "str": String,
    "int": Integer,
    "float": Float,
    "bool": Boolean,
    "any": String,  # Fallback to String for 'any' type
}


class DatabaseSinkConfig(DataPluginConfig):
    """Configuration for database sink plugin.

    Inherits from DataPluginConfig, which requires schema configuration.
    """

    url: str
    table: str
    if_exists: Literal["append", "replace"] = "append"
    validate_input: bool = False  # Optional runtime validation of incoming rows


class DatabaseSink(BaseSink):
    """Write rows to a database table.

    Creates the table on first write. When schema is explicit, columns are
    derived from schema field definitions with proper type mapping. When schema
    is dynamic, columns are inferred from the first row's keys.

    Uses SQLAlchemy Core for direct SQL control.

    Returns ArtifactDescriptor with SHA-256 hash of canonical JSON payload
    BEFORE insert. This proves intent - the database may transform data.

    Config options:
        url: Database connection URL (required)
        table: Table name (required)
        schema: Schema configuration (required, via DataPluginConfig)
        if_exists: "append" or "replace" (default: "append")
        validate_input: Validate incoming rows against schema (default: False)

    The schema can be:
        - Dynamic: {"fields": "dynamic"} - accept any fields (columns inferred from first row)
        - Strict: {"mode": "strict", "fields": ["id: int", "name: str"]} - columns from schema
        - Free: {"mode": "free", "fields": ["id: int"]} - columns from schema, extras allowed
    """

    name = "database"
    plugin_version = "1.0.0"
    # determinism inherited from BaseSink (IO_WRITE)

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        cfg = DatabaseSinkConfig.from_dict(config)

        self._url = cfg.url
        self._table_name = cfg.table
        self._if_exists = cfg.if_exists
        self._validate_input = cfg.validate_input

        # Store schema config for audit trail
        # DataPluginConfig ensures schema_config is not None
        assert cfg.schema_config is not None
        self._schema_config = cfg.schema_config

        # CRITICAL: allow_coercion=False - wrong types are bugs, not data to fix
        # Sinks receive PIPELINE DATA (already validated by source)
        self._schema_class: type[PluginSchema] = create_schema_from_config(
            self._schema_config,
            "DatabaseRowSchema",
            allow_coercion=False,  # Sinks reject wrong types (upstream bug)
        )

        # Set input_schema for protocol compliance
        self.input_schema = self._schema_class

        self._engine: Engine | None = None
        self._table: Table | None = None
        self._metadata: MetaData | None = None

    def _ensure_table(self, row: dict[str, Any]) -> None:
        """Create table if it doesn't exist.

        When schema is explicit (not dynamic), columns are derived from schema
        fields with proper type mapping. This ensures all defined fields
        (including optional ones) are present in the table.

        When schema is dynamic, columns are inferred from the first row's keys.
        """
        if self._engine is None:
            self._engine = create_engine(self._url)
            self._metadata = MetaData()

        if self._table is None:
            columns = self._create_columns_from_schema_or_row(row)
            # Metadata is always set when engine is created
            assert self._metadata is not None
            self._table = Table(
                self._table_name,
                self._metadata,
                *columns,
            )
            self._metadata.create_all(self._engine, checkfirst=True)

    def _create_columns_from_schema_or_row(self, row: dict[str, Any]) -> list[Column[Any]]:
        """Create SQLAlchemy columns from schema or row keys.

        When schema is explicit, creates columns for ALL defined fields with
        proper type mapping. This ensures optional fields are present.

        When schema is dynamic, falls back to inferring from row keys.
        """
        if not self._schema_config.is_dynamic and self._schema_config.fields:
            # Explicit schema: use field definitions with proper types
            columns: list[Column[Any]] = []
            for field_def in self._schema_config.fields:
                sql_type = SCHEMA_TYPE_TO_SQLALCHEMY[field_def.field_type]
                # Note: nullable=True for optional fields, but SQLAlchemy Column
                # defaults to nullable=True anyway, so we don't need to set it
                columns.append(Column(field_def.name, sql_type))
            return columns
        else:
            # Dynamic schema: infer from row keys (all as String)
            return [Column(key, String) for key in row]

    def write(self, rows: list[dict[str, Any]], ctx: PluginContext) -> ArtifactDescriptor:
        """Write a batch of rows to the database.

        CRITICAL: Hashes the canonical JSON payload BEFORE insert.
        This proves intent - the database may transform data (add timestamps,
        auto-increment IDs, normalize strings, etc.).

        Args:
            rows: List of row dicts to write
            ctx: Plugin context

        Returns:
            ArtifactDescriptor with content_hash (SHA-256) and size_bytes

        Raises:
            ValidationError: If validate_input=True and a row fails validation.
                This indicates a bug in an upstream transform.
        """
        # Compute canonical JSON payload BEFORE any database operation
        payload_json = json.dumps(rows, sort_keys=True, separators=(",", ":"))
        payload_bytes = payload_json.encode("utf-8")
        content_hash = hashlib.sha256(payload_bytes).hexdigest()
        payload_size = len(payload_bytes)

        if not rows:
            # Empty batch - return descriptor without DB operations
            return ArtifactDescriptor.for_database(
                url=self._url,
                table=self._table_name,
                content_hash=content_hash,
                payload_size=0,
                row_count=0,
            )

        # Optional input validation - crash on failure (upstream bug!)
        if self._validate_input and not self._schema_config.is_dynamic:
            for row in rows:
                # Raises ValidationError on failure - this is intentional
                self._schema_class.model_validate(row)

        # Ensure table exists (infer from first row)
        self._ensure_table(rows[0])

        # Insert all rows in batch
        if self._engine is not None and self._table is not None:
            with self._engine.begin() as conn:
                conn.execute(insert(self._table), rows)

        return ArtifactDescriptor.for_database(
            url=self._url,
            table=self._table_name,
            content_hash=content_hash,
            payload_size=payload_size,
            row_count=len(rows),
        )

    def flush(self) -> None:
        """Flush any pending operations.

        No-op for DatabaseSink - writes are immediate.
        """

    def close(self) -> None:
        """Close database connection."""
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None
            self._table = None
            self._metadata = None

    # === Lifecycle Hooks ===

    def on_start(self, ctx: PluginContext) -> None:
        """Called before processing begins."""
        pass

    def on_complete(self, ctx: PluginContext) -> None:
        """Called after processing completes."""
        pass
