# src/elspeth/plugins/sinks/database_sink.py
"""Database sink plugin for ELSPETH.

Writes rows to a database table using SQLAlchemy Core.

IMPORTANT: Sinks use allow_coercion=False to enforce that transforms
output correct types. Wrong types = upstream bug = crash.
"""

import hashlib
import os
import time
from typing import TYPE_CHECKING, Any, Literal

from sqlalchemy import Boolean, Column, Float, Integer, MetaData, Table, Text, create_engine, insert

if TYPE_CHECKING:
    from elspeth.contracts.sink import OutputValidationResult
from sqlalchemy.engine import Engine
from sqlalchemy.types import TypeEngine

from elspeth.contracts import ArtifactDescriptor, CallStatus, CallType, PluginSchema
from elspeth.contracts.plugin_context import PluginContext
from elspeth.contracts.url import SanitizedDatabaseUrl
from elspeth.core.canonical import canonical_json
from elspeth.plugins.base import BaseSink
from elspeth.plugins.config_base import DataPluginConfig
from elspeth.plugins.schema_factory import create_schema_from_config

# Map schema field types to SQLAlchemy column types.
# Text (not String) is used for string columns because String() without a length
# argument causes truncation or errors on MySQL/MSSQL — Text maps to TEXT on all
# backends and accepts arbitrary-length values without portability issues.
SCHEMA_TYPE_TO_SQLALCHEMY: dict[str, type[TypeEngine[Any]]] = {
    "str": Text,
    "int": Integer,
    "float": Float,
    "bool": Boolean,
    "any": Text,  # Fallback to Text for 'any' type
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
        - Observed: {"mode": "observed"} - accept any fields (columns inferred from first row)
        - Fixed: {"mode": "fixed", "fields": ["id: int", "name: str"]} - columns from schema
        - Flexible: {"mode": "flexible", "fields": ["id: int"]} - columns from schema, extras allowed
    """

    name = "database"
    plugin_version = "1.0.0"
    # determinism inherited from BaseSink (IO_WRITE)

    # Resume capability: Database can append to existing tables
    supports_resume: bool = True

    def configure_for_resume(self) -> None:
        """Configure database sink for resume mode.

        Switches from replace mode to append mode so resume operations
        add to existing table instead of dropping and recreating.
        """
        self._if_exists = "append"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        cfg = DatabaseSinkConfig.from_dict(config)

        # Honor ELSPETH_ALLOW_RAW_SECRETS for dev environments (consistent with config.py)
        allow_raw = os.environ.get("ELSPETH_ALLOW_RAW_SECRETS", "").lower() == "true"
        fail_if_no_key = not allow_raw

        self._url = cfg.url  # Raw URL for database connection
        self._sanitized_url = SanitizedDatabaseUrl.from_raw_url(cfg.url, fail_if_no_key=fail_if_no_key)  # For audit trail
        self._table_name = cfg.table
        self._if_exists = cfg.if_exists
        self._validate_input = cfg.validate_input

        # Store schema config for audit trail
        # DataPluginConfig ensures schema_config is not None
        self._schema_config = cfg.schema_config

        # Database supports all schema modes via infer-and-lock:
        # - mode='fixed': columns from config, extras rejected at insert time
        # - mode='flexible': declared columns + extras from first row, then locked
        # - mode='observed': columns from first row, then locked
        #
        # Table schema is created on first write; subsequent rows must match.

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
        self._table_replaced: bool = False  # Track if we've done the replace for this instance

    def _ensure_engine_and_metadata_initialized(self) -> None:
        """Initialize engine/metadata pair together.

        Invariant: if self._engine is set, self._metadata must also be set.
        This keeps validate_output_target() and write() lifecycle paths consistent.
        """
        if self._engine is None:
            self._engine = create_engine(self._url)
        if self._metadata is None:
            self._metadata = MetaData()

    def validate_output_target(self) -> "OutputValidationResult":
        """Validate existing database table columns against configured schema.

        Checks that:
        - Strict mode: Table columns match schema fields exactly (set comparison)
        - Free mode: All schema fields present as columns (extras allowed)
        - Dynamic mode: No validation (schema adapts to existing columns)

        Note: Unlike CSV, column order is not validated for databases.

        Returns:
            OutputValidationResult indicating compatibility.
        """
        from sqlalchemy import inspect

        from elspeth.contracts.sink import OutputValidationResult

        # Ensure engine/metadata are initialized consistently before inspection.
        self._ensure_engine_and_metadata_initialized()
        if self._engine is None:
            raise RuntimeError("Database sink validation called before initialization")

        inspector = inspect(self._engine)
        if not inspector.has_table(self._table_name):
            return OutputValidationResult.success()  # Will create table

        # Get existing columns
        columns = inspector.get_columns(self._table_name)
        existing = [col["name"] for col in columns]

        # Dynamic schema = no validation needed
        if self._schema_config.is_observed:
            return OutputValidationResult.success(target_fields=existing)

        # Get expected fields from schema (guaranteed non-None when not dynamic)
        fields = self._schema_config.fields
        if fields is None:
            return OutputValidationResult.success(target_fields=existing)
        expected = [f.name for f in fields]
        existing_set, expected_set = set(existing), set(expected)

        if self._schema_config.mode == "fixed":
            # Fixed: exact column match (set comparison, no order)
            if existing_set != expected_set:
                return OutputValidationResult.failure(
                    message="Table columns do not match schema (fixed mode)",
                    target_fields=existing,
                    schema_fields=expected,
                    missing_fields=sorted(expected_set - existing_set),
                    extra_fields=sorted(existing_set - expected_set),
                )
        else:  # mode == "flexible"
            # Flexible: schema fields must exist as columns (extras allowed)
            missing = expected_set - existing_set
            if missing:
                return OutputValidationResult.failure(
                    message="Table missing required schema columns (flexible mode)",
                    target_fields=existing,
                    schema_fields=expected,
                    missing_fields=sorted(missing),
                )

        return OutputValidationResult.success(target_fields=existing)

    def _ensure_table(self, row: dict[str, Any]) -> None:
        """Create table, handling if_exists behavior.

        if_exists behavior (follows pandas to_sql semantics):
        - "append": Create table if not exists, insert rows (default)
        - "replace": Drop table on first write, recreate with fresh schema

        When schema is explicit (not dynamic), columns are derived from schema
        fields with proper type mapping. This ensures all defined fields
        (including optional ones) are present in the table.

        When schema is dynamic, columns are inferred from the first row's keys.
        """
        self._ensure_engine_and_metadata_initialized()
        if self._engine is None:
            raise RuntimeError("Database sink write() called before initialization")

        if self._table is None:
            # Handle if_exists="replace": drop table on first write
            if self._if_exists == "replace" and not self._table_replaced:
                self._drop_table_if_exists()
                self._table_replaced = True

            columns = self._create_columns_from_schema_or_row(row)
            # Metadata is always set when engine is created
            if self._metadata is None:
                raise RuntimeError("Database sink write() called before initialization")
            self._table = Table(
                self._table_name,
                self._metadata,
                *columns,
            )
            self._metadata.create_all(self._engine, checkfirst=True)

    def _drop_table_if_exists(self) -> None:
        """Drop the table if it exists (for replace mode).

        Uses SQLAlchemy's Table.drop() for portable, dialect-safe drops.
        This handles identifier quoting correctly across all databases
        (SQLite, PostgreSQL, MySQL, etc.).
        """
        if self._engine is None:
            return

        from sqlalchemy import MetaData, Table, inspect

        inspector = inspect(self._engine)
        if inspector.has_table(self._table_name):
            # Use SQLAlchemy's Table.drop() for dialect-safe drop
            # This generates correct identifier quoting for any database
            temp_metadata = MetaData()
            table = Table(self._table_name, temp_metadata)
            table.drop(self._engine)

    def _create_columns_from_schema_or_row(self, row: dict[str, Any]) -> list[Column[Any]]:
        """Create SQLAlchemy columns from schema or row keys.

        Column creation depends on schema mode:
        - fixed: Only declared fields with proper types
        - flexible: Declared fields with proper types, then extras as Text
        - observed: All fields from first row as Text (infer and lock)
        """
        if self._schema_config.is_observed:
            # Observed mode: infer from row keys (all as Text for portability)
            return [Column(key, Text) for key in row]

        if self._schema_config.fields:
            # Explicit schema: start with declared fields and their types
            columns: list[Column[Any]] = []
            declared_names: set[str] = set()

            for field_def in self._schema_config.fields:
                sql_type = SCHEMA_TYPE_TO_SQLALCHEMY[field_def.field_type]
                # Note: nullable=True for optional fields, but SQLAlchemy Column
                # defaults to nullable=True anyway, so we don't need to set it
                columns.append(Column(field_def.name, sql_type))
                declared_names.add(field_def.name)

            if self._schema_config.mode == "flexible":
                # Flexible mode: add extra columns from row as Text type
                for key in row:
                    if key not in declared_names:
                        columns.append(Column(key, Text))

            return columns

        # Fallback (shouldn't happen with valid config): use row keys
        return [Column(key, Text) for key in row]

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
        # Compute canonical JSON payload ONCE before any database operation.
        # Uses RFC 8785 canonical JSON for deterministic hashing:
        # - Normalizes pandas/numpy types to JSON primitives
        # - Rejects NaN/Infinity (invalid JSON per RFC 8785)
        # - Deterministic unicode escaping
        canonical_payload = canonical_json(rows).encode("utf-8")
        content_hash = hashlib.sha256(canonical_payload).hexdigest()
        payload_size = len(canonical_payload)

        if not rows:
            # Empty batch - return descriptor without DB operations
            return ArtifactDescriptor.for_database(
                url=self._sanitized_url,
                table=self._table_name,
                content_hash=content_hash,
                payload_size=payload_size,
                row_count=0,
            )

        # Optional input validation - crash on failure (upstream bug!)
        if self._validate_input and not self._schema_config.is_observed:
            for row in rows:
                # Raises ValidationError on failure - this is intentional
                self._schema_class.model_validate(row)

        # Ensure table exists (infer from first row)
        self._ensure_table(rows[0])

        # Insert all rows in batch with call recording for audit trail
        # (ctx.operation_id is set by executor)
        start_time = time.perf_counter()
        try:
            if self._engine is not None and self._table is not None:
                with self._engine.begin() as conn:
                    conn.execute(insert(self._table), rows)
            latency_ms = (time.perf_counter() - start_time) * 1000

            # Record successful INSERT in audit trail
            ctx.record_call(
                call_type=CallType.SQL,
                status=CallStatus.SUCCESS,
                request_data={
                    "operation": "INSERT",
                    "table": self._table_name,
                    "row_count": len(rows),
                },
                response_data={"rows_inserted": len(rows)},
                latency_ms=latency_ms,
                provider="sqlalchemy",
            )
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000

            # Record failed INSERT in audit trail
            ctx.record_call(
                call_type=CallType.SQL,
                status=CallStatus.ERROR,
                request_data={
                    "operation": "INSERT",
                    "table": self._table_name,
                    "row_count": len(rows),
                },
                error={"type": type(e).__name__, "message": str(e)},
                latency_ms=latency_ms,
                provider="sqlalchemy",
            )
            raise

        return ArtifactDescriptor.for_database(
            url=self._sanitized_url,
            table=self._table_name,
            content_hash=content_hash,
            payload_size=payload_size,
            row_count=len(rows),
        )

    def flush(self) -> None:
        """Flush any pending operations.

        No-op for DatabaseSink - durability is guaranteed by auto-commit in write().

        DatabaseSink uses `engine.begin()` context manager which commits the
        transaction when write() returns. This provides the same durability
        guarantee as an explicit flush() - all data is committed to the database
        before this method is called.

        Future enhancement: Hold transaction open between write() and flush()
        for explicit two-phase durability control.
        """

    def close(self) -> None:
        """Close database connection."""
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None
            self._table = None
            self._metadata = None
            self._table_replaced = False

    # === Lifecycle Hooks ===

    def on_start(self, ctx: PluginContext) -> None:
        """Called before processing begins."""
        pass

    def on_complete(self, ctx: PluginContext) -> None:
        """Called after processing completes."""
        pass
