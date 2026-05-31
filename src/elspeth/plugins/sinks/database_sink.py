"""Database sink plugin for ELSPETH.

Writes rows to a database table using SQLAlchemy Core.

IMPORTANT: Sinks use allow_coercion=False to enforce that transforms
output correct types. Wrong types = upstream bug = crash.
"""

import hashlib
import json
import os
import time
from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import Field, field_validator
from sqlalchemy import Boolean, Column, Float, Integer, MetaData, Table, Text, create_engine, insert
from sqlalchemy.exc import IntegrityError

if TYPE_CHECKING:
    from elspeth.contracts.sink import OutputValidationResult
from sqlalchemy.engine import Engine
from sqlalchemy.types import TypeEngine

from elspeth.contracts import ArtifactDescriptor, CallStatus, CallType, Determinism, PluginSchema
from elspeth.contracts.contexts import SinkContext
from elspeth.contracts.diversion import SinkWriteResult
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.plugin_assistance import PluginAssistance
from elspeth.contracts.url import SanitizedDatabaseUrl
from elspeth.contracts.wire_visible_identity import reject_placeholder_value
from elspeth.core.canonical import canonical_json
from elspeth.plugins.infrastructure.base import BaseSink
from elspeth.plugins.infrastructure.config_base import DataPluginConfig
from elspeth.plugins.infrastructure.schema_factory import create_schema_from_config

# Map schema field types to SQLAlchemy column types.
# Text (not String) is used for string columns because String() without a length
# argument causes truncation or errors on MySQL/MSSQL — Text maps to TEXT on all
# backends and accepts arbitrary-length values without portability issues.
SCHEMA_TYPE_TO_SQLALCHEMY: Mapping[str, type[TypeEngine[Any]]] = MappingProxyType(
    {
        "str": Text,
        "int": Integer,
        "float": Float,
        "bool": Boolean,
        "any": Text,  # Fallback to Text for 'any' type
    }
)


class DatabaseSinkConfig(DataPluginConfig):
    """Configuration for database sink plugin.

    Inherits from DataPluginConfig, which requires schema configuration.
    """

    _plugin_component_type: ClassVar[str | None] = "sink"

    url: str = Field(description="Database connection URL for SQLAlchemy.")
    table: str = Field(description="Database table name to write rows into.")
    if_exists: Literal["append", "replace"] = Field(
        default="append",
        description="Whether to append to an existing table or replace it before writing.",
    )

    @field_validator("table")
    @classmethod
    def _reject_empty_table(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("table name must not be empty")
        return reject_placeholder_value(v, field_name="table")


class DatabaseSink(BaseSink):
    """Write rows to a database table.

    Creates the table on first write. When schema is explicit, columns are
    derived from schema field definitions with proper type mapping. When schema
    is dynamic, columns are inferred from the first row's keys.

    Uses SQLAlchemy Core for direct SQL control.

    Returns ArtifactDescriptor with a SHA-256 hash of the canonical JSON
    payload of the rows that were COMMITTED (after per-row diversion). The
    database may further transform data (timestamps, auto-increment IDs);
    the hash proves what was durably written. Rows diverted on a per-row
    constraint failure are NOT in this hash — they are audited via the
    diversion log and routed to the configured failsink.

    Config options:
        url: Database connection URL (required)
        table: Table name (required)
        schema: Schema configuration (required, via DataPluginConfig)
        if_exists: "append" or "replace" (default: "append")

    Input validation is always enabled. Incoming rows are validated against
    the schema before INSERT — wrong types indicate an upstream plugin bug
    and will crash the pipeline (Tier 2 contract).

    The schema can be:
        - Observed: {"mode": "observed"} - accept any fields (columns inferred from first row)
        - Fixed: {"mode": "fixed", "fields": ["id: int", "name: str"]} - columns from schema
        - Flexible: {"mode": "flexible", "fields": ["id: int"]} - columns from schema, extras allowed
    """

    name = "database"
    determinism = Determinism.IO_WRITE
    plugin_version = "1.0.0"
    source_file_hash: str | None = "sha256:daed354d4afac650"
    config_model = DatabaseSinkConfig
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
        cfg = DatabaseSinkConfig.from_dict(config, plugin_name=self.name)

        # Honor ELSPETH_ALLOW_RAW_SECRETS for dev environments (consistent with config.py)
        allow_raw = os.environ.get("ELSPETH_ALLOW_RAW_SECRETS", "").lower() == "true"
        fail_if_no_key = not allow_raw

        self._url = cfg.url  # Raw URL for database connection
        self._sanitized_url = SanitizedDatabaseUrl.from_raw_url(cfg.url, fail_if_no_key=fail_if_no_key)  # For audit trail
        self._table_name = cfg.table
        self._if_exists = cfg.if_exists

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

        # Required-field enforcement (centralized in SinkExecutor)
        self.declared_required_fields = self._schema_config.get_effective_required_fields()

        # Track which fields have 'any' type so we can serialize dict/list values
        # to JSON strings before INSERT (SQL TEXT columns can't store Python dicts).
        self._any_typed_fields: frozenset[str] = self._compute_any_typed_fields()

        self._engine: Engine | None = None
        self._table: Table | None = None
        self._metadata: MetaData | None = None
        self._table_replaced: bool = False  # Track if we've done the replace for this instance

    def _compute_any_typed_fields(self) -> frozenset[str]:
        """Identify fields with 'any' type from the schema config.

        These fields may contain dict/list values that must be serialized
        to JSON strings before INSERT into TEXT columns.
        """
        if self._schema_config.is_observed or not self._schema_config.fields:
            return frozenset()
        return frozenset(f.name for f in self._schema_config.fields if f.field_type == "any")

    def _serialize_any_typed_fields(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Serialize dict/list values in 'any'-typed fields to JSON strings.

        SQL TEXT columns cannot store Python dicts or lists. This method
        converts non-scalar values to their JSON string representation
        before INSERT, ensuring valid 'any' payloads (e.g., {"k": 1})
        are stored as '{"k": 1}' rather than crashing with a driver error.

        For observed-mode schemas, ALL fields are checked since any field
        could contain a complex value when the schema is inferred.

        Scalar values (str, int, float, bool, None) are left unchanged.
        """
        if not self._any_typed_fields and not self._schema_config.is_observed:
            return rows

        result = []
        for row in rows:
            new_row = dict(row)
            fields_to_check = self._any_typed_fields if self._any_typed_fields else set(row.keys())
            for field in fields_to_check:
                if field in new_row:
                    value = new_row[field]
                    if isinstance(value, (dict, list)):
                        new_row[field] = json.dumps(value)
            result.append(new_row)
        return result

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

    def _ensure_table(self, row: dict[str, Any], ctx: SinkContext) -> None:
        """Create table, handling if_exists behavior.

        if_exists behavior (follows pandas to_sql semantics):
        - "append": Create table if not exists, insert rows (default)
        - "replace": Drop table on first write, recreate with fresh schema

        When schema is explicit (not dynamic), columns are derived from schema
        fields with proper type mapping. This ensures all defined fields
        (including optional ones) are present in the table.

        When schema is dynamic, columns are inferred from the first row's keys.

        DDL operations (DROP TABLE, CREATE TABLE) are instrumented via
        ctx.record_call for audit trail completeness.
        """
        self._ensure_engine_and_metadata_initialized()
        if self._engine is None:
            raise RuntimeError("Database sink write() called before initialization")

        if self._table is None:
            # Handle if_exists="replace": drop table on first write
            if self._if_exists == "replace" and not self._table_replaced:
                self._drop_table_if_exists(ctx)
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

            # Instrument CREATE TABLE DDL for audit trail
            start_time = time.perf_counter()
            try:
                self._metadata.create_all(self._engine, checkfirst=True)
                latency_ms = (time.perf_counter() - start_time) * 1000
                try:
                    ctx.record_call(
                        call_type=CallType.SQL,
                        status=CallStatus.SUCCESS,
                        request_data={
                            "operation": "CREATE_TABLE",
                            "table": self._table_name,
                            "if_not_exists": True,
                        },
                        response_data={"table_created": self._table_name},
                        latency_ms=latency_ms,
                        provider="sqlalchemy",
                    )
                except Exception as exc:
                    raise AuditIntegrityError(
                        f"Failed to record successful CREATE TABLE to audit trail "
                        f"(table={self._table_name!r}). "
                        f"DDL completed but audit record is missing."
                    ) from exc
            except AuditIntegrityError:
                raise  # Audit failure — do not misattribute as SQL error
            except Exception as e:
                latency_ms = (time.perf_counter() - start_time) * 1000
                ctx.record_call(
                    call_type=CallType.SQL,
                    status=CallStatus.ERROR,
                    request_data={
                        "operation": "CREATE_TABLE",
                        "table": self._table_name,
                        "if_not_exists": True,
                    },
                    error={"type": type(e).__name__, "message": str(e)},
                    latency_ms=latency_ms,
                    provider="sqlalchemy",
                )
                raise

    def _drop_table_if_exists(self, ctx: SinkContext) -> None:
        """Drop the table if it exists (for replace mode).

        Uses SQLAlchemy's Table.drop() for portable, dialect-safe drops.
        This handles identifier quoting correctly across all databases
        (SQLite, PostgreSQL, MySQL, etc.).

        DDL is instrumented via ctx.record_call for audit trail completeness.
        """
        assert self._engine is not None, (
            "engine is None at DROP TABLE time — invariant violation (_ensure_engine_and_metadata_initialized must run first)"
        )

        from sqlalchemy import MetaData, Table, inspect

        inspector = inspect(self._engine)
        if inspector.has_table(self._table_name):
            # Use SQLAlchemy's Table.drop() for dialect-safe drop
            # This generates correct identifier quoting for any database
            temp_metadata = MetaData()
            table = Table(self._table_name, temp_metadata)

            start_time = time.perf_counter()
            try:
                table.drop(self._engine)
                latency_ms = (time.perf_counter() - start_time) * 1000
                try:
                    ctx.record_call(
                        call_type=CallType.SQL,
                        status=CallStatus.SUCCESS,
                        request_data={
                            "operation": "DROP_TABLE",
                            "table": self._table_name,
                            "mode": self._if_exists,
                        },
                        response_data={"table_dropped": self._table_name},
                        latency_ms=latency_ms,
                        provider="sqlalchemy",
                    )
                except Exception as exc:
                    raise AuditIntegrityError(
                        f"Failed to record successful DROP TABLE to audit trail "
                        f"(table={self._table_name!r}). "
                        f"DDL completed but audit record is missing."
                    ) from exc
            except AuditIntegrityError:
                raise  # Audit failure — do not misattribute as SQL error
            except Exception as e:
                latency_ms = (time.perf_counter() - start_time) * 1000
                ctx.record_call(
                    call_type=CallType.SQL,
                    status=CallStatus.ERROR,
                    request_data={
                        "operation": "DROP_TABLE",
                        "table": self._table_name,
                        "mode": self._if_exists,
                    },
                    error={"type": type(e).__name__, "message": str(e)},
                    latency_ms=latency_ms,
                    provider="sqlalchemy",
                )
                raise

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
                # Enforce nullable based on required status: required fields
                # are NOT NULL, optional fields are nullable.
                columns.append(Column(field_def.name, sql_type, nullable=not field_def.required))
                declared_names.add(field_def.name)

            if self._schema_config.mode == "flexible":
                # Flexible mode: add extra columns from row as Text type
                for key in row:
                    if key not in declared_names:
                        columns.append(Column(key, Text))

            return columns

        # Fallback (shouldn't happen with valid config): use row keys
        return [Column(key, Text) for key in row]

    def write(self, rows: list[dict[str, Any]], ctx: SinkContext) -> SinkWriteResult:
        """Write a batch of rows to the database.

        CRITICAL: Hashes the canonical JSON payload of the ACTUAL SQL rows
        that were COMMITTED (after any-field serialization and after per-row
        diversion). This proves what was durably written — the database may
        further transform data (add timestamps, auto-increment IDs, normalize
        strings, etc.). Rows diverted on a per-row constraint failure are not
        committed, so they are excluded from the hash and ``row_count``; they
        are recorded in the diversion log and returned in ``SinkWriteResult``.

        A per-row-attributable constraint violation (UNIQUE / NOT NULL / CHECK
        / foreign-key) on one row diverts that row and commits the rest (see
        ``_insert_with_per_row_diversion``). Batch-integrity failures
        (connection loss, lock timeout, bad SQL) are not row-attributable:
        they roll the whole transaction back, are recorded as an ERROR call,
        and re-raise.

        Args:
            rows: List of row dicts to write
            ctx: Plugin context

        Returns:
            SinkWriteResult with the committed-rows artifact and any per-row
            diversions.

        Raises:
            ValidationError: If a row fails schema validation.
                This indicates a bug in an upstream transform.
        """
        if not rows:
            # Empty batch - hash the empty list for consistent audit trail
            canonical_payload = canonical_json(rows).encode("utf-8")
            content_hash = hashlib.sha256(canonical_payload).hexdigest()
            return SinkWriteResult(
                artifact=ArtifactDescriptor.for_database(
                    url=self._sanitized_url,
                    table=self._table_name,
                    content_hash=content_hash,
                    payload_size=len(canonical_payload),
                    row_count=0,
                )
            )

        # Ensure table exists (infer from first row)
        self._ensure_table(rows[0], ctx)

        # Validate rows against table columns before INSERT.
        # SQLAlchemy silently drops keys not in the table schema, which
        # hides upstream bugs. In fixed mode, extra fields are rejected.
        # In all modes after table creation, unknown columns are rejected.
        if self._table is not None:
            known_columns = {c.name for c in self._table.columns}
            for i, row in enumerate(rows):
                extra = sorted(set(row) - known_columns)
                if extra:
                    raise ValueError(
                        f"DatabaseSink row {i} has fields not in table schema: {extra}. This indicates an upstream transform/schema bug."
                    )

        # Serialize dict/list values in 'any'-typed fields to JSON strings
        # before INSERT. SQL TEXT columns cannot store Python dicts/lists.
        # insert_rows is 1:1 with `rows`; rows[i] is the canonical pipeline row
        # handed to the failsink on diversion, insert_rows[i] is what we INSERT.
        insert_rows = self._serialize_any_typed_fields(rows)

        # Insert all rows in batch with call recording for audit trail
        # (ctx.operation_id is set by executor)
        assert self._engine is not None and self._table is not None, (
            "engine/table is None at INSERT time — invariant violation (_ensure_table must set both before write)"
        )
        start_time = time.perf_counter()
        try:
            written_rows = self._insert_with_per_row_diversion(rows, insert_rows)
            latency_ms = (time.perf_counter() - start_time) * 1000

            # Hash the ACTUAL SQL payload that was committed (post-serialization,
            # post-diversion) using RFC 8785 canonical JSON. This proves what was
            # written to the database — diverted rows were NOT written, so they
            # must not appear in the artifact hash or row_count, or the audit
            # trail would assert rows landed that were actually routed elsewhere.
            canonical_payload = canonical_json(written_rows).encode("utf-8")
            content_hash = hashlib.sha256(canonical_payload).hexdigest()
            payload_size = len(canonical_payload)
            rows_written = len(written_rows)

            # Record successful INSERT in audit trail.
            try:
                ctx.record_call(
                    call_type=CallType.SQL,
                    status=CallStatus.SUCCESS,
                    request_data={
                        "operation": "INSERT",
                        "table": self._table_name,
                        "row_count": rows_written,
                    },
                    response_data={"rows_inserted": rows_written},
                    latency_ms=latency_ms,
                    provider="sqlalchemy",
                )
            except Exception as exc:
                raise AuditIntegrityError(
                    f"Failed to record successful INSERT to audit trail "
                    f"(table={self._table_name!r}, row_count={rows_written}). "
                    f"INSERT completed but audit record is missing."
                ) from exc
        except AuditIntegrityError:
            raise  # Audit failure — do not misattribute as SQL error
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000

            # Record failed INSERT in audit trail. We reach here only for
            # batch-integrity failures (connection/operational/programming errors)
            # that are NOT per-row attributable: the outer transaction rolled back,
            # so nothing was committed and the recorded ERROR is accurate.
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

        return SinkWriteResult(
            artifact=ArtifactDescriptor.for_database(
                url=self._sanitized_url,
                table=self._table_name,
                content_hash=content_hash,
                payload_size=payload_size,
                row_count=rows_written,
            ),
            diversions=self._get_diversions(),
        )

    def _insert_with_per_row_diversion(
        self,
        rows: list[dict[str, Any]],
        insert_rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """INSERT the batch, diverting per-row-attributable constraint failures.

        Transaction strategy (exactly-once write-or-divert):

        The entire operation runs inside ONE outer ``engine.begin()``
        transaction. Nothing is durable until that outer block commits, so
        any exception that propagates out of it rolls back EVERYTHING.

        1. Fast path: attempt the whole batch inside a SAVEPOINT
           (``begin_nested()``). If it commits, every row is written — return
           them all. This is the common all-good case.
        2. On :class:`~sqlalchemy.exc.IntegrityError` the batch cannot tell us
           WHICH row violated a UNIQUE / NOT NULL / CHECK / foreign-key
           constraint, so we roll the batch savepoint back and re-execute the
           batch row-by-row, each row in its own SAVEPOINT. A row that commits
           is recorded as written; a row that raises ``IntegrityError`` is
           rolled back to its savepoint and diverted via ``_divert_row`` (the
           canonical pipeline row ``rows[i]`` is handed to the failsink, not
           the JSON-serialized ``insert_rows[i]``).

        Why a SAVEPOINT and not a fresh per-row ``engine.begin()``: on
        Postgres / SQL Server an IntegrityError aborts the surrounding
        transaction, so subsequent rows cannot be probed without a savepoint
        to roll back to. Separate per-row transactions would also leave
        already-committed good rows durable if a later row raised a
        connection/operational error — the audit trail would then record an
        ERROR while rows were physically written (a double-write on retry).
        Keeping one outer transaction guarantees that a non-attributable
        failure rolls back the good rows too, so the recorded ERROR is honest.

        Per-row-attributable failures (``IntegrityError``) divert. All other
        failures (connection/operational/programming/auth) are batch-integrity
        failures that affect every row equally — they propagate out of the
        outer transaction, roll everything back, and are re-raised by the
        caller (recorded as an ERROR call).

        Returns the rows that were actually committed (1:1 subset of
        ``insert_rows``), in input order.

        SAVEPOINT dependency: this recovery pattern REQUIRES the backend to
        support SAVEPOINT (``begin_nested()``). Every dialect this sink
        targets does — sqlite (pysqlite), PostgreSQL, MySQL/InnoDB, SQL
        Server. If a backend lacked savepoint support, ``begin_nested()``
        would raise; that raise propagates out of the outer transaction as a
        batch-integrity failure (recorded ERROR + re-raise), rolling back
        everything. That is a safe crash — no double-write, no row loss — not
        a silent or partial-write path, so no speculative non-savepoint
        fallback is provided.
        """
        assert self._engine is not None and self._table is not None, "engine/table is None at INSERT time — invariant violation"
        table = self._table
        written_rows: list[dict[str, Any]] = []
        with self._engine.begin() as conn:
            batch_savepoint = conn.begin_nested()
            try:
                conn.execute(insert(table), insert_rows)
                batch_savepoint.commit()
            except IntegrityError:
                # Per-row-attributable: re-execute row-by-row to identify the
                # offending row(s). Roll the failed batch attempt back first.
                batch_savepoint.rollback()
                written_rows = []
                for i, sql_row in enumerate(insert_rows):
                    row_savepoint = conn.begin_nested()
                    try:
                        conn.execute(insert(table), [sql_row])
                        row_savepoint.commit()
                        written_rows.append(sql_row)
                    except IntegrityError as exc:
                        row_savepoint.rollback()
                        self._divert_row(
                            rows[i],
                            row_index=i,
                            reason=f"Constraint violation: {exc.orig}",
                        )
            else:
                # Batch fast path committed every row.
                written_rows = list(insert_rows)
        return written_rows

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

    @classmethod
    def get_agent_assistance(cls, *, issue_code: str | None = None) -> PluginAssistance | None:
        if issue_code is None:
            return PluginAssistance(
                plugin_name="database",
                issue_code=None,
                summary="Write rows to a SQL table (Postgres, SQLite, SQL Server) via SQLAlchemy. Write modes: insert, upsert, replace.",
                composer_hints=(
                    "write_mode: 'insert' (default, append-only), 'upsert' (requires unique key constraint), 'replace' (drops + recreates table at run start — destructive).",
                    "url is sanitised and audit-recorded — never put credentials inline; use the secrets store.",
                    "Schema fields map to column types: string→TEXT, int→Integer, float→Float, bool→Boolean. Other types need explicit type_coerce upstream.",
                    "table-replace mode is irreversible mid-run. Confirm with the operator before declaring 'replace' on existing data.",
                    "on_write_failure routes per-row constraint violations (UNIQUE / NOT NULL / CHECK / foreign-key) — the offending row is diverted and the rest of the batch commits. Batch-integrity failures (connection loss, lock timeout, bad SQL) are not row-attributable and crash the run.",
                ),
            )
        return None

    @classmethod
    def get_post_call_hints(
        cls,
        *,
        tool_name: str,
        config_snapshot: Mapping[str, object],
    ) -> tuple[str, ...]:
        hints: list[str] = []
        if "write_mode" in config_snapshot:
            write_mode = config_snapshot["write_mode"]
            if write_mode == "replace":
                hints.append(
                    "write_mode: 'replace' DROPS and recreates the target table at run start. Confirm with the operator that the existing data is expendable before running."
                )
            elif write_mode == "upsert":
                hints.append(
                    "write_mode: 'upsert' requires a unique constraint on the target table for the merge key. Verify the constraint exists before running, or the upsert will degenerate to insert."
                )
        return tuple(hints)
