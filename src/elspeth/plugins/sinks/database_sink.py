# src/elspeth/plugins/sinks/database_sink.py
"""Database sink plugin for ELSPETH.

Writes rows to a database table using SQLAlchemy Core.
"""

import hashlib
import json
from typing import Any, Literal

from sqlalchemy import Column, MetaData, String, Table, create_engine, insert
from sqlalchemy.engine import Engine

from elspeth.contracts import ArtifactDescriptor, PluginSchema
from elspeth.plugins.base import BaseSink
from elspeth.plugins.config_base import PluginConfig
from elspeth.plugins.context import PluginContext


class DatabaseInputSchema(PluginSchema):
    """Dynamic schema - accepts any row structure."""

    model_config = {"extra": "allow"}  # noqa: RUF012 - Pydantic pattern


class DatabaseSinkConfig(PluginConfig):
    """Configuration for database sink plugin."""

    url: str
    table: str
    if_exists: Literal["append", "replace"] = "append"


class DatabaseSink(BaseSink):
    """Write rows to a database table.

    Creates the table on first write, inferring columns from row keys.
    Uses SQLAlchemy Core for direct SQL control.

    Returns ArtifactDescriptor with SHA-256 hash of canonical JSON payload
    BEFORE insert. This proves intent - the database may transform data.

    Config options:
        url: Database connection URL (required)
        table: Table name (required)
        if_exists: "append" or "replace" (default: "append")
    """

    name = "database"
    input_schema = DatabaseInputSchema
    plugin_version = "1.0.0"
    # determinism inherited from BaseSink (IO_WRITE)

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        cfg = DatabaseSinkConfig.from_dict(config)
        self._url = cfg.url
        self._table_name = cfg.table
        self._if_exists = cfg.if_exists

        self._engine: Engine | None = None
        self._table: Table | None = None
        self._metadata: MetaData | None = None

    def _ensure_table(self, row: dict[str, Any]) -> None:
        """Create table if it doesn't exist, inferring schema from row."""
        if self._engine is None:
            self._engine = create_engine(self._url)
            self._metadata = MetaData()

        if self._table is None:
            # Infer columns from first row (all as String for simplicity)
            columns = [Column(key, String) for key in row]
            # Metadata is always set when engine is created
            assert self._metadata is not None
            self._table = Table(
                self._table_name,
                self._metadata,
                *columns,
            )
            self._metadata.create_all(self._engine, checkfirst=True)

    def write(
        self, rows: list[dict[str, Any]], ctx: PluginContext
    ) -> ArtifactDescriptor:
        """Write a batch of rows to the database.

        CRITICAL: Hashes the canonical JSON payload BEFORE insert.
        This proves intent - the database may transform data (add timestamps,
        auto-increment IDs, normalize strings, etc.).

        Args:
            rows: List of row dicts to write
            ctx: Plugin context

        Returns:
            ArtifactDescriptor with content_hash (SHA-256) and size_bytes
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
