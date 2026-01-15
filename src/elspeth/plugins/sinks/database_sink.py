# src/elspeth/plugins/sinks/database_sink.py
"""Database sink plugin for ELSPETH.

Writes rows to a database table using SQLAlchemy Core.
"""

from typing import Any, Literal

from sqlalchemy import Column, MetaData, String, Table, create_engine, insert
from sqlalchemy.engine import Engine

from elspeth.plugins.base import BaseSink
from elspeth.plugins.config_base import PluginConfig
from elspeth.plugins.context import PluginContext
from elspeth.plugins.schemas import PluginSchema


class DatabaseInputSchema(PluginSchema):
    """Dynamic schema - accepts any row structure."""

    model_config = {"extra": "allow"}  # noqa: RUF012 - Pydantic pattern


class DatabaseSinkConfig(PluginConfig):
    """Configuration for database sink plugin."""

    url: str
    table: str
    batch_size: int = 100
    if_exists: Literal["append", "replace"] = "append"


class DatabaseSink(BaseSink):
    """Write rows to a database table.

    Creates the table on first write, inferring columns from row keys.
    Uses SQLAlchemy Core for direct SQL control.

    Config options:
        url: Database connection URL (required)
        table: Table name (required)
        batch_size: Rows to buffer before insert (default: 100)
        if_exists: "append" or "replace" (default: "append")
    """

    name = "database"
    input_schema = DatabaseInputSchema

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        cfg = DatabaseSinkConfig.from_dict(config)
        self._url = cfg.url
        self._table_name = cfg.table
        self._batch_size = cfg.batch_size
        self._if_exists = cfg.if_exists

        self._engine: Engine | None = None
        self._table: Table | None = None
        self._metadata: MetaData | None = None
        self._buffer: list[dict[str, Any]] = []

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

    def write(self, row: dict[str, Any], ctx: PluginContext) -> None:
        """Write a row to the database.

        Rows are buffered and inserted in batches for efficiency.
        """
        self._ensure_table(row)
        self._buffer.append(row)

        if len(self._buffer) >= self._batch_size:
            self._flush_buffer()

    def _flush_buffer(self) -> None:
        """Insert buffered rows into database."""
        if not self._buffer or self._engine is None or self._table is None:
            return

        with self._engine.begin() as conn:
            conn.execute(insert(self._table), self._buffer)

        self._buffer = []

    def flush(self) -> None:
        """Flush any remaining buffered rows."""
        self._flush_buffer()

    def close(self) -> None:
        """Close database connection."""
        # Flush any remaining rows
        self._flush_buffer()

        if self._engine is not None:
            self._engine.dispose()
            self._engine = None
            self._table = None
            self._metadata = None
