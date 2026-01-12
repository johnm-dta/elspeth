# Phase 3: SDA Engine Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the SDA Engine that orchestrates plugin execution while recording complete audit trails to Landscape and emitting OpenTelemetry spans.

**Architecture:** The engine wraps plugin calls (transform, gate, aggregation, sink) to add audit behavior without modifying plugin code. LandscapeRecorder provides the high-level API for audit recording. TokenManager handles row instance identity through forks/joins. The Orchestrator coordinates the full run lifecycle.

**Tech Stack:** Python 3.11+, SQLAlchemy Core (database), OpenTelemetry (tracing), tenacity (retries), structlog (logging)

**Dependencies:**
- Phase 1: `elspeth.core.canonical`, `elspeth.core.config`, `elspeth.core.dag`, `elspeth.core.payload_store`, `elspeth.core.landscape.models`
- Phase 2: `elspeth.plugins` (protocols, results, context, schemas, manager)

---

## Task 1: LandscapeSchema - SQLAlchemy Table Definitions

**Files:**
- Create: `src/elspeth/core/landscape/schema.py`
- Create: `tests/core/landscape/test_schema.py`

### Step 1: Write the failing test

```python
# tests/core/landscape/test_schema.py
"""Tests for Landscape SQLAlchemy schema."""

import pytest
from sqlalchemy import create_engine, inspect


class TestLandscapeSchema:
    """SQLAlchemy table definitions for Landscape."""

    def test_all_tables_exist(self) -> None:
        from sqlalchemy import MetaData

        from elspeth.core.landscape.schema import metadata

        table_names = set(metadata.tables.keys())
        expected = {
            "runs",
            "nodes",
            "edges",
            "rows",
            "tokens",
            "token_parents",
            "node_states",
            "routing_events",
            "calls",
            "batches",
            "batch_members",
            "batch_outputs",
            "artifacts",
        }
        assert expected.issubset(table_names)

    def test_create_all_tables(self) -> None:
        from sqlalchemy import create_engine

        from elspeth.core.landscape.schema import metadata

        engine = create_engine("sqlite:///:memory:")
        metadata.create_all(engine)

        inspector = inspect(engine)
        tables = inspector.get_table_names()
        assert "runs" in tables
        assert "tokens" in tables
        assert "node_states" in tables

    def test_runs_table_columns(self) -> None:
        from elspeth.core.landscape.schema import runs_table

        columns = {c.name for c in runs_table.columns}
        assert "run_id" in columns
        assert "started_at" in columns
        assert "config_hash" in columns
        assert "status" in columns
```

### Step 2: Run test to verify it fails

Run: `pytest tests/core/landscape/test_schema.py -v`
Expected: FAIL (ImportError)

### Step 3: Create schema module

```python
# src/elspeth/core/landscape/schema.py
"""SQLAlchemy schema definitions for Landscape audit tables.

These tables track everything that happens during pipeline execution:
- Runs and configuration
- Nodes (plugin instances) and edges (graph structure)
- Rows and tokens (data flow through DAG)
- Node states (what happened at each node)
- Routing events, batches, artifacts
"""

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
)

metadata = MetaData()

# === Runs and Configuration ===

runs_table = Table(
    "runs",
    metadata,
    Column("run_id", String(64), primary_key=True),
    Column("started_at", DateTime, nullable=False),
    Column("completed_at", DateTime, nullable=True),
    Column("config_hash", String(64), nullable=False),
    Column("settings_json", Text, nullable=False),
    Column("reproducibility_grade", String(32), nullable=True),
    Column("canonical_version", String(32), nullable=False),
    Column("status", String(32), nullable=False),  # running, completed, failed
)

# === Nodes and Edges (Execution Graph) ===

nodes_table = Table(
    "nodes",
    metadata,
    Column("node_id", String(64), primary_key=True),
    Column("run_id", String(64), ForeignKey("runs.run_id"), nullable=False),
    Column("plugin_name", String(128), nullable=False),
    Column("node_type", String(32), nullable=False),  # source, transform, gate, etc.
    Column("plugin_version", String(32), nullable=False),
    Column("config_hash", String(64), nullable=False),
    Column("config_json", Text, nullable=False),
    Column("schema_hash", String(64), nullable=True),
    Column("sequence_in_pipeline", Integer, nullable=True),
    Column("registered_at", DateTime, nullable=False),
    Index("ix_nodes_run_id", "run_id"),
)

edges_table = Table(
    "edges",
    metadata,
    Column("edge_id", String(64), primary_key=True),
    Column("run_id", String(64), ForeignKey("runs.run_id"), nullable=False),
    Column("from_node_id", String(64), ForeignKey("nodes.node_id"), nullable=False),
    Column("to_node_id", String(64), ForeignKey("nodes.node_id"), nullable=False),
    Column("label", String(128), nullable=False),  # "continue", route name
    Column("default_mode", String(16), nullable=False),  # move, copy
    Column("created_at", DateTime, nullable=False),
    UniqueConstraint("run_id", "from_node_id", "label", name="uq_edge_from_label"),
    Index("ix_edges_run_id", "run_id"),
)

# === Rows and Tokens (Data Flow) ===

rows_table = Table(
    "rows",
    metadata,
    Column("row_id", String(64), primary_key=True),
    Column("run_id", String(64), ForeignKey("runs.run_id"), nullable=False),
    Column("source_node_id", String(64), ForeignKey("nodes.node_id"), nullable=False),
    Column("row_index", Integer, nullable=False),
    Column("source_data_hash", String(64), nullable=False),
    Column("source_data_ref", String(256), nullable=True),  # Payload store ref
    Column("created_at", DateTime, nullable=False),
    UniqueConstraint("run_id", "row_index", name="uq_row_index"),
    Index("ix_rows_run_id", "run_id"),
)

tokens_table = Table(
    "tokens",
    metadata,
    Column("token_id", String(64), primary_key=True),
    Column("row_id", String(64), ForeignKey("rows.row_id"), nullable=False),
    Column("fork_group_id", String(64), nullable=True),
    Column("join_group_id", String(64), nullable=True),
    Column("branch_name", String(128), nullable=True),
    Column("created_at", DateTime, nullable=False),
    Index("ix_tokens_row_id", "row_id"),
    Index("ix_tokens_fork_group", "fork_group_id"),
)

token_parents_table = Table(
    "token_parents",
    metadata,
    Column("token_id", String(64), ForeignKey("tokens.token_id"), nullable=False),
    Column("parent_token_id", String(64), ForeignKey("tokens.token_id"), nullable=False),
    Column("ordinal", Integer, nullable=False),
    UniqueConstraint("token_id", "ordinal", name="uq_token_parent_ordinal"),
    Index("ix_token_parents_token", "token_id"),
)

# === Node States (Processing Records) ===

node_states_table = Table(
    "node_states",
    metadata,
    Column("state_id", String(64), primary_key=True),
    Column("token_id", String(64), ForeignKey("tokens.token_id"), nullable=False),
    Column("node_id", String(64), ForeignKey("nodes.node_id"), nullable=False),
    Column("step_index", Integer, nullable=False),
    Column("attempt", Integer, nullable=False, default=0),
    Column("status", String(32), nullable=False),  # open, completed, failed
    Column("input_hash", String(64), nullable=False),
    Column("output_hash", String(64), nullable=True),
    Column("context_before_json", Text, nullable=True),
    Column("context_after_json", Text, nullable=True),
    Column("duration_ms", Float, nullable=True),
    Column("error_json", Text, nullable=True),
    Column("started_at", DateTime, nullable=False),
    Column("completed_at", DateTime, nullable=True),
    UniqueConstraint("token_id", "node_id", "attempt", name="uq_state_token_node_attempt"),
    UniqueConstraint("token_id", "step_index", name="uq_state_token_step"),
    Index("ix_node_states_token", "token_id"),
    Index("ix_node_states_node", "node_id"),
)

# === Routing Events ===

routing_events_table = Table(
    "routing_events",
    metadata,
    Column("event_id", String(64), primary_key=True),
    Column("state_id", String(64), ForeignKey("node_states.state_id"), nullable=False),
    Column("edge_id", String(64), ForeignKey("edges.edge_id"), nullable=False),
    Column("routing_group_id", String(64), nullable=False),
    Column("ordinal", Integer, nullable=False),
    Column("mode", String(16), nullable=False),  # move, copy
    Column("reason_hash", String(64), nullable=True),
    Column("reason_ref", String(256), nullable=True),
    Column("created_at", DateTime, nullable=False),
    UniqueConstraint("routing_group_id", "ordinal", name="uq_routing_ordinal"),
    Index("ix_routing_state", "state_id"),
)

# === External Calls ===

calls_table = Table(
    "calls",
    metadata,
    Column("call_id", String(64), primary_key=True),
    Column("state_id", String(64), ForeignKey("node_states.state_id"), nullable=False),
    Column("call_index", Integer, nullable=False),
    Column("call_type", String(32), nullable=False),  # llm, http, sql, filesystem
    Column("status", String(32), nullable=False),  # success, error
    Column("request_hash", String(64), nullable=False),
    Column("request_ref", String(256), nullable=True),
    Column("response_hash", String(64), nullable=True),
    Column("response_ref", String(256), nullable=True),
    Column("error_json", Text, nullable=True),
    Column("latency_ms", Float, nullable=True),
    Column("created_at", DateTime, nullable=False),
    UniqueConstraint("state_id", "call_index", name="uq_call_index"),
    Index("ix_calls_state", "state_id"),
)

# === Batches (Aggregation) ===

batches_table = Table(
    "batches",
    metadata,
    Column("batch_id", String(64), primary_key=True),
    Column("run_id", String(64), ForeignKey("runs.run_id"), nullable=False),
    Column("aggregation_node_id", String(64), ForeignKey("nodes.node_id"), nullable=False),
    Column("aggregation_state_id", String(64), ForeignKey("node_states.state_id"), nullable=True),
    Column("trigger_reason", String(128), nullable=True),
    Column("attempt", Integer, nullable=False, default=0),
    Column("status", String(32), nullable=False),  # draft, executing, completed, failed
    Column("created_at", DateTime, nullable=False),
    Column("completed_at", DateTime, nullable=True),
    Index("ix_batches_run", "run_id"),
    Index("ix_batches_node", "aggregation_node_id"),
)

batch_members_table = Table(
    "batch_members",
    metadata,
    Column("batch_id", String(64), ForeignKey("batches.batch_id"), nullable=False),
    Column("token_id", String(64), ForeignKey("tokens.token_id"), nullable=False),
    Column("ordinal", Integer, nullable=False),
    UniqueConstraint("batch_id", "ordinal", name="uq_batch_member_ordinal"),
    Index("ix_batch_members_batch", "batch_id"),
)

batch_outputs_table = Table(
    "batch_outputs",
    metadata,
    Column("batch_id", String(64), ForeignKey("batches.batch_id"), nullable=False),
    Column("output_type", String(32), nullable=False),  # token, artifact
    Column("output_id", String(64), nullable=False),
    Index("ix_batch_outputs_batch", "batch_id"),
)

# === Artifacts ===

artifacts_table = Table(
    "artifacts",
    metadata,
    Column("artifact_id", String(64), primary_key=True),
    Column("run_id", String(64), ForeignKey("runs.run_id"), nullable=False),
    Column("produced_by_state_id", String(64), ForeignKey("node_states.state_id"), nullable=False),
    Column("sink_node_id", String(64), ForeignKey("nodes.node_id"), nullable=False),
    Column("artifact_type", String(64), nullable=False),
    Column("path_or_uri", Text, nullable=False),
    Column("content_hash", String(64), nullable=False),
    Column("size_bytes", Integer, nullable=False),
    Column("created_at", DateTime, nullable=False),
    Index("ix_artifacts_run", "run_id"),
    Index("ix_artifacts_sink", "sink_node_id"),
)
```

### Step 4: Create test directory

```bash
mkdir -p tests/core/landscape
touch tests/core/landscape/__init__.py
```

### Step 5: Run tests to verify they pass

Run: `pytest tests/core/landscape/test_schema.py -v`
Expected: PASS

### Step 6: Commit

```bash
git add src/elspeth/core/landscape/schema.py tests/core/landscape/
git commit -m "feat(landscape): add SQLAlchemy schema definitions"
```

---

## Task 2: LandscapeDB - Database Connection Manager

**Files:**
- Create: `src/elspeth/core/landscape/database.py`
- Create: `tests/core/landscape/test_database.py`

### Step 1: Write the failing test

```python
# tests/core/landscape/test_database.py
"""Tests for Landscape database connection."""

import pytest
from sqlalchemy import inspect


class TestLandscapeDB:
    """Database connection and session management."""

    def test_create_in_memory(self) -> None:
        from elspeth.core.landscape.database import LandscapeDB

        db = LandscapeDB.in_memory()
        assert db.engine is not None

    def test_tables_created(self) -> None:
        from elspeth.core.landscape.database import LandscapeDB

        db = LandscapeDB.in_memory()
        inspector = inspect(db.engine)
        tables = inspector.get_table_names()
        assert "runs" in tables
        assert "tokens" in tables

    def test_connection_context(self) -> None:
        from elspeth.core.landscape.database import LandscapeDB

        db = LandscapeDB.in_memory()
        with db.connection() as conn:
            result = conn.execute("SELECT 1")
            assert result.scalar() == 1

    def test_from_url(self, tmp_path) -> None:
        from elspeth.core.landscape.database import LandscapeDB

        db_path = tmp_path / "test.db"
        db = LandscapeDB.from_url(f"sqlite:///{db_path}")
        assert db_path.exists()
```

### Step 2: Run test to verify it fails

Run: `pytest tests/core/landscape/test_database.py -v`
Expected: FAIL (ImportError)

### Step 3: Create database module

```python
# src/elspeth/core/landscape/database.py
"""Database connection manager for Landscape.

Provides connection pooling and session management for the audit database.
Supports SQLite (development/testing) and PostgreSQL (production).
"""

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import Connection, create_engine, text
from sqlalchemy.engine import Engine

from elspeth.core.landscape.schema import metadata


class LandscapeDB:
    """Database connection manager for Landscape audit tables.

    Example:
        db = LandscapeDB.in_memory()  # For testing
        db = LandscapeDB.from_url("postgresql://...")  # For production

        with db.connection() as conn:
            conn.execute(...)
    """

    def __init__(self, engine: Engine) -> None:
        """Initialize with SQLAlchemy engine."""
        self._engine = engine

    @property
    def engine(self) -> Engine:
        """Get the SQLAlchemy engine."""
        return self._engine

    @classmethod
    def in_memory(cls) -> "LandscapeDB":
        """Create an in-memory SQLite database for testing.

        Tables are created automatically.
        """
        engine = create_engine("sqlite:///:memory:", echo=False)
        metadata.create_all(engine)
        return cls(engine)

    @classmethod
    def from_url(cls, url: str, *, create_tables: bool = True) -> "LandscapeDB":
        """Create database from connection URL.

        Args:
            url: SQLAlchemy connection URL
            create_tables: Whether to create tables if they don't exist

        Returns:
            LandscapeDB instance
        """
        engine = create_engine(url, echo=False)
        if create_tables:
            metadata.create_all(engine)
        return cls(engine)

    @contextmanager
    def connection(self) -> Iterator[Connection]:
        """Get a database connection.

        Usage:
            with db.connection() as conn:
                conn.execute(...)
        """
        with self._engine.connect() as conn:
            yield conn
            conn.commit()

    def execute(self, statement: str, parameters: dict | None = None) -> None:
        """Execute a SQL statement.

        Args:
            statement: SQL statement
            parameters: Optional parameters
        """
        with self.connection() as conn:
            conn.execute(text(statement), parameters or {})

    def close(self) -> None:
        """Close the database connection."""
        self._engine.dispose()
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/core/landscape/test_database.py -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/elspeth/core/landscape/database.py tests/core/landscape/test_database.py
git commit -m "feat(landscape): add LandscapeDB connection manager"
```

---

## Task 3: LandscapeRecorder - Run Management

**Files:**
- Create: `src/elspeth/core/landscape/recorder.py`
- Create: `tests/core/landscape/test_recorder.py`

### Step 1: Write the failing test

```python
# tests/core/landscape/test_recorder.py
"""Tests for LandscapeRecorder."""

from datetime import datetime, timezone

import pytest


class TestLandscapeRecorderRuns:
    """Run lifecycle management."""

    def test_begin_run(self) -> None:
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(
            config={"source": "test.csv"},
            canonical_version="sha256-rfc8785-v1",
        )

        assert run.run_id is not None
        assert run.status == "running"
        assert run.started_at is not None

    def test_complete_run_success(self) -> None:
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={}, canonical_version="v1")
        completed = recorder.complete_run(run.run_id, status="completed")

        assert completed.status == "completed"
        assert completed.completed_at is not None

    def test_complete_run_failed(self) -> None:
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={}, canonical_version="v1")
        completed = recorder.complete_run(run.run_id, status="failed")

        assert completed.status == "failed"

    def test_get_run(self) -> None:
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={"key": "value"}, canonical_version="v1")
        retrieved = recorder.get_run(run.run_id)

        assert retrieved is not None
        assert retrieved.run_id == run.run_id
```

### Step 2: Run test to verify it fails

Run: `pytest tests/core/landscape/test_recorder.py::TestLandscapeRecorderRuns -v`
Expected: FAIL (ImportError)

### Step 3: Create recorder module with run management

```python
# src/elspeth/core/landscape/recorder.py
"""LandscapeRecorder: High-level API for audit recording.

This is the main interface for recording audit trail entries during
pipeline execution. It wraps the low-level database operations.
"""

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select

from elspeth.core.canonical import canonical_json, stable_hash
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.models import Run
from elspeth.core.landscape.schema import runs_table


def _now() -> datetime:
    """Get current UTC timestamp."""
    return datetime.now(timezone.utc)


def _generate_id() -> str:
    """Generate a unique ID."""
    return uuid.uuid4().hex


class LandscapeRecorder:
    """High-level API for recording audit trail entries.

    This class provides methods to record:
    - Runs and their configuration
    - Nodes (plugin instances) and edges
    - Rows and tokens (data flow)
    - Node states (processing records)
    - Routing events, batches, artifacts

    Example:
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={"source": "data.csv"})
        # ... execute pipeline ...
        recorder.complete_run(run.run_id, status="completed")
    """

    def __init__(self, db: LandscapeDB) -> None:
        """Initialize recorder with database connection."""
        self._db = db

    # === Run Management ===

    def begin_run(
        self,
        config: dict[str, Any],
        canonical_version: str,
        *,
        run_id: str | None = None,
        reproducibility_grade: str | None = None,
    ) -> Run:
        """Begin a new pipeline run.

        Args:
            config: Resolved configuration dictionary
            canonical_version: Version of canonical hash algorithm
            run_id: Optional run ID (generated if not provided)
            reproducibility_grade: Optional grade (FULL_REPRODUCIBLE, etc.)

        Returns:
            Run model with generated run_id
        """
        run_id = run_id or _generate_id()
        settings_json = canonical_json(config)
        config_hash = stable_hash(config)
        now = _now()

        run = Run(
            run_id=run_id,
            started_at=now,
            config_hash=config_hash,
            settings_json=settings_json,
            canonical_version=canonical_version,
            status="running",
            reproducibility_grade=reproducibility_grade,
        )

        with self._db.connection() as conn:
            conn.execute(
                runs_table.insert().values(
                    run_id=run.run_id,
                    started_at=run.started_at,
                    config_hash=run.config_hash,
                    settings_json=run.settings_json,
                    canonical_version=run.canonical_version,
                    status=run.status,
                    reproducibility_grade=run.reproducibility_grade,
                )
            )

        return run

    def complete_run(
        self,
        run_id: str,
        status: str,
        *,
        reproducibility_grade: str | None = None,
    ) -> Run:
        """Complete a pipeline run.

        Args:
            run_id: Run to complete
            status: Final status (completed, failed)
            reproducibility_grade: Optional final grade

        Returns:
            Updated Run model
        """
        now = _now()

        with self._db.connection() as conn:
            conn.execute(
                runs_table.update()
                .where(runs_table.c.run_id == run_id)
                .values(
                    status=status,
                    completed_at=now,
                    reproducibility_grade=reproducibility_grade,
                )
            )

        return self.get_run(run_id)  # type: ignore

    def get_run(self, run_id: str) -> Run | None:
        """Get a run by ID.

        Args:
            run_id: Run ID to retrieve

        Returns:
            Run model or None if not found
        """
        with self._db.connection() as conn:
            result = conn.execute(
                select(runs_table).where(runs_table.c.run_id == run_id)
            )
            row = result.fetchone()

        if row is None:
            return None

        return Run(
            run_id=row.run_id,
            started_at=row.started_at,
            completed_at=row.completed_at,
            config_hash=row.config_hash,
            settings_json=row.settings_json,
            canonical_version=row.canonical_version,
            status=row.status,
            reproducibility_grade=row.reproducibility_grade,
        )
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/core/landscape/test_recorder.py::TestLandscapeRecorderRuns -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/elspeth/core/landscape/recorder.py tests/core/landscape/test_recorder.py
git commit -m "feat(landscape): add LandscapeRecorder with run management"
```

---

## Task 4: LandscapeRecorder - Node and Edge Registration

**Files:**
- Modify: `src/elspeth/core/landscape/recorder.py`
- Modify: `tests/core/landscape/test_recorder.py`

### Step 1: Write the failing tests

```python
# Add to tests/core/landscape/test_recorder.py

class TestLandscapeRecorderNodes:
    """Node and edge registration."""

    def test_register_node(self) -> None:
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
            node_type="source",
            plugin_version="1.0.0",
            config={"path": "data.csv"},
            sequence=0,
        )

        assert node.node_id is not None
        assert node.plugin_name == "csv_source"
        assert node.node_type == "source"

    def test_register_edge(self) -> None:
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type="source",
            plugin_version="1.0",
            config={},
        )
        transform = recorder.register_node(
            run_id=run.run_id,
            plugin_name="transform",
            node_type="transform",
            plugin_version="1.0",
            config={},
        )

        edge = recorder.register_edge(
            run_id=run.run_id,
            from_node_id=source.node_id,
            to_node_id=transform.node_id,
            label="continue",
            mode="move",
        )

        assert edge.edge_id is not None
        assert edge.label == "continue"

    def test_get_nodes_for_run(self) -> None:
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type="source",
            plugin_version="1.0",
            config={},
        )
        recorder.register_node(
            run_id=run.run_id,
            plugin_name="sink",
            node_type="sink",
            plugin_version="1.0",
            config={},
        )

        nodes = recorder.get_nodes(run.run_id)
        assert len(nodes) == 2
```

### Step 2: Run test to verify it fails

Run: `pytest tests/core/landscape/test_recorder.py::TestLandscapeRecorderNodes -v`
Expected: FAIL

### Step 3: Add node and edge registration

```python
# Add to src/elspeth/core/landscape/recorder.py

from elspeth.core.landscape.models import Edge, Node
from elspeth.core.landscape.schema import edges_table, nodes_table

# Add to LandscapeRecorder class:

    # === Node and Edge Registration ===

    def register_node(
        self,
        run_id: str,
        plugin_name: str,
        node_type: str,
        plugin_version: str,
        config: dict[str, Any],
        *,
        node_id: str | None = None,
        sequence: int | None = None,
        schema_hash: str | None = None,
    ) -> Node:
        """Register a plugin instance (node) in the execution graph.

        Args:
            run_id: Run this node belongs to
            plugin_name: Name of the plugin
            node_type: Type (source, transform, gate, aggregation, coalesce, sink)
            plugin_version: Version of the plugin
            config: Plugin configuration
            node_id: Optional node ID (generated if not provided)
            sequence: Position in pipeline
            schema_hash: Optional input/output schema hash

        Returns:
            Node model
        """
        node_id = node_id or _generate_id()
        config_json = canonical_json(config)
        config_hash = stable_hash(config)
        now = _now()

        node = Node(
            node_id=node_id,
            run_id=run_id,
            plugin_name=plugin_name,
            node_type=node_type,
            plugin_version=plugin_version,
            config_hash=config_hash,
            config_json=config_json,
            schema_hash=schema_hash,
            sequence_in_pipeline=sequence,
            registered_at=now,
        )

        with self._db.connection() as conn:
            conn.execute(
                nodes_table.insert().values(
                    node_id=node.node_id,
                    run_id=node.run_id,
                    plugin_name=node.plugin_name,
                    node_type=node.node_type,
                    plugin_version=node.plugin_version,
                    config_hash=node.config_hash,
                    config_json=node.config_json,
                    schema_hash=node.schema_hash,
                    sequence_in_pipeline=node.sequence_in_pipeline,
                    registered_at=node.registered_at,
                )
            )

        return node

    def register_edge(
        self,
        run_id: str,
        from_node_id: str,
        to_node_id: str,
        label: str,
        mode: str,
        *,
        edge_id: str | None = None,
    ) -> Edge:
        """Register an edge in the execution graph.

        Args:
            run_id: Run this edge belongs to
            from_node_id: Source node
            to_node_id: Destination node
            label: Edge label ("continue", route name, etc.)
            mode: Default routing mode ("move" or "copy")
            edge_id: Optional edge ID (generated if not provided)

        Returns:
            Edge model
        """
        edge_id = edge_id or _generate_id()
        now = _now()

        edge = Edge(
            edge_id=edge_id,
            run_id=run_id,
            from_node_id=from_node_id,
            to_node_id=to_node_id,
            label=label,
            default_mode=mode,
            created_at=now,
        )

        with self._db.connection() as conn:
            conn.execute(
                edges_table.insert().values(
                    edge_id=edge.edge_id,
                    run_id=edge.run_id,
                    from_node_id=edge.from_node_id,
                    to_node_id=edge.to_node_id,
                    label=edge.label,
                    default_mode=edge.default_mode,
                    created_at=edge.created_at,
                )
            )

        return edge

    def get_nodes(self, run_id: str) -> list[Node]:
        """Get all nodes for a run.

        Args:
            run_id: Run ID

        Returns:
            List of Node models
        """
        with self._db.connection() as conn:
            result = conn.execute(
                select(nodes_table)
                .where(nodes_table.c.run_id == run_id)
                .order_by(nodes_table.c.sequence_in_pipeline)
            )
            rows = result.fetchall()

        return [
            Node(
                node_id=row.node_id,
                run_id=row.run_id,
                plugin_name=row.plugin_name,
                node_type=row.node_type,
                plugin_version=row.plugin_version,
                config_hash=row.config_hash,
                config_json=row.config_json,
                schema_hash=row.schema_hash,
                sequence_in_pipeline=row.sequence_in_pipeline,
                registered_at=row.registered_at,
            )
            for row in rows
        ]
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/core/landscape/test_recorder.py::TestLandscapeRecorderNodes -v`
Expected: PASS

### Step 5: Commit

```bash
git add -u
git commit -m "feat(landscape): add node and edge registration to LandscapeRecorder"
```

---

## Task 5: LandscapeRecorder - Row and Token Creation

**Files:**
- Modify: `src/elspeth/core/landscape/recorder.py`
- Modify: `tests/core/landscape/test_recorder.py`

### Step 1: Write the failing tests

```python
# Add to tests/core/landscape/test_recorder.py

class TestLandscapeRecorderTokens:
    """Row and token management."""

    def test_create_row(self) -> None:
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type="source",
            plugin_version="1.0",
            config={},
        )

        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data={"value": 42},
        )

        assert row.row_id is not None
        assert row.row_index == 0
        assert row.source_data_hash is not None

    def test_create_initial_token(self) -> None:
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type="source",
            plugin_version="1.0",
            config={},
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data={"value": 42},
        )

        token = recorder.create_token(row_id=row.row_id)

        assert token.token_id is not None
        assert token.row_id == row.row_id
        assert token.fork_group_id is None  # Initial token

    def test_fork_token(self) -> None:
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type="source",
            plugin_version="1.0",
            config={},
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data={},
        )
        parent_token = recorder.create_token(row_id=row.row_id)

        # Fork to two branches
        child_tokens = recorder.fork_token(
            parent_token_id=parent_token.token_id,
            row_id=row.row_id,
            branches=["stats", "classifier"],
        )

        assert len(child_tokens) == 2
        assert child_tokens[0].branch_name == "stats"
        assert child_tokens[1].branch_name == "classifier"
        # All children share same fork_group_id
        assert child_tokens[0].fork_group_id == child_tokens[1].fork_group_id

    def test_coalesce_tokens(self) -> None:
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type="source",
            plugin_version="1.0",
            config={},
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data={},
        )
        parent = recorder.create_token(row_id=row.row_id)
        children = recorder.fork_token(
            parent_token_id=parent.token_id,
            row_id=row.row_id,
            branches=["a", "b"],
        )

        # Coalesce back together
        merged = recorder.coalesce_tokens(
            parent_token_ids=[c.token_id for c in children],
            row_id=row.row_id,
        )

        assert merged.token_id is not None
        assert merged.join_group_id is not None
```

### Step 2: Run test to verify it fails

Run: `pytest tests/core/landscape/test_recorder.py::TestLandscapeRecorderTokens -v`
Expected: FAIL

### Step 3: Add row and token management

```python
# Add imports to src/elspeth/core/landscape/recorder.py
from elspeth.core.landscape.models import Row, Token, TokenParent
from elspeth.core.landscape.schema import rows_table, token_parents_table, tokens_table

# Add to LandscapeRecorder class:

    # === Row and Token Management ===

    def create_row(
        self,
        run_id: str,
        source_node_id: str,
        row_index: int,
        data: dict[str, Any],
        *,
        row_id: str | None = None,
        payload_ref: str | None = None,
    ) -> Row:
        """Create a source row record.

        Args:
            run_id: Run this row belongs to
            source_node_id: Source node that loaded this row
            row_index: Position in source (0-indexed)
            data: Row data for hashing
            row_id: Optional row ID (generated if not provided)
            payload_ref: Optional reference to payload store

        Returns:
            Row model
        """
        row_id = row_id or _generate_id()
        data_hash = stable_hash(data)
        now = _now()

        row = Row(
            row_id=row_id,
            run_id=run_id,
            source_node_id=source_node_id,
            row_index=row_index,
            source_data_hash=data_hash,
            source_data_ref=payload_ref,
            created_at=now,
        )

        with self._db.connection() as conn:
            conn.execute(
                rows_table.insert().values(
                    row_id=row.row_id,
                    run_id=row.run_id,
                    source_node_id=row.source_node_id,
                    row_index=row.row_index,
                    source_data_hash=row.source_data_hash,
                    source_data_ref=row.source_data_ref,
                    created_at=row.created_at,
                )
            )

        return row

    def create_token(
        self,
        row_id: str,
        *,
        token_id: str | None = None,
        branch_name: str | None = None,
        fork_group_id: str | None = None,
        join_group_id: str | None = None,
    ) -> Token:
        """Create a token (row instance in DAG path).

        Args:
            row_id: Source row this token represents
            token_id: Optional token ID (generated if not provided)
            branch_name: Optional branch name (for forked tokens)
            fork_group_id: Optional fork group (links siblings)
            join_group_id: Optional join group (links merged tokens)

        Returns:
            Token model
        """
        token_id = token_id or _generate_id()
        now = _now()

        token = Token(
            token_id=token_id,
            row_id=row_id,
            fork_group_id=fork_group_id,
            join_group_id=join_group_id,
            branch_name=branch_name,
            created_at=now,
        )

        with self._db.connection() as conn:
            conn.execute(
                tokens_table.insert().values(
                    token_id=token.token_id,
                    row_id=token.row_id,
                    fork_group_id=token.fork_group_id,
                    join_group_id=token.join_group_id,
                    branch_name=token.branch_name,
                    created_at=token.created_at,
                )
            )

        return token

    def fork_token(
        self,
        parent_token_id: str,
        row_id: str,
        branches: list[str],
    ) -> list[Token]:
        """Fork a token to multiple branches.

        Creates child tokens for each branch, all sharing a fork_group_id.
        Records parent relationships.

        Args:
            parent_token_id: Token being forked
            row_id: Row ID (same for all children)
            branches: List of branch names

        Returns:
            List of child Token models
        """
        fork_group_id = _generate_id()
        children = []

        with self._db.connection() as conn:
            for ordinal, branch_name in enumerate(branches):
                child_id = _generate_id()
                now = _now()

                # Create child token
                conn.execute(
                    tokens_table.insert().values(
                        token_id=child_id,
                        row_id=row_id,
                        fork_group_id=fork_group_id,
                        branch_name=branch_name,
                        created_at=now,
                    )
                )

                # Record parent relationship
                conn.execute(
                    token_parents_table.insert().values(
                        token_id=child_id,
                        parent_token_id=parent_token_id,
                        ordinal=ordinal,
                    )
                )

                children.append(
                    Token(
                        token_id=child_id,
                        row_id=row_id,
                        fork_group_id=fork_group_id,
                        branch_name=branch_name,
                        created_at=now,
                    )
                )

        return children

    def coalesce_tokens(
        self,
        parent_token_ids: list[str],
        row_id: str,
    ) -> Token:
        """Coalesce multiple tokens into one (join operation).

        Creates a new token representing the merged result.
        Records all parent relationships.

        Args:
            parent_token_ids: Tokens being merged
            row_id: Row ID for the merged token

        Returns:
            Merged Token model
        """
        join_group_id = _generate_id()
        token_id = _generate_id()
        now = _now()

        with self._db.connection() as conn:
            # Create merged token
            conn.execute(
                tokens_table.insert().values(
                    token_id=token_id,
                    row_id=row_id,
                    join_group_id=join_group_id,
                    created_at=now,
                )
            )

            # Record all parent relationships
            for ordinal, parent_id in enumerate(parent_token_ids):
                conn.execute(
                    token_parents_table.insert().values(
                        token_id=token_id,
                        parent_token_id=parent_id,
                        ordinal=ordinal,
                    )
                )

        return Token(
            token_id=token_id,
            row_id=row_id,
            join_group_id=join_group_id,
            created_at=now,
        )
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/core/landscape/test_recorder.py::TestLandscapeRecorderTokens -v`
Expected: PASS

### Step 5: Commit

```bash
git add -u
git commit -m "feat(landscape): add row and token management to LandscapeRecorder"
```

---

## Task 6: LandscapeRecorder - NodeState Recording

**Files:**
- Modify: `src/elspeth/core/landscape/recorder.py`
- Modify: `tests/core/landscape/test_recorder.py`

### Step 1: Write the failing tests

```python
# Add to tests/core/landscape/test_recorder.py

class TestLandscapeRecorderNodeStates:
    """Node state recording (what happened at each node)."""

    def test_begin_node_state(self) -> None:
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type="source",
            plugin_version="1.0",
            config={},
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data={},
        )
        token = recorder.create_token(row_id=row.row_id)

        state = recorder.begin_node_state(
            token_id=token.token_id,
            node_id=source.node_id,
            step_index=0,
            input_data={"value": 42},
        )

        assert state.state_id is not None
        assert state.status == "open"
        assert state.input_hash is not None

    def test_complete_node_state_success(self) -> None:
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="transform",
            node_type="transform",
            plugin_version="1.0",
            config={},
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=node.node_id,
            row_index=0,
            data={},
        )
        token = recorder.create_token(row_id=row.row_id)
        state = recorder.begin_node_state(
            token_id=token.token_id,
            node_id=node.node_id,
            step_index=0,
            input_data={"x": 1},
        )

        completed = recorder.complete_node_state(
            state_id=state.state_id,
            status="completed",
            output_data={"x": 1, "y": 2},
            duration_ms=10.5,
        )

        assert completed.status == "completed"
        assert completed.output_hash is not None
        assert completed.duration_ms == 10.5
        assert completed.completed_at is not None

    def test_complete_node_state_failed(self) -> None:
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="transform",
            node_type="transform",
            plugin_version="1.0",
            config={},
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=node.node_id,
            row_index=0,
            data={},
        )
        token = recorder.create_token(row_id=row.row_id)
        state = recorder.begin_node_state(
            token_id=token.token_id,
            node_id=node.node_id,
            step_index=0,
            input_data={},
        )

        completed = recorder.complete_node_state(
            state_id=state.state_id,
            status="failed",
            error={"message": "Validation failed", "code": "E001"},
            duration_ms=5.0,
        )

        assert completed.status == "failed"
        assert completed.error_json is not None
        assert "Validation failed" in completed.error_json

    def test_retry_increments_attempt(self) -> None:
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="transform",
            node_type="transform",
            plugin_version="1.0",
            config={},
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=node.node_id,
            row_index=0,
            data={},
        )
        token = recorder.create_token(row_id=row.row_id)

        # First attempt fails
        state1 = recorder.begin_node_state(
            token_id=token.token_id,
            node_id=node.node_id,
            step_index=0,
            input_data={},
            attempt=0,
        )
        recorder.complete_node_state(state1.state_id, status="failed", error={})

        # Second attempt
        state2 = recorder.begin_node_state(
            token_id=token.token_id,
            node_id=node.node_id,
            step_index=0,
            input_data={},
            attempt=1,
        )

        assert state2.attempt == 1
```

### Step 2: Run test to verify it fails

Run: `pytest tests/core/landscape/test_recorder.py::TestLandscapeRecorderNodeStates -v`
Expected: FAIL

### Step 3: Add node state recording

```python
# Add import to src/elspeth/core/landscape/recorder.py
from elspeth.core.landscape.models import NodeState
from elspeth.core.landscape.schema import node_states_table

# Add to LandscapeRecorder class:

    # === Node State Recording ===

    def begin_node_state(
        self,
        token_id: str,
        node_id: str,
        step_index: int,
        input_data: dict[str, Any],
        *,
        state_id: str | None = None,
        attempt: int = 0,
        context_before: dict[str, Any] | None = None,
    ) -> NodeState:
        """Begin recording a node state (token visiting a node).

        Args:
            token_id: Token being processed
            node_id: Node processing the token
            step_index: Position in token's execution path
            input_data: Input data for hashing
            state_id: Optional state ID (generated if not provided)
            attempt: Attempt number (0 for first attempt)
            context_before: Optional context snapshot before processing

        Returns:
            NodeState model with status="open"
        """
        state_id = state_id or _generate_id()
        input_hash = stable_hash(input_data)
        now = _now()

        context_json = canonical_json(context_before) if context_before else None

        state = NodeState(
            state_id=state_id,
            token_id=token_id,
            node_id=node_id,
            step_index=step_index,
            attempt=attempt,
            status="open",
            input_hash=input_hash,
            context_before_json=context_json,
            started_at=now,
        )

        with self._db.connection() as conn:
            conn.execute(
                node_states_table.insert().values(
                    state_id=state.state_id,
                    token_id=state.token_id,
                    node_id=state.node_id,
                    step_index=state.step_index,
                    attempt=state.attempt,
                    status=state.status,
                    input_hash=state.input_hash,
                    context_before_json=state.context_before_json,
                    started_at=state.started_at,
                )
            )

        return state

    def complete_node_state(
        self,
        state_id: str,
        status: str,
        *,
        output_data: dict[str, Any] | None = None,
        duration_ms: float | None = None,
        error: dict[str, Any] | None = None,
        context_after: dict[str, Any] | None = None,
    ) -> NodeState:
        """Complete a node state.

        Args:
            state_id: State to complete
            status: Final status (completed, failed)
            output_data: Output data for hashing (if success)
            duration_ms: Processing duration
            error: Error details (if failed)
            context_after: Optional context snapshot after processing

        Returns:
            Updated NodeState model
        """
        now = _now()
        output_hash = stable_hash(output_data) if output_data else None
        error_json = canonical_json(error) if error else None
        context_json = canonical_json(context_after) if context_after else None

        with self._db.connection() as conn:
            conn.execute(
                node_states_table.update()
                .where(node_states_table.c.state_id == state_id)
                .values(
                    status=status,
                    output_hash=output_hash,
                    duration_ms=duration_ms,
                    error_json=error_json,
                    context_after_json=context_json,
                    completed_at=now,
                )
            )

        return self.get_node_state(state_id)  # type: ignore

    def get_node_state(self, state_id: str) -> NodeState | None:
        """Get a node state by ID.

        Args:
            state_id: State ID to retrieve

        Returns:
            NodeState model or None
        """
        with self._db.connection() as conn:
            result = conn.execute(
                select(node_states_table).where(
                    node_states_table.c.state_id == state_id
                )
            )
            row = result.fetchone()

        if row is None:
            return None

        return NodeState(
            state_id=row.state_id,
            token_id=row.token_id,
            node_id=row.node_id,
            step_index=row.step_index,
            attempt=row.attempt,
            status=row.status,
            input_hash=row.input_hash,
            output_hash=row.output_hash,
            context_before_json=row.context_before_json,
            context_after_json=row.context_after_json,
            duration_ms=row.duration_ms,
            error_json=row.error_json,
            started_at=row.started_at,
            completed_at=row.completed_at,
        )
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/core/landscape/test_recorder.py::TestLandscapeRecorderNodeStates -v`
Expected: PASS

### Step 5: Commit

```bash
git add -u
git commit -m "feat(landscape): add node state recording to LandscapeRecorder"
```

---

## Task 7: LandscapeRecorder - Routing Events

**Files:**
- Modify: `src/elspeth/core/landscape/recorder.py`
- Modify: `tests/core/landscape/test_recorder.py`

### Step 1: Write the failing tests

```python
# Add to tests/core/landscape/test_recorder.py

class TestLandscapeRecorderRouting:
    """Routing event recording."""

    def test_record_routing_event(self) -> None:
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        gate = recorder.register_node(
            run_id=run.run_id,
            plugin_name="gate",
            node_type="gate",
            plugin_version="1.0",
            config={},
        )
        sink = recorder.register_node(
            run_id=run.run_id,
            plugin_name="sink",
            node_type="sink",
            plugin_version="1.0",
            config={},
        )
        edge = recorder.register_edge(
            run_id=run.run_id,
            from_node_id=gate.node_id,
            to_node_id=sink.node_id,
            label="flagged",
            mode="move",
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=gate.node_id,
            row_index=0,
            data={},
        )
        token = recorder.create_token(row_id=row.row_id)
        state = recorder.begin_node_state(
            token_id=token.token_id,
            node_id=gate.node_id,
            step_index=0,
            input_data={},
        )

        event = recorder.record_routing_event(
            state_id=state.state_id,
            edge_id=edge.edge_id,
            mode="move",
            reason={"confidence": 0.95},
        )

        assert event.event_id is not None
        assert event.routing_group_id is not None

    def test_record_multiple_routing_events(self) -> None:
        """Fork routes to multiple destinations."""
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        gate = recorder.register_node(
            run_id=run.run_id,
            plugin_name="gate",
            node_type="gate",
            plugin_version="1.0",
            config={},
        )
        sink_a = recorder.register_node(
            run_id=run.run_id,
            plugin_name="sink_a",
            node_type="sink",
            plugin_version="1.0",
            config={},
        )
        sink_b = recorder.register_node(
            run_id=run.run_id,
            plugin_name="sink_b",
            node_type="sink",
            plugin_version="1.0",
            config={},
        )
        edge_a = recorder.register_edge(
            run_id=run.run_id,
            from_node_id=gate.node_id,
            to_node_id=sink_a.node_id,
            label="stats",
            mode="copy",
        )
        edge_b = recorder.register_edge(
            run_id=run.run_id,
            from_node_id=gate.node_id,
            to_node_id=sink_b.node_id,
            label="archive",
            mode="copy",
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=gate.node_id,
            row_index=0,
            data={},
        )
        token = recorder.create_token(row_id=row.row_id)
        state = recorder.begin_node_state(
            token_id=token.token_id,
            node_id=gate.node_id,
            step_index=0,
            input_data={},
        )

        # Record fork to both destinations
        events = recorder.record_routing_events(
            state_id=state.state_id,
            routes=[
                {"edge_id": edge_a.edge_id, "mode": "copy"},
                {"edge_id": edge_b.edge_id, "mode": "copy"},
            ],
            reason={"action": "fork"},
        )

        assert len(events) == 2
        # All events share the same routing_group_id
        assert events[0].routing_group_id == events[1].routing_group_id
```

### Step 2: Run test to verify it fails

Run: `pytest tests/core/landscape/test_recorder.py::TestLandscapeRecorderRouting -v`
Expected: FAIL

### Step 3: Add routing event recording

```python
# Add to src/elspeth/core/landscape/recorder.py

from dataclasses import dataclass


@dataclass
class RoutingEvent:
    """A routing decision recorded in Landscape."""

    event_id: str
    state_id: str
    edge_id: str
    routing_group_id: str
    ordinal: int
    mode: str
    reason_hash: str | None
    reason_ref: str | None
    created_at: datetime


# Add import
from elspeth.core.landscape.schema import routing_events_table

# Add to LandscapeRecorder class:

    # === Routing Events ===

    def record_routing_event(
        self,
        state_id: str,
        edge_id: str,
        mode: str,
        reason: dict[str, Any] | None = None,
        *,
        event_id: str | None = None,
        routing_group_id: str | None = None,
        ordinal: int = 0,
        reason_ref: str | None = None,
    ) -> RoutingEvent:
        """Record a single routing event.

        Args:
            state_id: Node state that made the routing decision
            edge_id: Edge that was taken
            mode: Routing mode (move or copy)
            reason: Reason for this routing decision
            event_id: Optional event ID
            routing_group_id: Group ID (for multi-destination routing)
            ordinal: Position in routing group
            reason_ref: Optional payload store reference

        Returns:
            RoutingEvent model
        """
        event_id = event_id or _generate_id()
        routing_group_id = routing_group_id or _generate_id()
        reason_hash = stable_hash(reason) if reason else None
        now = _now()

        event = RoutingEvent(
            event_id=event_id,
            state_id=state_id,
            edge_id=edge_id,
            routing_group_id=routing_group_id,
            ordinal=ordinal,
            mode=mode,
            reason_hash=reason_hash,
            reason_ref=reason_ref,
            created_at=now,
        )

        with self._db.connection() as conn:
            conn.execute(
                routing_events_table.insert().values(
                    event_id=event.event_id,
                    state_id=event.state_id,
                    edge_id=event.edge_id,
                    routing_group_id=event.routing_group_id,
                    ordinal=event.ordinal,
                    mode=event.mode,
                    reason_hash=event.reason_hash,
                    reason_ref=event.reason_ref,
                    created_at=event.created_at,
                )
            )

        return event

    def record_routing_events(
        self,
        state_id: str,
        routes: list[dict[str, str]],
        reason: dict[str, Any] | None = None,
    ) -> list[RoutingEvent]:
        """Record multiple routing events (fork/multi-destination).

        All events share the same routing_group_id.

        Args:
            state_id: Node state that made the routing decision
            routes: List of {"edge_id": str, "mode": str}
            reason: Shared reason for all routes

        Returns:
            List of RoutingEvent models
        """
        routing_group_id = _generate_id()
        reason_hash = stable_hash(reason) if reason else None
        now = _now()
        events = []

        with self._db.connection() as conn:
            for ordinal, route in enumerate(routes):
                event_id = _generate_id()
                event = RoutingEvent(
                    event_id=event_id,
                    state_id=state_id,
                    edge_id=route["edge_id"],
                    routing_group_id=routing_group_id,
                    ordinal=ordinal,
                    mode=route["mode"],
                    reason_hash=reason_hash,
                    reason_ref=None,
                    created_at=now,
                )

                conn.execute(
                    routing_events_table.insert().values(
                        event_id=event.event_id,
                        state_id=event.state_id,
                        edge_id=event.edge_id,
                        routing_group_id=event.routing_group_id,
                        ordinal=event.ordinal,
                        mode=event.mode,
                        reason_hash=event.reason_hash,
                        created_at=event.created_at,
                    )
                )

                events.append(event)

        return events
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/core/landscape/test_recorder.py::TestLandscapeRecorderRouting -v`
Expected: PASS

### Step 5: Commit

```bash
git add -u
git commit -m "feat(landscape): add routing event recording to LandscapeRecorder"
```

---

## Task 8: LandscapeRecorder - Batch Management

**Files:**
- Modify: `src/elspeth/core/landscape/recorder.py`
- Modify: `tests/core/landscape/test_recorder.py`

### Step 1: Write the failing tests

```python
# Add to tests/core/landscape/test_recorder.py

class TestLandscapeRecorderBatches:
    """Aggregation batch management."""

    def test_create_batch(self) -> None:
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        agg = recorder.register_node(
            run_id=run.run_id,
            plugin_name="sum_agg",
            node_type="aggregation",
            plugin_version="1.0",
            config={},
        )

        batch = recorder.create_batch(
            run_id=run.run_id,
            aggregation_node_id=agg.node_id,
        )

        assert batch.batch_id is not None
        assert batch.status == "draft"

    def test_add_batch_member(self) -> None:
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        agg = recorder.register_node(
            run_id=run.run_id,
            plugin_name="sum_agg",
            node_type="aggregation",
            plugin_version="1.0",
            config={},
        )
        batch = recorder.create_batch(
            run_id=run.run_id,
            aggregation_node_id=agg.node_id,
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=agg.node_id,
            row_index=0,
            data={},
        )
        token = recorder.create_token(row_id=row.row_id)

        recorder.add_batch_member(
            batch_id=batch.batch_id,
            token_id=token.token_id,
            ordinal=0,
        )

        members = recorder.get_batch_members(batch.batch_id)
        assert len(members) == 1
        assert members[0].token_id == token.token_id

    def test_update_batch_status(self) -> None:
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        agg = recorder.register_node(
            run_id=run.run_id,
            plugin_name="sum_agg",
            node_type="aggregation",
            plugin_version="1.0",
            config={},
        )
        batch = recorder.create_batch(
            run_id=run.run_id,
            aggregation_node_id=agg.node_id,
        )

        # Transition: draft -> executing -> completed
        recorder.update_batch_status(batch.batch_id, "executing")
        recorder.update_batch_status(
            batch.batch_id,
            "completed",
            trigger_reason="threshold_reached",
        )

        updated = recorder.get_batch(batch.batch_id)
        assert updated.status == "completed"
        assert updated.trigger_reason == "threshold_reached"
        assert updated.completed_at is not None

    def test_get_draft_batches(self) -> None:
        """For crash recovery - find incomplete batches."""
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        agg = recorder.register_node(
            run_id=run.run_id,
            plugin_name="sum_agg",
            node_type="aggregation",
            plugin_version="1.0",
            config={},
        )

        batch1 = recorder.create_batch(
            run_id=run.run_id,
            aggregation_node_id=agg.node_id,
        )
        batch2 = recorder.create_batch(
            run_id=run.run_id,
            aggregation_node_id=agg.node_id,
        )
        recorder.update_batch_status(batch2.batch_id, "completed")

        drafts = recorder.get_batches(run.run_id, status="draft")
        assert len(drafts) == 1
        assert drafts[0].batch_id == batch1.batch_id
```

### Step 2: Run test to verify it fails

Run: `pytest tests/core/landscape/test_recorder.py::TestLandscapeRecorderBatches -v`
Expected: FAIL

### Step 3: Add batch management

```python
# Add to src/elspeth/core/landscape/recorder.py

@dataclass
class Batch:
    """An aggregation batch."""

    batch_id: str
    run_id: str
    aggregation_node_id: str
    status: str  # draft, executing, completed, failed
    created_at: datetime
    aggregation_state_id: str | None = None
    trigger_reason: str | None = None
    attempt: int = 0
    completed_at: datetime | None = None


@dataclass
class BatchMember:
    """A token belonging to a batch."""

    batch_id: str
    token_id: str
    ordinal: int


# Add imports
from elspeth.core.landscape.schema import batch_members_table, batches_table

# Add to LandscapeRecorder class:

    # === Batch Management ===

    def create_batch(
        self,
        run_id: str,
        aggregation_node_id: str,
        *,
        batch_id: str | None = None,
    ) -> Batch:
        """Create a new aggregation batch in draft status.

        Args:
            run_id: Run this batch belongs to
            aggregation_node_id: Aggregation node processing this batch
            batch_id: Optional batch ID

        Returns:
            Batch model with status="draft"
        """
        batch_id = batch_id or _generate_id()
        now = _now()

        batch = Batch(
            batch_id=batch_id,
            run_id=run_id,
            aggregation_node_id=aggregation_node_id,
            status="draft",
            created_at=now,
        )

        with self._db.connection() as conn:
            conn.execute(
                batches_table.insert().values(
                    batch_id=batch.batch_id,
                    run_id=batch.run_id,
                    aggregation_node_id=batch.aggregation_node_id,
                    status=batch.status,
                    attempt=batch.attempt,
                    created_at=batch.created_at,
                )
            )

        return batch

    def add_batch_member(
        self,
        batch_id: str,
        token_id: str,
        ordinal: int,
    ) -> BatchMember:
        """Add a token to a batch.

        Called immediately on accept() for crash safety.

        Args:
            batch_id: Batch to add to
            token_id: Token being added
            ordinal: Position in batch

        Returns:
            BatchMember model
        """
        member = BatchMember(
            batch_id=batch_id,
            token_id=token_id,
            ordinal=ordinal,
        )

        with self._db.connection() as conn:
            conn.execute(
                batch_members_table.insert().values(
                    batch_id=member.batch_id,
                    token_id=member.token_id,
                    ordinal=member.ordinal,
                )
            )

        return member

    def update_batch_status(
        self,
        batch_id: str,
        status: str,
        *,
        trigger_reason: str | None = None,
        state_id: str | None = None,
    ) -> None:
        """Update batch status.

        Args:
            batch_id: Batch to update
            status: New status (executing, completed, failed)
            trigger_reason: Why the batch was triggered
            state_id: Node state for the flush operation
        """
        updates: dict[str, Any] = {"status": status}

        if trigger_reason:
            updates["trigger_reason"] = trigger_reason
        if state_id:
            updates["aggregation_state_id"] = state_id
        if status in ("completed", "failed"):
            updates["completed_at"] = _now()

        with self._db.connection() as conn:
            conn.execute(
                batches_table.update()
                .where(batches_table.c.batch_id == batch_id)
                .values(**updates)
            )

    def get_batch(self, batch_id: str) -> Batch | None:
        """Get a batch by ID.

        Args:
            batch_id: Batch ID to retrieve

        Returns:
            Batch model or None
        """
        with self._db.connection() as conn:
            result = conn.execute(
                select(batches_table).where(batches_table.c.batch_id == batch_id)
            )
            row = result.fetchone()

        if row is None:
            return None

        return Batch(
            batch_id=row.batch_id,
            run_id=row.run_id,
            aggregation_node_id=row.aggregation_node_id,
            status=row.status,
            created_at=row.created_at,
            aggregation_state_id=row.aggregation_state_id,
            trigger_reason=row.trigger_reason,
            attempt=row.attempt,
            completed_at=row.completed_at,
        )

    def get_batches(
        self,
        run_id: str,
        *,
        status: str | None = None,
        node_id: str | None = None,
    ) -> list[Batch]:
        """Get batches for a run.

        Args:
            run_id: Run ID
            status: Optional status filter
            node_id: Optional aggregation node filter

        Returns:
            List of Batch models
        """
        query = select(batches_table).where(batches_table.c.run_id == run_id)

        if status:
            query = query.where(batches_table.c.status == status)
        if node_id:
            query = query.where(batches_table.c.aggregation_node_id == node_id)

        with self._db.connection() as conn:
            result = conn.execute(query)
            rows = result.fetchall()

        return [
            Batch(
                batch_id=row.batch_id,
                run_id=row.run_id,
                aggregation_node_id=row.aggregation_node_id,
                status=row.status,
                created_at=row.created_at,
                aggregation_state_id=row.aggregation_state_id,
                trigger_reason=row.trigger_reason,
                attempt=row.attempt,
                completed_at=row.completed_at,
            )
            for row in rows
        ]

    def get_batch_members(self, batch_id: str) -> list[BatchMember]:
        """Get all members of a batch.

        Args:
            batch_id: Batch ID

        Returns:
            List of BatchMember models (ordered by ordinal)
        """
        with self._db.connection() as conn:
            result = conn.execute(
                select(batch_members_table)
                .where(batch_members_table.c.batch_id == batch_id)
                .order_by(batch_members_table.c.ordinal)
            )
            rows = result.fetchall()

        return [
            BatchMember(
                batch_id=row.batch_id,
                token_id=row.token_id,
                ordinal=row.ordinal,
            )
            for row in rows
        ]
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/core/landscape/test_recorder.py::TestLandscapeRecorderBatches -v`
Expected: PASS

### Step 5: Commit

```bash
git add -u
git commit -m "feat(landscape): add batch management to LandscapeRecorder"
```

---

## Task 9: LandscapeRecorder - Artifact Registration

**Files:**
- Modify: `src/elspeth/core/landscape/recorder.py`
- Modify: `tests/core/landscape/test_recorder.py`

### Step 1: Write the failing tests

```python
# Add to tests/core/landscape/test_recorder.py

class TestLandscapeRecorderArtifacts:
    """Artifact registration."""

    def test_register_artifact(self) -> None:
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        sink = recorder.register_node(
            run_id=run.run_id,
            plugin_name="csv_sink",
            node_type="sink",
            plugin_version="1.0",
            config={},
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=sink.node_id,
            row_index=0,
            data={},
        )
        token = recorder.create_token(row_id=row.row_id)
        state = recorder.begin_node_state(
            token_id=token.token_id,
            node_id=sink.node_id,
            step_index=0,
            input_data={},
        )

        artifact = recorder.register_artifact(
            run_id=run.run_id,
            state_id=state.state_id,
            sink_node_id=sink.node_id,
            artifact_type="csv",
            path="/output/result.csv",
            content_hash="abc123",
            size_bytes=1024,
        )

        assert artifact.artifact_id is not None
        assert artifact.path_or_uri == "/output/result.csv"

    def test_get_artifacts_for_run(self) -> None:
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        sink = recorder.register_node(
            run_id=run.run_id,
            plugin_name="csv_sink",
            node_type="sink",
            plugin_version="1.0",
            config={},
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=sink.node_id,
            row_index=0,
            data={},
        )
        token = recorder.create_token(row_id=row.row_id)
        state = recorder.begin_node_state(
            token_id=token.token_id,
            node_id=sink.node_id,
            step_index=0,
            input_data={},
        )

        recorder.register_artifact(
            run_id=run.run_id,
            state_id=state.state_id,
            sink_node_id=sink.node_id,
            artifact_type="csv",
            path="/output/a.csv",
            content_hash="hash1",
            size_bytes=100,
        )
        recorder.register_artifact(
            run_id=run.run_id,
            state_id=state.state_id,
            sink_node_id=sink.node_id,
            artifact_type="csv",
            path="/output/b.csv",
            content_hash="hash2",
            size_bytes=200,
        )

        artifacts = recorder.get_artifacts(run.run_id)
        assert len(artifacts) == 2
```

### Step 2: Run test to verify it fails

Run: `pytest tests/core/landscape/test_recorder.py::TestLandscapeRecorderArtifacts -v`
Expected: FAIL

### Step 3: Add artifact registration

```python
# Add import
from elspeth.core.landscape.models import Artifact
from elspeth.core.landscape.schema import artifacts_table

# Add to LandscapeRecorder class:

    # === Artifact Registration ===

    def register_artifact(
        self,
        run_id: str,
        state_id: str,
        sink_node_id: str,
        artifact_type: str,
        path: str,
        content_hash: str,
        size_bytes: int,
        *,
        artifact_id: str | None = None,
    ) -> Artifact:
        """Register an artifact produced by a sink.

        Args:
            run_id: Run that produced this artifact
            state_id: Node state that produced this artifact
            sink_node_id: Sink node that wrote the artifact
            artifact_type: Type of artifact (csv, json, etc.)
            path: File path or URI
            content_hash: Hash of artifact content
            size_bytes: Size of artifact in bytes
            artifact_id: Optional artifact ID

        Returns:
            Artifact model
        """
        artifact_id = artifact_id or _generate_id()
        now = _now()

        artifact = Artifact(
            artifact_id=artifact_id,
            run_id=run_id,
            produced_by_state_id=state_id,
            sink_node_id=sink_node_id,
            artifact_type=artifact_type,
            path_or_uri=path,
            content_hash=content_hash,
            size_bytes=size_bytes,
            created_at=now,
        )

        with self._db.connection() as conn:
            conn.execute(
                artifacts_table.insert().values(
                    artifact_id=artifact.artifact_id,
                    run_id=artifact.run_id,
                    produced_by_state_id=artifact.produced_by_state_id,
                    sink_node_id=artifact.sink_node_id,
                    artifact_type=artifact.artifact_type,
                    path_or_uri=artifact.path_or_uri,
                    content_hash=artifact.content_hash,
                    size_bytes=artifact.size_bytes,
                    created_at=artifact.created_at,
                )
            )

        return artifact

    def get_artifacts(
        self,
        run_id: str,
        *,
        sink_node_id: str | None = None,
    ) -> list[Artifact]:
        """Get artifacts for a run.

        Args:
            run_id: Run ID
            sink_node_id: Optional filter by sink

        Returns:
            List of Artifact models
        """
        query = select(artifacts_table).where(artifacts_table.c.run_id == run_id)

        if sink_node_id:
            query = query.where(artifacts_table.c.sink_node_id == sink_node_id)

        with self._db.connection() as conn:
            result = conn.execute(query)
            rows = result.fetchall()

        return [
            Artifact(
                artifact_id=row.artifact_id,
                run_id=row.run_id,
                produced_by_state_id=row.produced_by_state_id,
                sink_node_id=row.sink_node_id,
                artifact_type=row.artifact_type,
                path_or_uri=row.path_or_uri,
                content_hash=row.content_hash,
                size_bytes=row.size_bytes,
                created_at=row.created_at,
            )
            for row in rows
        ]
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/core/landscape/test_recorder.py::TestLandscapeRecorderArtifacts -v`
Expected: PASS

### Step 5: Commit

```bash
git add -u
git commit -m "feat(landscape): add artifact registration to LandscapeRecorder"
```

---

## Task 10: Landscape Module Exports

**Files:**
- Modify: `src/elspeth/core/landscape/__init__.py`
- Create: `tests/core/landscape/test_exports.py`

### Step 1: Write the failing test

```python
# tests/core/landscape/test_exports.py
"""Tests for Landscape module exports."""


class TestLandscapeExports:
    """Public API exports."""

    def test_can_import_database(self) -> None:
        from elspeth.core.landscape import LandscapeDB

        assert LandscapeDB is not None

    def test_can_import_recorder(self) -> None:
        from elspeth.core.landscape import LandscapeRecorder

        assert LandscapeRecorder is not None

    def test_can_import_models(self) -> None:
        from elspeth.core.landscape import (
            Artifact,
            Edge,
            Node,
            NodeState,
            Row,
            Run,
            Token,
        )

        assert Run is not None
        assert Node is not None

    def test_can_import_recorder_types(self) -> None:
        from elspeth.core.landscape import Batch, BatchMember, RoutingEvent

        assert Batch is not None
```

### Step 2: Run test to verify it fails

Run: `pytest tests/core/landscape/test_exports.py -v`
Expected: FAIL

### Step 3: Update module exports

```python
# src/elspeth/core/landscape/__init__.py
"""Landscape: The audit backbone for complete traceability.

This module provides the infrastructure for recording everything that happens
during pipeline execution, enabling any output to be traced to its source.

Main Classes:
    LandscapeDB: Database connection manager
    LandscapeRecorder: High-level API for audit recording

Models:
    Run, Node, Edge, Row, Token, NodeState, Artifact (from models.py)
    Batch, BatchMember, RoutingEvent (from recorder.py)

Example:
    from elspeth.core.landscape import LandscapeDB, LandscapeRecorder

    db = LandscapeDB.in_memory()
    recorder = LandscapeRecorder(db)

    run = recorder.begin_run(config={"source": "data.csv"}, canonical_version="v1")
    # ... execute pipeline ...
    recorder.complete_run(run.run_id, status="completed")
"""

from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.models import (
    Artifact,
    Call,
    Edge,
    Node,
    NodeState,
    Row,
    Run,
    Token,
    TokenParent,
)
from elspeth.core.landscape.recorder import (
    Batch,
    BatchMember,
    LandscapeRecorder,
    RoutingEvent,
)

__all__ = [
    # Database
    "LandscapeDB",
    # Recorder
    "LandscapeRecorder",
    "Batch",
    "BatchMember",
    "RoutingEvent",
    # Models
    "Artifact",
    "Call",
    "Edge",
    "Node",
    "NodeState",
    "Row",
    "Run",
    "Token",
    "TokenParent",
]
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/core/landscape/test_exports.py -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/elspeth/core/landscape/__init__.py tests/core/landscape/test_exports.py
git commit -m "feat(landscape): export public API from elspeth.core.landscape"
```

---

# END OF FIRST HALF

**Tasks 1-10 Complete:**
1. ✅ LandscapeSchema - SQLAlchemy table definitions
2. ✅ LandscapeDB - Database connection manager
3. ✅ LandscapeRecorder - Run management
4. ✅ LandscapeRecorder - Node and edge registration
5. ✅ LandscapeRecorder - Row and token creation
6. ✅ LandscapeRecorder - NodeState recording
7. ✅ LandscapeRecorder - Routing events
8. ✅ LandscapeRecorder - Batch management
9. ✅ LandscapeRecorder - Artifact registration
10. ✅ Landscape module exports

**Second Half Preview (Tasks 11-20):**
11. SpanFactory - OpenTelemetry span creation
12. TokenManager - High-level token operations
13. TransformExecutor - Wraps transform.process() with audit
14. GateExecutor - Wraps gate.evaluate() with routing recording
15. AggregationExecutor - Wraps aggregation with batch tracking
16. SinkExecutor - Wraps sink.write() with artifact recording
17. RetryManager - tenacity integration
18. RowProcessor - Main row processing orchestration
19. Orchestrator - Full run lifecycle management
20. Integration tests and verification

---

# SECOND HALF - Tasks 11-20

## Task 11: SpanFactory - OpenTelemetry Integration

**Files:**
- Create: `src/elspeth/engine/spans.py`
- Create: `tests/engine/test_spans.py`

### Step 1: Write the failing test

```python
# tests/engine/test_spans.py
"""Tests for OpenTelemetry span factory."""

import pytest


class TestSpanFactory:
    """OpenTelemetry span creation."""

    def test_create_run_span(self) -> None:
        from elspeth.engine.spans import SpanFactory

        factory = SpanFactory()  # No tracer = no-op mode

        with factory.run_span("run-001") as span:
            assert span is None  # No-op mode returns None

    def test_create_row_span(self) -> None:
        from elspeth.engine.spans import SpanFactory

        factory = SpanFactory()

        with factory.row_span("row-001", "token-001") as span:
            assert span is None

    def test_create_transform_span(self) -> None:
        from elspeth.engine.spans import SpanFactory

        factory = SpanFactory()

        with factory.transform_span("my_transform", input_hash="abc123") as span:
            assert span is None

    def test_with_tracer(self) -> None:
        """Test with actual tracer if opentelemetry available."""
        pytest.importorskip("opentelemetry")
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider

        from elspeth.engine.spans import SpanFactory

        # Set up in-memory tracer
        provider = TracerProvider()
        trace.set_tracer_provider(provider)
        tracer = trace.get_tracer("test")

        factory = SpanFactory(tracer=tracer)

        with factory.run_span("run-001") as span:
            assert span is not None
            assert span.is_recording()
```

### Step 2: Run test to verify it fails

Run: `pytest tests/engine/test_spans.py -v`
Expected: FAIL (ImportError)

### Step 3: Create spans module

```python
# src/elspeth/engine/spans.py
"""OpenTelemetry span factory for SDA Engine.

Provides structured span creation for pipeline execution.
Falls back to no-op mode when no tracer is configured.

Span Hierarchy:
    run:{run_id}
    ├── source:{source_name}
    │   └── load
    ├── row:{row_id}
    │   ├── transform:{transform_name}
    │   ├── gate:{gate_name}
    │   └── sink:{sink_name}
    └── aggregation:{agg_name}
        └── flush
"""

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    from opentelemetry.trace import Span, Tracer


class NoOpSpan:
    """No-op span for when tracing is disabled."""

    def set_attribute(self, key: str, value: Any) -> None:
        """No-op."""
        pass

    def set_status(self, status: Any) -> None:
        """No-op."""
        pass

    def record_exception(self, exception: Exception) -> None:
        """No-op."""
        pass

    def is_recording(self) -> bool:
        """Always False for no-op."""
        return False


class SpanFactory:
    """Factory for creating OpenTelemetry spans.

    When no tracer is provided, all span methods return no-op contexts.

    Example:
        factory = SpanFactory(tracer=opentelemetry.trace.get_tracer("elspeth"))

        with factory.run_span("run-001") as span:
            with factory.row_span("row-001", "token-001") as row_span:
                with factory.transform_span("my_transform") as transform_span:
                    # Do work
                    pass
    """

    def __init__(self, tracer: "Tracer | None" = None) -> None:
        """Initialize with optional tracer.

        Args:
            tracer: OpenTelemetry tracer. If None, spans are no-ops.
        """
        self._tracer = tracer

    @property
    def enabled(self) -> bool:
        """Whether tracing is enabled."""
        return self._tracer is not None

    @contextmanager
    def run_span(self, run_id: str) -> Iterator["Span | None"]:
        """Create a span for the entire run.

        Args:
            run_id: Run identifier

        Yields:
            Span or None if tracing disabled
        """
        if self._tracer is None:
            yield None
            return

        with self._tracer.start_as_current_span(f"run:{run_id}") as span:
            span.set_attribute("run.id", run_id)
            yield span

    @contextmanager
    def source_span(self, source_name: str) -> Iterator["Span | None"]:
        """Create a span for source loading.

        Args:
            source_name: Name of the source plugin

        Yields:
            Span or None
        """
        if self._tracer is None:
            yield None
            return

        with self._tracer.start_as_current_span(f"source:{source_name}") as span:
            span.set_attribute("plugin.name", source_name)
            span.set_attribute("plugin.type", "source")
            yield span

    @contextmanager
    def row_span(
        self,
        row_id: str,
        token_id: str,
    ) -> Iterator["Span | None"]:
        """Create a span for processing a row.

        Args:
            row_id: Row identifier
            token_id: Token identifier

        Yields:
            Span or None
        """
        if self._tracer is None:
            yield None
            return

        with self._tracer.start_as_current_span(f"row:{row_id}") as span:
            span.set_attribute("row.id", row_id)
            span.set_attribute("token.id", token_id)
            yield span

    @contextmanager
    def transform_span(
        self,
        transform_name: str,
        *,
        input_hash: str | None = None,
    ) -> Iterator["Span | None"]:
        """Create a span for a transform operation.

        Args:
            transform_name: Name of the transform plugin
            input_hash: Optional input data hash

        Yields:
            Span or None
        """
        if self._tracer is None:
            yield None
            return

        with self._tracer.start_as_current_span(f"transform:{transform_name}") as span:
            span.set_attribute("plugin.name", transform_name)
            span.set_attribute("plugin.type", "transform")
            if input_hash:
                span.set_attribute("input.hash", input_hash)
            yield span

    @contextmanager
    def gate_span(
        self,
        gate_name: str,
        *,
        input_hash: str | None = None,
    ) -> Iterator["Span | None"]:
        """Create a span for a gate operation.

        Args:
            gate_name: Name of the gate plugin
            input_hash: Optional input data hash

        Yields:
            Span or None
        """
        if self._tracer is None:
            yield None
            return

        with self._tracer.start_as_current_span(f"gate:{gate_name}") as span:
            span.set_attribute("plugin.name", gate_name)
            span.set_attribute("plugin.type", "gate")
            if input_hash:
                span.set_attribute("input.hash", input_hash)
            yield span

    @contextmanager
    def aggregation_span(
        self,
        aggregation_name: str,
        *,
        batch_id: str | None = None,
    ) -> Iterator["Span | None"]:
        """Create a span for an aggregation flush.

        Args:
            aggregation_name: Name of the aggregation plugin
            batch_id: Optional batch identifier

        Yields:
            Span or None
        """
        if self._tracer is None:
            yield None
            return

        with self._tracer.start_as_current_span(f"aggregation:{aggregation_name}") as span:
            span.set_attribute("plugin.name", aggregation_name)
            span.set_attribute("plugin.type", "aggregation")
            if batch_id:
                span.set_attribute("batch.id", batch_id)
            yield span

    @contextmanager
    def sink_span(
        self,
        sink_name: str,
    ) -> Iterator["Span | None"]:
        """Create a span for a sink write.

        Args:
            sink_name: Name of the sink plugin

        Yields:
            Span or None
        """
        if self._tracer is None:
            yield None
            return

        with self._tracer.start_as_current_span(f"sink:{sink_name}") as span:
            span.set_attribute("plugin.name", sink_name)
            span.set_attribute("plugin.type", "sink")
            yield span
```

### Step 4: Create engine directory

```bash
mkdir -p src/elspeth/engine tests/engine
touch src/elspeth/engine/__init__.py tests/engine/__init__.py
```

### Step 5: Run tests to verify they pass

Run: `pytest tests/engine/test_spans.py -v`
Expected: PASS

### Step 6: Commit

```bash
git add src/elspeth/engine/ tests/engine/
git commit -m "feat(engine): add SpanFactory for OpenTelemetry integration"
```

---

## Task 12: TokenManager - High-Level Token Operations

**Files:**
- Create: `src/elspeth/engine/tokens.py`
- Create: `tests/engine/test_tokens.py`

### Step 1: Write the failing test

```python
# tests/engine/test_tokens.py
"""Tests for TokenManager."""

import pytest


class TestTokenManager:
    """High-level token management."""

    def test_create_initial_token(self) -> None:
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.tokens import TokenManager

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type="source",
            plugin_version="1.0",
            config={},
        )

        manager = TokenManager(recorder)

        token_info = manager.create_initial_token(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            row_data={"value": 42},
        )

        assert token_info.row_id is not None
        assert token_info.token_id is not None
        assert token_info.row_data == {"value": 42}

    def test_fork_token(self) -> None:
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.tokens import TokenManager

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type="source",
            plugin_version="1.0",
            config={},
        )

        manager = TokenManager(recorder)
        initial = manager.create_initial_token(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            row_data={"value": 42},
        )

        children = manager.fork_token(
            parent=initial,
            branches=["stats", "classifier"],
        )

        assert len(children) == 2
        assert children[0].branch_name == "stats"
        assert children[1].branch_name == "classifier"
        # Children inherit row_data
        assert children[0].row_data == {"value": 42}

    def test_update_row_data(self) -> None:
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.tokens import TokenManager

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type="source",
            plugin_version="1.0",
            config={},
        )

        manager = TokenManager(recorder)
        token_info = manager.create_initial_token(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            row_data={"x": 1},
        )

        updated = manager.update_row_data(
            token_info,
            new_data={"x": 1, "y": 2},
        )

        assert updated.row_data == {"x": 1, "y": 2}
        assert updated.token_id == token_info.token_id  # Same token
```

### Step 2: Run test to verify it fails

Run: `pytest tests/engine/test_tokens.py -v`
Expected: FAIL (ImportError)

### Step 3: Create tokens module

```python
# src/elspeth/engine/tokens.py
"""TokenManager: High-level token operations for the SDA engine.

Provides a simplified interface over LandscapeRecorder for managing
tokens (row instances flowing through the DAG).
"""

from dataclasses import dataclass, field
from typing import Any

from elspeth.core.landscape import LandscapeRecorder


@dataclass
class TokenInfo:
    """Information about a token in flight.

    Carries both identity (IDs) and current state (row_data).
    """

    row_id: str
    token_id: str
    row_data: dict[str, Any]
    branch_name: str | None = None
    step_index: int = 0


class TokenManager:
    """Manages token lifecycle for the SDA engine.

    Provides high-level operations:
    - Create initial token from source row
    - Fork token to multiple branches
    - Coalesce tokens from branches
    - Update token row data after transforms

    Example:
        manager = TokenManager(recorder)

        # Create token for source row
        token = manager.create_initial_token(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            row_data={"value": 42},
        )

        # After transform
        token = manager.update_row_data(token, {"value": 42, "processed": True})

        # Fork to branches
        children = manager.fork_token(token, ["stats", "classifier"])
    """

    def __init__(self, recorder: LandscapeRecorder) -> None:
        """Initialize with recorder.

        Args:
            recorder: LandscapeRecorder for audit trail
        """
        self._recorder = recorder

    def create_initial_token(
        self,
        run_id: str,
        source_node_id: str,
        row_index: int,
        row_data: dict[str, Any],
    ) -> TokenInfo:
        """Create a token for a source row.

        Args:
            run_id: Run identifier
            source_node_id: Source node that loaded the row
            row_index: Position in source (0-indexed)
            row_data: Row data from source

        Returns:
            TokenInfo with row and token IDs
        """
        # Create row record
        row = self._recorder.create_row(
            run_id=run_id,
            source_node_id=source_node_id,
            row_index=row_index,
            data=row_data,
        )

        # Create initial token
        token = self._recorder.create_token(row_id=row.row_id)

        return TokenInfo(
            row_id=row.row_id,
            token_id=token.token_id,
            row_data=row_data,
        )

    def fork_token(
        self,
        parent: TokenInfo,
        branches: list[str],
    ) -> list[TokenInfo]:
        """Fork a token to multiple branches.

        Args:
            parent: Parent token to fork
            branches: List of branch names

        Returns:
            List of child TokenInfo, one per branch
        """
        children = self._recorder.fork_token(
            parent_token_id=parent.token_id,
            row_id=parent.row_id,
            branches=branches,
        )

        return [
            TokenInfo(
                row_id=parent.row_id,
                token_id=child.token_id,
                row_data=parent.row_data.copy(),
                branch_name=child.branch_name,
                step_index=parent.step_index,
            )
            for child in children
        ]

    def coalesce_tokens(
        self,
        parents: list[TokenInfo],
        merged_data: dict[str, Any],
    ) -> TokenInfo:
        """Coalesce multiple tokens into one.

        Args:
            parents: Parent tokens to merge
            merged_data: Merged row data

        Returns:
            Merged TokenInfo
        """
        # Use first parent's row_id (they should all be the same)
        row_id = parents[0].row_id

        merged = self._recorder.coalesce_tokens(
            parent_token_ids=[p.token_id for p in parents],
            row_id=row_id,
        )

        # Step index is max of parents + 1
        max_step = max(p.step_index for p in parents)

        return TokenInfo(
            row_id=row_id,
            token_id=merged.token_id,
            row_data=merged_data,
            step_index=max_step + 1,
        )

    def update_row_data(
        self,
        token: TokenInfo,
        new_data: dict[str, Any],
    ) -> TokenInfo:
        """Update token's row data after a transform.

        Args:
            token: Token to update
            new_data: New row data

        Returns:
            Updated TokenInfo (same token_id, new row_data)
        """
        return TokenInfo(
            row_id=token.row_id,
            token_id=token.token_id,
            row_data=new_data,
            branch_name=token.branch_name,
            step_index=token.step_index,
        )

    def advance_step(self, token: TokenInfo) -> TokenInfo:
        """Increment token's step index.

        Args:
            token: Token to advance

        Returns:
            Updated TokenInfo with incremented step_index
        """
        return TokenInfo(
            row_id=token.row_id,
            token_id=token.token_id,
            row_data=token.row_data,
            branch_name=token.branch_name,
            step_index=token.step_index + 1,
        )
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/engine/test_tokens.py -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/elspeth/engine/tokens.py tests/engine/test_tokens.py
git commit -m "feat(engine): add TokenManager for high-level token operations"
```

---

## Task 13: TransformExecutor - Audit-Wrapped Transform Execution

**Files:**
- Create: `src/elspeth/engine/executors.py`
- Create: `tests/engine/test_executors.py`

### Step 1: Write the failing test

```python
# tests/engine/test_executors.py
"""Tests for plugin executors."""

import pytest


class TestTransformExecutor:
    """Transform execution with audit."""

    def test_execute_transform_success(self) -> None:
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import TransformExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.engine.tokens import TokenInfo, TokenManager
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import TransformResult

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="double",
            node_type="transform",
            plugin_version="1.0",
            config={},
        )

        # Mock transform plugin
        class DoubleTransform:
            name = "double"
            node_id = node.node_id

            def process(self, row: dict, ctx: PluginContext) -> TransformResult:
                return TransformResult.success({"value": row["value"] * 2})

        transform = DoubleTransform()
        ctx = PluginContext(run_id=run.run_id, config={})
        executor = TransformExecutor(recorder, SpanFactory())

        token = TokenInfo(
            row_id="row-1",
            token_id="token-1",
            row_data={"value": 21},
        )

        # Need to create row/token in landscape first
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=node.node_id,
            row_index=0,
            data=token.row_data,
            row_id=token.row_id,
        )
        recorder.create_token(row_id=row.row_id, token_id=token.token_id)

        result, updated_token = executor.execute_transform(
            transform=transform,
            token=token,
            ctx=ctx,
        )

        assert result.status == "success"
        assert result.row == {"value": 42}
        # Audit fields populated
        assert result.input_hash is not None
        assert result.output_hash is not None
        assert result.duration_ms is not None

    def test_execute_transform_error(self) -> None:
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import TransformExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.engine.tokens import TokenInfo
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import TransformResult

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="failing",
            node_type="transform",
            plugin_version="1.0",
            config={},
        )

        class FailingTransform:
            name = "failing"
            node_id = node.node_id

            def process(self, row: dict, ctx: PluginContext) -> TransformResult:
                return TransformResult.error({"message": "validation failed"})

        transform = FailingTransform()
        ctx = PluginContext(run_id=run.run_id, config={})
        executor = TransformExecutor(recorder, SpanFactory())

        token = TokenInfo(
            row_id="row-1",
            token_id="token-1",
            row_data={"value": -1},
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=node.node_id,
            row_index=0,
            data=token.row_data,
            row_id=token.row_id,
        )
        recorder.create_token(row_id=row.row_id, token_id=token.token_id)

        result, _ = executor.execute_transform(
            transform=transform,
            token=token,
            ctx=ctx,
        )

        assert result.status == "error"
        assert result.reason == {"message": "validation failed"}
```

### Step 2: Run test to verify it fails

Run: `pytest tests/engine/test_executors.py::TestTransformExecutor -v`
Expected: FAIL (ImportError)

### Step 3: Create executors module

```python
# src/elspeth/engine/executors.py
"""Plugin executors that wrap plugin calls with audit recording.

Each executor handles a specific plugin type:
- TransformExecutor: Row transforms
- GateExecutor: Routing gates
- AggregationExecutor: Stateful aggregations
- SinkExecutor: Output sinks
"""

import time
from typing import Any, Protocol

from elspeth.core.canonical import stable_hash
from elspeth.core.landscape import LandscapeRecorder
from elspeth.engine.spans import SpanFactory
from elspeth.engine.tokens import TokenInfo
from elspeth.plugins.context import PluginContext
from elspeth.plugins.results import TransformResult


class TransformLike(Protocol):
    """Protocol for transform-like plugins."""

    name: str
    node_id: str

    def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
        """Process a row."""
        ...


class TransformExecutor:
    """Executes transforms with audit recording.

    Wraps transform.process() to:
    1. Record node state start
    2. Time the operation
    3. Populate audit fields in result
    4. Record node state completion
    5. Emit OpenTelemetry span

    Example:
        executor = TransformExecutor(recorder, span_factory)
        result, updated_token = executor.execute_transform(
            transform=my_transform,
            token=token,
            ctx=ctx,
        )
    """

    def __init__(
        self,
        recorder: LandscapeRecorder,
        span_factory: SpanFactory,
    ) -> None:
        """Initialize executor.

        Args:
            recorder: Landscape recorder for audit trail
            span_factory: Span factory for tracing
        """
        self._recorder = recorder
        self._spans = span_factory

    def execute_transform(
        self,
        transform: TransformLike,
        token: TokenInfo,
        ctx: PluginContext,
    ) -> tuple[TransformResult, TokenInfo]:
        """Execute a transform with full audit recording.

        Args:
            transform: Transform plugin to execute
            token: Current token with row data
            ctx: Plugin context

        Returns:
            Tuple of (TransformResult with audit fields, updated TokenInfo)
        """
        input_hash = stable_hash(token.row_data)

        # Begin node state
        state = self._recorder.begin_node_state(
            token_id=token.token_id,
            node_id=transform.node_id,
            step_index=token.step_index,
            input_data=token.row_data,
        )

        # Execute with timing and span
        with self._spans.transform_span(transform.name, input_hash=input_hash):
            start = time.perf_counter()
            try:
                result = transform.process(token.row_data, ctx)
                duration_ms = (time.perf_counter() - start) * 1000
            except Exception as e:
                duration_ms = (time.perf_counter() - start) * 1000
                # Record failure
                self._recorder.complete_node_state(
                    state_id=state.state_id,
                    status="failed",
                    duration_ms=duration_ms,
                    error={"exception": str(e), "type": type(e).__name__},
                )
                raise

        # Populate audit fields
        result.input_hash = input_hash
        result.output_hash = stable_hash(result.row) if result.row else None
        result.duration_ms = duration_ms

        # Complete node state
        if result.status == "success":
            self._recorder.complete_node_state(
                state_id=state.state_id,
                status="completed",
                output_data=result.row,
                duration_ms=duration_ms,
            )
            # Update token with new row data
            updated_token = TokenInfo(
                row_id=token.row_id,
                token_id=token.token_id,
                row_data=result.row,
                branch_name=token.branch_name,
                step_index=token.step_index + 1,
            )
        else:
            self._recorder.complete_node_state(
                state_id=state.state_id,
                status="failed",
                duration_ms=duration_ms,
                error=result.reason,
            )
            updated_token = token

        return result, updated_token
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/engine/test_executors.py::TestTransformExecutor -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/elspeth/engine/executors.py tests/engine/test_executors.py
git commit -m "feat(engine): add TransformExecutor with audit recording"
```

---

## Task 14: GateExecutor - Routing with Audit

**Files:**
- Modify: `src/elspeth/engine/executors.py`
- Modify: `tests/engine/test_executors.py`

### Step 1: Write the failing tests

```python
# Add to tests/engine/test_executors.py

class TestGateExecutor:
    """Gate execution with routing audit."""

    def test_execute_gate_continue(self) -> None:
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import GateExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.engine.tokens import TokenInfo
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import GateResult, RoutingAction

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        gate_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="threshold",
            node_type="gate",
            plugin_version="1.0",
            config={},
        )

        class ThresholdGate:
            name = "threshold"
            node_id = gate_node.node_id

            def evaluate(self, row: dict, ctx: PluginContext) -> GateResult:
                if row["value"] < 100:
                    return GateResult(row=row, action=RoutingAction.continue_())
                return GateResult(
                    row=row,
                    action=RoutingAction.route_to_sink("high_values"),
                )

        gate = ThresholdGate()
        ctx = PluginContext(run_id=run.run_id, config={})
        executor = GateExecutor(recorder, SpanFactory())

        token = TokenInfo(
            row_id="row-1",
            token_id="token-1",
            row_data={"value": 50},
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=gate_node.node_id,
            row_index=0,
            data=token.row_data,
            row_id=token.row_id,
        )
        recorder.create_token(row_id=row.row_id, token_id=token.token_id)

        result, _ = executor.execute_gate(gate=gate, token=token, ctx=ctx)

        assert result.action.kind == "continue"

    def test_execute_gate_route(self) -> None:
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import GateExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.engine.tokens import TokenInfo
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import GateResult, RoutingAction

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        gate_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="threshold",
            node_type="gate",
            plugin_version="1.0",
            config={},
        )
        sink_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="high_values",
            node_type="sink",
            plugin_version="1.0",
            config={},
        )
        edge = recorder.register_edge(
            run_id=run.run_id,
            from_node_id=gate_node.node_id,
            to_node_id=sink_node.node_id,
            label="high_values",
            mode="move",
        )

        class ThresholdGate:
            name = "threshold"
            node_id = gate_node.node_id

            def evaluate(self, row: dict, ctx: PluginContext) -> GateResult:
                return GateResult(
                    row=row,
                    action=RoutingAction.route_to_sink(
                        "high_values",
                        reason={"value": row["value"]},
                    ),
                )

        gate = ThresholdGate()
        ctx = PluginContext(run_id=run.run_id, config={})
        executor = GateExecutor(recorder, SpanFactory(), edge_map={
            (gate_node.node_id, "high_values"): edge.edge_id,
        })

        token = TokenInfo(
            row_id="row-1",
            token_id="token-1",
            row_data={"value": 200},
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=gate_node.node_id,
            row_index=0,
            data=token.row_data,
            row_id=token.row_id,
        )
        recorder.create_token(row_id=row.row_id, token_id=token.token_id)

        result, _ = executor.execute_gate(gate=gate, token=token, ctx=ctx)

        assert result.action.kind == "route_to_sink"
        assert result.action.destinations == ("high_values",)
```

### Step 2: Run test to verify it fails

Run: `pytest tests/engine/test_executors.py::TestGateExecutor -v`
Expected: FAIL

### Step 3: Add GateExecutor

```python
# Add to src/elspeth/engine/executors.py

from elspeth.plugins.results import GateResult, RoutingAction


class GateLike(Protocol):
    """Protocol for gate-like plugins."""

    name: str
    node_id: str

    def evaluate(self, row: dict[str, Any], ctx: PluginContext) -> GateResult:
        """Evaluate routing decision."""
        ...


class GateExecutor:
    """Executes gates with routing audit recording.

    Wraps gate.evaluate() to:
    1. Record node state start
    2. Time the operation
    3. Record routing events
    4. Record node state completion
    5. Emit OpenTelemetry span

    Example:
        executor = GateExecutor(recorder, span_factory, edge_map)
        result, updated_token = executor.execute_gate(
            gate=my_gate,
            token=token,
            ctx=ctx,
        )
    """

    def __init__(
        self,
        recorder: LandscapeRecorder,
        span_factory: SpanFactory,
        edge_map: dict[tuple[str, str], str] | None = None,
    ) -> None:
        """Initialize executor.

        Args:
            recorder: Landscape recorder for audit trail
            span_factory: Span factory for tracing
            edge_map: Map of (node_id, label) -> edge_id
        """
        self._recorder = recorder
        self._spans = span_factory
        self._edge_map = edge_map or {}

    def execute_gate(
        self,
        gate: GateLike,
        token: TokenInfo,
        ctx: PluginContext,
    ) -> tuple[GateResult, TokenInfo]:
        """Execute a gate with full audit recording.

        Args:
            gate: Gate plugin to execute
            token: Current token with row data
            ctx: Plugin context

        Returns:
            Tuple of (GateResult with audit fields, updated TokenInfo)
        """
        input_hash = stable_hash(token.row_data)

        # Begin node state
        state = self._recorder.begin_node_state(
            token_id=token.token_id,
            node_id=gate.node_id,
            step_index=token.step_index,
            input_data=token.row_data,
        )

        # Execute with timing and span
        with self._spans.gate_span(gate.name, input_hash=input_hash):
            start = time.perf_counter()
            try:
                result = gate.evaluate(token.row_data, ctx)
                duration_ms = (time.perf_counter() - start) * 1000
            except Exception as e:
                duration_ms = (time.perf_counter() - start) * 1000
                self._recorder.complete_node_state(
                    state_id=state.state_id,
                    status="failed",
                    duration_ms=duration_ms,
                    error={"exception": str(e), "type": type(e).__name__},
                )
                raise

        # Populate audit fields
        result.input_hash = input_hash
        result.output_hash = stable_hash(result.row)
        result.duration_ms = duration_ms

        # Record routing event(s)
        if result.action.kind != "continue":
            self._record_routing(state.state_id, gate.node_id, result.action)

        # Complete node state
        self._recorder.complete_node_state(
            state_id=state.state_id,
            status="completed",
            output_data=result.row,
            duration_ms=duration_ms,
        )

        updated_token = TokenInfo(
            row_id=token.row_id,
            token_id=token.token_id,
            row_data=result.row,
            branch_name=token.branch_name,
            step_index=token.step_index + 1,
        )

        return result, updated_token

    def _record_routing(
        self,
        state_id: str,
        node_id: str,
        action: RoutingAction,
    ) -> None:
        """Record routing events for a routing action."""
        if len(action.destinations) == 1:
            # Single destination
            edge_id = self._edge_map.get((node_id, action.destinations[0]))
            if edge_id:
                self._recorder.record_routing_event(
                    state_id=state_id,
                    edge_id=edge_id,
                    mode=action.mode,
                    reason=action.reason,
                )
        else:
            # Multiple destinations (fork)
            routes = []
            for dest in action.destinations:
                edge_id = self._edge_map.get((node_id, dest))
                if edge_id:
                    routes.append({"edge_id": edge_id, "mode": action.mode})

            if routes:
                self._recorder.record_routing_events(
                    state_id=state_id,
                    routes=routes,
                    reason=action.reason,
                )
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/engine/test_executors.py::TestGateExecutor -v`
Expected: PASS

### Step 5: Commit

```bash
git add -u
git commit -m "feat(engine): add GateExecutor with routing audit"
```

---

## Task 15: AggregationExecutor - Batch Tracking

**Files:**
- Modify: `src/elspeth/engine/executors.py`
- Modify: `tests/engine/test_executors.py`

### Step 1: Write the failing tests

```python
# Add to tests/engine/test_executors.py

class TestAggregationExecutor:
    """Aggregation execution with batch tracking."""

    def test_accept_creates_batch(self) -> None:
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import AggregationExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.engine.tokens import TokenInfo
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import AcceptResult

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        agg_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="sum",
            node_type="aggregation",
            plugin_version="1.0",
            config={},
        )

        class SumAggregation:
            name = "sum"
            node_id = agg_node.node_id
            _batch_id = None
            _count = 0

            def accept(self, row: dict, ctx: PluginContext) -> AcceptResult:
                self._count += 1
                return AcceptResult(accepted=True, trigger=self._count >= 2)

            def flush(self, ctx: PluginContext) -> list[dict]:
                return [{"sum": 100}]

        agg = SumAggregation()
        ctx = PluginContext(run_id=run.run_id, config={})
        executor = AggregationExecutor(recorder, SpanFactory(), run.run_id)

        token = TokenInfo(
            row_id="row-1",
            token_id="token-1",
            row_data={"value": 50},
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=agg_node.node_id,
            row_index=0,
            data=token.row_data,
            row_id=token.row_id,
        )
        recorder.create_token(row_id=row.row_id, token_id=token.token_id)

        result = executor.accept(aggregation=agg, token=token, ctx=ctx)

        assert result.accepted is True
        assert result.batch_id is not None  # Batch created

    def test_flush_with_audit(self) -> None:
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import AggregationExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.engine.tokens import TokenInfo
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import AcceptResult

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        agg_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="sum",
            node_type="aggregation",
            plugin_version="1.0",
            config={},
        )

        class SumAggregation:
            name = "sum"
            node_id = agg_node.node_id
            _batch_id = None
            _buffer = []

            def accept(self, row: dict, ctx: PluginContext) -> AcceptResult:
                self._buffer.append(row["value"])
                return AcceptResult(accepted=True, trigger=len(self._buffer) >= 2)

            def flush(self, ctx: PluginContext) -> list[dict]:
                result = [{"sum": sum(self._buffer)}]
                self._buffer = []
                return result

        agg = SumAggregation()
        ctx = PluginContext(run_id=run.run_id, config={})
        executor = AggregationExecutor(recorder, SpanFactory(), run.run_id)

        # Accept two rows
        for i in range(2):
            token = TokenInfo(
                row_id=f"row-{i}",
                token_id=f"token-{i}",
                row_data={"value": 50},
            )
            row = recorder.create_row(
                run_id=run.run_id,
                source_node_id=agg_node.node_id,
                row_index=i,
                data=token.row_data,
                row_id=token.row_id,
            )
            recorder.create_token(row_id=row.row_id, token_id=token.token_id)
            result = executor.accept(aggregation=agg, token=token, ctx=ctx)

        # Flush
        outputs = executor.flush(aggregation=agg, ctx=ctx, trigger_reason="threshold")

        assert len(outputs) == 1
        assert outputs[0] == {"sum": 100}

        # Batch should be completed
        batch = recorder.get_batch(result.batch_id)
        assert batch.status == "completed"
```

### Step 2: Run test to verify it fails

Run: `pytest tests/engine/test_executors.py::TestAggregationExecutor -v`
Expected: FAIL

### Step 3: Add AggregationExecutor

```python
# Add to src/elspeth/engine/executors.py

from elspeth.plugins.results import AcceptResult


class AggregationLike(Protocol):
    """Protocol for aggregation-like plugins."""

    name: str
    node_id: str
    _batch_id: str | None

    def accept(self, row: dict[str, Any], ctx: PluginContext) -> AcceptResult:
        """Accept a row into the aggregation."""
        ...

    def flush(self, ctx: PluginContext) -> list[dict[str, Any]]:
        """Flush the aggregation."""
        ...


class AggregationExecutor:
    """Executes aggregations with batch tracking.

    Manages the batch lifecycle:
    1. Create draft batch on first accept()
    2. Persist batch members immediately (crash-safe)
    3. Transition to executing on flush()
    4. Transition to completed/failed after flush()

    Example:
        executor = AggregationExecutor(recorder, span_factory, run_id)

        # Accept rows
        result = executor.accept(aggregation, token, ctx)
        if result.trigger:
            outputs = executor.flush(aggregation, ctx, "threshold")
    """

    def __init__(
        self,
        recorder: LandscapeRecorder,
        span_factory: SpanFactory,
        run_id: str,
    ) -> None:
        """Initialize executor.

        Args:
            recorder: Landscape recorder for audit trail
            span_factory: Span factory for tracing
            run_id: Current run ID
        """
        self._recorder = recorder
        self._spans = span_factory
        self._run_id = run_id
        self._member_counts: dict[str, int] = {}  # batch_id -> count

    def accept(
        self,
        aggregation: AggregationLike,
        token: TokenInfo,
        ctx: PluginContext,
    ) -> AcceptResult:
        """Accept a row into an aggregation with batch tracking.

        Args:
            aggregation: Aggregation plugin
            token: Current token
            ctx: Plugin context

        Returns:
            AcceptResult with batch_id populated
        """
        # Create batch on first accept
        if aggregation._batch_id is None:
            batch = self._recorder.create_batch(
                run_id=self._run_id,
                aggregation_node_id=aggregation.node_id,
            )
            aggregation._batch_id = batch.batch_id
            self._member_counts[batch.batch_id] = 0

        # Persist membership immediately (crash-safe)
        ordinal = self._member_counts[aggregation._batch_id]
        self._recorder.add_batch_member(
            batch_id=aggregation._batch_id,
            token_id=token.token_id,
            ordinal=ordinal,
        )
        self._member_counts[aggregation._batch_id] = ordinal + 1

        # Call plugin accept
        result = aggregation.accept(token.row_data, ctx)
        result.batch_id = aggregation._batch_id

        return result

    def flush(
        self,
        aggregation: AggregationLike,
        ctx: PluginContext,
        trigger_reason: str,
    ) -> list[dict[str, Any]]:
        """Flush an aggregation with status management.

        Args:
            aggregation: Aggregation plugin
            ctx: Plugin context
            trigger_reason: Why flush was triggered

        Returns:
            List of output rows
        """
        batch_id = aggregation._batch_id
        if batch_id is None:
            return []

        # Transition to executing
        self._recorder.update_batch_status(
            batch_id,
            "executing",
            trigger_reason=trigger_reason,
        )

        with self._spans.aggregation_span(aggregation.name, batch_id=batch_id):
            try:
                outputs = aggregation.flush(ctx)

                # Transition to completed
                self._recorder.update_batch_status(batch_id, "completed")

                # Reset for next batch
                aggregation._batch_id = None
                if batch_id in self._member_counts:
                    del self._member_counts[batch_id]

                return outputs

            except Exception as e:
                self._recorder.update_batch_status(
                    batch_id,
                    "failed",
                )
                raise
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/engine/test_executors.py::TestAggregationExecutor -v`
Expected: PASS

### Step 5: Commit

```bash
git add -u
git commit -m "feat(engine): add AggregationExecutor with batch tracking"
```

---

## Task 16: SinkExecutor - Artifact Recording

**Files:**
- Modify: `src/elspeth/engine/executors.py`
- Modify: `tests/engine/test_executors.py`

### Step 1: Write the failing tests

```python
# Add to tests/engine/test_executors.py

class TestSinkExecutor:
    """Sink execution with artifact recording."""

    def test_write_records_artifact(self) -> None:
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import SinkExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.engine.tokens import TokenInfo
        from elspeth.plugins.context import PluginContext

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        sink_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="csv_sink",
            node_type="sink",
            plugin_version="1.0",
            config={},
        )

        class CSVSink:
            name = "csv_sink"
            node_id = sink_node.node_id

            def write(self, rows: list[dict], ctx: PluginContext) -> dict:
                # Return artifact info
                return {
                    "path": "/output/result.csv",
                    "size_bytes": 1024,
                    "content_hash": "abc123",
                }

        sink = CSVSink()
        ctx = PluginContext(run_id=run.run_id, config={})
        executor = SinkExecutor(recorder, SpanFactory(), run.run_id)

        token = TokenInfo(
            row_id="row-1",
            token_id="token-1",
            row_data={"value": 42},
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=sink_node.node_id,
            row_index=0,
            data=token.row_data,
            row_id=token.row_id,
        )
        recorder.create_token(row_id=row.row_id, token_id=token.token_id)

        artifact = executor.write(
            sink=sink,
            tokens=[token],
            ctx=ctx,
        )

        assert artifact is not None
        assert artifact.path_or_uri == "/output/result.csv"

        # Artifact recorded in Landscape
        artifacts = recorder.get_artifacts(run.run_id)
        assert len(artifacts) == 1
```

### Step 2: Run test to verify it fails

Run: `pytest tests/engine/test_executors.py::TestSinkExecutor -v`
Expected: FAIL

### Step 3: Add SinkExecutor

```python
# Add to src/elspeth/engine/executors.py

from elspeth.core.landscape import Artifact


class SinkLike(Protocol):
    """Protocol for sink-like plugins."""

    name: str
    node_id: str

    def write(self, rows: list[dict[str, Any]], ctx: PluginContext) -> dict[str, Any]:
        """Write rows and return artifact info."""
        ...


class SinkExecutor:
    """Executes sinks with artifact recording.

    Wraps sink.write() to:
    1. Record node state for each input token
    2. Time the operation
    3. Register resulting artifact
    4. Emit OpenTelemetry span

    Example:
        executor = SinkExecutor(recorder, span_factory, run_id)
        artifact = executor.write(sink, tokens, ctx)
    """

    def __init__(
        self,
        recorder: LandscapeRecorder,
        span_factory: SpanFactory,
        run_id: str,
    ) -> None:
        """Initialize executor.

        Args:
            recorder: Landscape recorder for audit trail
            span_factory: Span factory for tracing
            run_id: Current run ID
        """
        self._recorder = recorder
        self._spans = span_factory
        self._run_id = run_id

    def write(
        self,
        sink: SinkLike,
        tokens: list[TokenInfo],
        ctx: PluginContext,
    ) -> Artifact | None:
        """Write tokens to sink with artifact recording.

        Args:
            sink: Sink plugin
            tokens: Tokens to write
            ctx: Plugin context

        Returns:
            Artifact if produced, None otherwise
        """
        if not tokens:
            return None

        # Use first token for node state (sink processes batch)
        first_token = tokens[0]
        rows = [t.row_data for t in tokens]

        # Begin node state
        state = self._recorder.begin_node_state(
            token_id=first_token.token_id,
            node_id=sink.node_id,
            step_index=first_token.step_index,
            input_data={"rows": rows},
        )

        with self._spans.sink_span(sink.name):
            start = time.perf_counter()
            try:
                artifact_info = sink.write(rows, ctx)
                duration_ms = (time.perf_counter() - start) * 1000
            except Exception as e:
                duration_ms = (time.perf_counter() - start) * 1000
                self._recorder.complete_node_state(
                    state_id=state.state_id,
                    status="failed",
                    duration_ms=duration_ms,
                    error={"exception": str(e), "type": type(e).__name__},
                )
                raise

        # Complete node state
        self._recorder.complete_node_state(
            state_id=state.state_id,
            status="completed",
            duration_ms=duration_ms,
        )

        # Register artifact
        artifact = self._recorder.register_artifact(
            run_id=self._run_id,
            state_id=state.state_id,
            sink_node_id=sink.node_id,
            artifact_type=sink.name,
            path=artifact_info.get("path", ""),
            content_hash=artifact_info.get("content_hash", ""),
            size_bytes=artifact_info.get("size_bytes", 0),
        )

        return artifact
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/engine/test_executors.py::TestSinkExecutor -v`
Expected: PASS

### Step 5: Commit

```bash
git add -u
git commit -m "feat(engine): add SinkExecutor with artifact recording"
```

---

## Task 17: RetryManager - tenacity Integration

**Files:**
- Create: `src/elspeth/engine/retry.py`
- Create: `tests/engine/test_retry.py`

### Step 1: Write the failing test

```python
# tests/engine/test_retry.py
"""Tests for RetryManager."""

import pytest


class TestRetryManager:
    """Retry logic with tenacity."""

    def test_retry_on_retryable_error(self) -> None:
        from elspeth.engine.retry import RetryManager, RetryConfig

        manager = RetryManager(RetryConfig(max_attempts=3, base_delay=0.01))

        call_count = 0

        def flaky_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Transient error")
            return "success"

        result = manager.execute_with_retry(
            flaky_operation,
            is_retryable=lambda e: isinstance(e, ValueError),
        )

        assert result == "success"
        assert call_count == 3

    def test_no_retry_on_non_retryable(self) -> None:
        from elspeth.engine.retry import RetryManager, RetryConfig

        manager = RetryManager(RetryConfig(max_attempts=3, base_delay=0.01))

        def failing_operation():
            raise TypeError("Not retryable")

        with pytest.raises(TypeError):
            manager.execute_with_retry(
                failing_operation,
                is_retryable=lambda e: isinstance(e, ValueError),
            )

    def test_max_attempts_exceeded(self) -> None:
        from elspeth.engine.retry import RetryManager, RetryConfig, MaxRetriesExceeded

        manager = RetryManager(RetryConfig(max_attempts=2, base_delay=0.01))

        def always_fails():
            raise ValueError("Always fails")

        with pytest.raises(MaxRetriesExceeded) as exc_info:
            manager.execute_with_retry(
                always_fails,
                is_retryable=lambda e: isinstance(e, ValueError),
            )

        assert exc_info.value.attempts == 2

    def test_records_attempts(self) -> None:
        from elspeth.engine.retry import RetryManager, RetryConfig

        manager = RetryManager(RetryConfig(max_attempts=3, base_delay=0.01))
        attempts = []

        call_count = 0

        def flaky_with_tracking():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Fail")
            return "ok"

        result = manager.execute_with_retry(
            flaky_with_tracking,
            is_retryable=lambda e: isinstance(e, ValueError),
            on_retry=lambda attempt, error: attempts.append((attempt, str(error))),
        )

        assert len(attempts) == 1
        assert attempts[0][0] == 1
```

### Step 2: Run test to verify it fails

Run: `pytest tests/engine/test_retry.py -v`
Expected: FAIL (ImportError)

### Step 3: Create retry module

```python
# src/elspeth/engine/retry.py
"""RetryManager: Retry logic with tenacity integration.

Provides configurable retry behavior for transform execution:
- Exponential backoff with jitter
- Configurable max attempts
- Retryable error filtering
- Attempt tracking for Landscape
"""

from dataclasses import dataclass
from typing import Any, Callable, TypeVar

from tenacity import (
    RetryError,
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

T = TypeVar("T")


class MaxRetriesExceeded(Exception):
    """Raised when max retry attempts are exceeded."""

    def __init__(self, attempts: int, last_error: Exception) -> None:
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(f"Max retries ({attempts}) exceeded: {last_error}")


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_attempts: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    jitter: float = 1.0  # seconds

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")


class RetryManager:
    """Manages retry logic for transform execution.

    Uses tenacity for exponential backoff with jitter.
    Integrates with Landscape for attempt tracking.

    Example:
        manager = RetryManager(RetryConfig(max_attempts=3))

        result = manager.execute_with_retry(
            operation=lambda: transform.process(row, ctx),
            is_retryable=lambda e: e.retryable,
            on_retry=lambda attempt, error: recorder.record_attempt(attempt, error),
        )
    """

    def __init__(self, config: RetryConfig) -> None:
        """Initialize with config.

        Args:
            config: Retry configuration
        """
        self._config = config

    def execute_with_retry(
        self,
        operation: Callable[[], T],
        *,
        is_retryable: Callable[[Exception], bool],
        on_retry: Callable[[int, Exception], None] | None = None,
    ) -> T:
        """Execute operation with retry logic.

        Args:
            operation: Operation to execute
            is_retryable: Function to check if error is retryable
            on_retry: Optional callback on retry (attempt, error)

        Returns:
            Result of operation

        Raises:
            MaxRetriesExceeded: If max attempts exceeded
            Exception: If non-retryable error occurs
        """
        attempt = 0
        last_error: Exception | None = None

        try:
            for attempt_state in Retrying(
                stop=stop_after_attempt(self._config.max_attempts),
                wait=wait_exponential_jitter(
                    initial=self._config.base_delay,
                    max=self._config.max_delay,
                    jitter=self._config.jitter,
                ),
                retry=retry_if_exception(is_retryable),
                reraise=True,
            ):
                with attempt_state:
                    attempt = attempt_state.retry_state.attempt_number
                    try:
                        return operation()
                    except Exception as e:
                        last_error = e
                        if is_retryable(e) and on_retry:
                            on_retry(attempt, e)
                        raise

        except RetryError as e:
            raise MaxRetriesExceeded(attempt, last_error or e.last_attempt.exception())

        # Should not reach here
        raise RuntimeError("Unexpected state in retry loop")
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/engine/test_retry.py -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/elspeth/engine/retry.py tests/engine/test_retry.py
git commit -m "feat(engine): add RetryManager with tenacity integration"
```

---

## Task 18: RowProcessor - Row Processing Orchestration

**Files:**
- Create: `src/elspeth/engine/processor.py`
- Create: `tests/engine/test_processor.py`

### Step 1: Write the failing test

```python
# tests/engine/test_processor.py
"""Tests for RowProcessor."""

import pytest


class TestRowProcessor:
    """Row processing through pipeline."""

    def test_process_through_transforms(self) -> None:
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import TransformResult

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        # Register nodes
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type="source",
            plugin_version="1.0",
            config={},
        )
        transform1 = recorder.register_node(
            run_id=run.run_id,
            plugin_name="double",
            node_type="transform",
            plugin_version="1.0",
            config={},
        )
        transform2 = recorder.register_node(
            run_id=run.run_id,
            plugin_name="add_one",
            node_type="transform",
            plugin_version="1.0",
            config={},
        )

        class DoubleTransform:
            name = "double"
            node_id = transform1.node_id

            def process(self, row, ctx):
                return TransformResult.success({"value": row["value"] * 2})

        class AddOneTransform:
            name = "add_one"
            node_id = transform2.node_id

            def process(self, row, ctx):
                return TransformResult.success({"value": row["value"] + 1})

        ctx = PluginContext(run_id=run.run_id, config={})
        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            source_node_id=source.node_id,
        )

        result = processor.process_row(
            row_index=0,
            row_data={"value": 10},
            transforms=[DoubleTransform(), AddOneTransform()],
            ctx=ctx,
        )

        # 10 * 2 = 20, 20 + 1 = 21
        assert result.final_data == {"value": 21}
        assert result.outcome == "completed"
```

### Step 2: Run test to verify it fails

Run: `pytest tests/engine/test_processor.py -v`
Expected: FAIL (ImportError)

### Step 3: Create processor module

```python
# src/elspeth/engine/processor.py
"""RowProcessor: Orchestrates row processing through pipeline.

Coordinates:
- Token creation
- Transform execution
- Gate evaluation
- Aggregation handling
- Final outcome recording
"""

from dataclasses import dataclass
from typing import Any

from elspeth.core.landscape import LandscapeRecorder
from elspeth.engine.executors import (
    AggregationExecutor,
    GateExecutor,
    TransformExecutor,
)
from elspeth.engine.spans import SpanFactory
from elspeth.engine.tokens import TokenInfo, TokenManager
from elspeth.plugins.context import PluginContext


@dataclass
class RowResult:
    """Result of processing a row through the pipeline."""

    token_id: str
    row_id: str
    final_data: dict[str, Any]
    outcome: str  # completed, routed, forked, consumed, failed


class RowProcessor:
    """Processes rows through the transform pipeline.

    Handles:
    1. Creating initial tokens from source rows
    2. Executing transforms in sequence
    3. Evaluating gates for routing decisions
    4. Accepting rows into aggregations
    5. Recording final outcomes

    Example:
        processor = RowProcessor(recorder, span_factory, run_id, source_node_id)

        result = processor.process_row(
            row_index=0,
            row_data={"value": 42},
            transforms=[transform1, transform2],
            ctx=ctx,
        )
    """

    def __init__(
        self,
        recorder: LandscapeRecorder,
        span_factory: SpanFactory,
        run_id: str,
        source_node_id: str,
        *,
        edge_map: dict[tuple[str, str], str] | None = None,
    ) -> None:
        """Initialize processor.

        Args:
            recorder: Landscape recorder
            span_factory: Span factory for tracing
            run_id: Current run ID
            source_node_id: Source node ID
            edge_map: Map of (node_id, label) -> edge_id
        """
        self._recorder = recorder
        self._spans = span_factory
        self._run_id = run_id
        self._source_node_id = source_node_id

        self._token_manager = TokenManager(recorder)
        self._transform_executor = TransformExecutor(recorder, span_factory)
        self._gate_executor = GateExecutor(recorder, span_factory, edge_map)
        self._aggregation_executor = AggregationExecutor(
            recorder, span_factory, run_id
        )

    def process_row(
        self,
        row_index: int,
        row_data: dict[str, Any],
        transforms: list[Any],
        ctx: PluginContext,
    ) -> RowResult:
        """Process a row through all transforms.

        Args:
            row_index: Position in source
            row_data: Initial row data
            transforms: List of transform plugins
            ctx: Plugin context

        Returns:
            RowResult with final outcome
        """
        # Create initial token
        token = self._token_manager.create_initial_token(
            run_id=self._run_id,
            source_node_id=self._source_node_id,
            row_index=row_index,
            row_data=row_data,
        )

        with self._spans.row_span(token.row_id, token.token_id):
            current_token = token

            for transform in transforms:
                # Check transform type and execute accordingly
                if hasattr(transform, "evaluate"):
                    # Gate transform
                    result, current_token = self._gate_executor.execute_gate(
                        gate=transform,
                        token=current_token,
                        ctx=ctx,
                    )

                    if result.action.kind == "route_to_sink":
                        return RowResult(
                            token_id=current_token.token_id,
                            row_id=current_token.row_id,
                            final_data=current_token.row_data,
                            outcome="routed",
                        )
                    elif result.action.kind == "fork_to_paths":
                        return RowResult(
                            token_id=current_token.token_id,
                            row_id=current_token.row_id,
                            final_data=current_token.row_data,
                            outcome="forked",
                        )

                elif hasattr(transform, "accept"):
                    # Aggregation transform
                    accept_result = self._aggregation_executor.accept(
                        aggregation=transform,
                        token=current_token,
                        ctx=ctx,
                    )

                    if accept_result.trigger:
                        self._aggregation_executor.flush(
                            aggregation=transform,
                            ctx=ctx,
                            trigger_reason="threshold",
                        )

                    return RowResult(
                        token_id=current_token.token_id,
                        row_id=current_token.row_id,
                        final_data=current_token.row_data,
                        outcome="consumed",
                    )

                else:
                    # Regular transform
                    result, current_token = self._transform_executor.execute_transform(
                        transform=transform,
                        token=current_token,
                        ctx=ctx,
                    )

                    if result.status == "error":
                        return RowResult(
                            token_id=current_token.token_id,
                            row_id=current_token.row_id,
                            final_data=current_token.row_data,
                            outcome="failed",
                        )

            return RowResult(
                token_id=current_token.token_id,
                row_id=current_token.row_id,
                final_data=current_token.row_data,
                outcome="completed",
            )
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/engine/test_processor.py -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/elspeth/engine/processor.py tests/engine/test_processor.py
git commit -m "feat(engine): add RowProcessor for pipeline orchestration"
```

---

## Task 19: Orchestrator - Full Run Lifecycle

**Files:**
- Create: `src/elspeth/engine/orchestrator.py`
- Create: `tests/engine/test_orchestrator.py`

### Step 1: Write the failing test

```python
# tests/engine/test_orchestrator.py
"""Tests for Orchestrator."""

import pytest


class TestOrchestrator:
    """Full run orchestration."""

    def test_run_simple_pipeline(self) -> None:
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.results import TransformResult
        from elspeth.plugins.schemas import PluginSchema

        db = LandscapeDB.in_memory()

        class InputSchema(PluginSchema):
            value: int

        class OutputSchema(PluginSchema):
            value: int
            doubled: int

        class ListSource:
            name = "list_source"
            output_schema = InputSchema

            def __init__(self, data: list[dict]) -> None:
                self._data = data

            def load(self, ctx):
                yield from self._data

            def close(self):
                pass

        class DoubleTransform:
            name = "double"
            input_schema = InputSchema
            output_schema = OutputSchema

            def process(self, row, ctx):
                return TransformResult.success({
                    "value": row["value"],
                    "doubled": row["value"] * 2,
                })

        class CollectSink:
            name = "collect"
            results: list = []

            def write(self, rows, ctx):
                self.results.extend(rows)
                return {"path": "memory", "size_bytes": 0, "content_hash": ""}

        source = ListSource([{"value": 1}, {"value": 2}, {"value": 3}])
        transform = DoubleTransform()
        sink = CollectSink()

        config = PipelineConfig(
            source=source,
            transforms=[transform],
            sinks={"default": sink},
        )

        orchestrator = Orchestrator(db)
        run_result = orchestrator.run(config)

        assert run_result.status == "completed"
        assert run_result.rows_processed == 3
        assert len(sink.results) == 3
        assert sink.results[0] == {"value": 1, "doubled": 2}

    def test_run_with_gate_routing(self) -> None:
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.results import GateResult, RoutingAction, TransformResult
        from elspeth.plugins.schemas import PluginSchema

        db = LandscapeDB.in_memory()

        class RowSchema(PluginSchema):
            value: int

        class ListSource:
            name = "list_source"
            output_schema = RowSchema

            def __init__(self, data: list[dict]) -> None:
                self._data = data

            def load(self, ctx):
                yield from self._data

            def close(self):
                pass

        class ThresholdGate:
            name = "threshold"
            input_schema = RowSchema
            output_schema = RowSchema

            def evaluate(self, row, ctx):
                if row["value"] > 50:
                    return GateResult(
                        row=row,
                        action=RoutingAction.route_to_sink("high"),
                    )
                return GateResult(row=row, action=RoutingAction.continue_())

        class CollectSink:
            name = "collect"
            results: list = []

            def write(self, rows, ctx):
                self.results.extend(rows)
                return {"path": "memory", "size_bytes": 0, "content_hash": ""}

        source = ListSource([{"value": 10}, {"value": 100}, {"value": 30}])
        gate = ThresholdGate()
        default_sink = CollectSink()
        high_sink = CollectSink()

        config = PipelineConfig(
            source=source,
            transforms=[gate],
            sinks={"default": default_sink, "high": high_sink},
        )

        orchestrator = Orchestrator(db)
        run_result = orchestrator.run(config)

        assert run_result.status == "completed"
        # value=10 and value=30 go to default, value=100 goes to high
        assert len(default_sink.results) == 2
        assert len(high_sink.results) == 1
```

### Step 2: Run test to verify it fails

Run: `pytest tests/engine/test_orchestrator.py -v`
Expected: FAIL (ImportError)

### Step 3: Create orchestrator module

```python
# src/elspeth/engine/orchestrator.py
"""Orchestrator: Full run lifecycle management.

Coordinates:
- Run initialization
- Source loading
- Row processing
- Sink writing
- Run completion
"""

from dataclasses import dataclass, field
from typing import Any

from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
from elspeth.engine.processor import RowProcessor
from elspeth.engine.spans import SpanFactory
from elspeth.plugins.context import PluginContext


@dataclass
class PipelineConfig:
    """Configuration for a pipeline run."""

    source: Any  # SourceProtocol
    transforms: list[Any]  # List of transform/gate plugins
    sinks: dict[str, Any]  # sink_name -> SinkProtocol
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunResult:
    """Result of a pipeline run."""

    run_id: str
    status: str  # completed, failed
    rows_processed: int
    rows_succeeded: int
    rows_failed: int
    rows_routed: int


class Orchestrator:
    """Orchestrates full pipeline runs.

    Manages the complete lifecycle:
    1. Begin run in Landscape
    2. Register all nodes
    3. Load rows from source
    4. Process rows through transforms
    5. Write to sinks
    6. Complete run

    Example:
        orchestrator = Orchestrator(db)

        config = PipelineConfig(
            source=csv_source,
            transforms=[transform1, gate1, transform2],
            sinks={"default": csv_sink, "flagged": review_sink},
        )

        result = orchestrator.run(config)
    """

    def __init__(
        self,
        db: LandscapeDB,
        *,
        canonical_version: str = "sha256-rfc8785-v1",
    ) -> None:
        """Initialize orchestrator.

        Args:
            db: Landscape database
            canonical_version: Canonical hash version
        """
        self._db = db
        self._canonical_version = canonical_version
        self._span_factory = SpanFactory()

    def run(self, config: PipelineConfig) -> RunResult:
        """Execute a pipeline run.

        Args:
            config: Pipeline configuration

        Returns:
            RunResult with execution summary
        """
        recorder = LandscapeRecorder(self._db)

        # Begin run
        run = recorder.begin_run(
            config=config.config,
            canonical_version=self._canonical_version,
        )

        try:
            with self._span_factory.run_span(run.run_id):
                result = self._execute_run(recorder, run.run_id, config)

            # Complete run
            recorder.complete_run(run.run_id, status="completed")
            result.status = "completed"
            return result

        except Exception as e:
            recorder.complete_run(run.run_id, status="failed")
            raise

    def _execute_run(
        self,
        recorder: LandscapeRecorder,
        run_id: str,
        config: PipelineConfig,
    ) -> RunResult:
        """Execute the run (internal)."""

        # Register source node
        source_node = recorder.register_node(
            run_id=run_id,
            plugin_name=config.source.name,
            node_type="source",
            plugin_version="1.0.0",
            config={},
            sequence=0,
        )

        # Register transform nodes
        edge_map: dict[tuple[str, str], str] = {}
        prev_node_id = source_node.node_id

        for i, transform in enumerate(config.transforms):
            node_type = "gate" if hasattr(transform, "evaluate") else "transform"
            node = recorder.register_node(
                run_id=run_id,
                plugin_name=transform.name,
                node_type=node_type,
                plugin_version="1.0.0",
                config={},
                sequence=i + 1,
            )
            transform.node_id = node.node_id

            # Register continue edge
            edge = recorder.register_edge(
                run_id=run_id,
                from_node_id=prev_node_id,
                to_node_id=node.node_id,
                label="continue",
                mode="move",
            )
            edge_map[(prev_node_id, "continue")] = edge.edge_id
            prev_node_id = node.node_id

        # Register sink nodes
        sink_nodes: dict[str, Any] = {}
        for sink_name, sink in config.sinks.items():
            node = recorder.register_node(
                run_id=run_id,
                plugin_name=sink.name,
                node_type="sink",
                plugin_version="1.0.0",
                config={},
            )
            sink.node_id = node.node_id
            sink_nodes[sink_name] = node

            # Register edge from last transform to sink
            edge = recorder.register_edge(
                run_id=run_id,
                from_node_id=prev_node_id,
                to_node_id=node.node_id,
                label=sink_name,
                mode="move",
            )
            edge_map[(prev_node_id, sink_name)] = edge.edge_id

        # Create context
        ctx = PluginContext(
            run_id=run_id,
            config=config.config,
            landscape=recorder,
        )

        # Create processor
        processor = RowProcessor(
            recorder=recorder,
            span_factory=self._span_factory,
            run_id=run_id,
            source_node_id=source_node.node_id,
            edge_map=edge_map,
        )

        # Process rows
        rows_processed = 0
        rows_succeeded = 0
        rows_failed = 0
        rows_routed = 0
        pending_rows: dict[str, list[dict]] = {name: [] for name in config.sinks}

        with self._span_factory.source_span(config.source.name):
            for row_index, row_data in enumerate(config.source.load(ctx)):
                rows_processed += 1

                result = processor.process_row(
                    row_index=row_index,
                    row_data=row_data,
                    transforms=config.transforms,
                    ctx=ctx,
                )

                if result.outcome == "completed":
                    rows_succeeded += 1
                    pending_rows["default"].append(result.final_data)
                elif result.outcome == "routed":
                    rows_routed += 1
                    # Find which sink was routed to (simplified)
                    for sink_name in config.sinks:
                        if sink_name != "default":
                            pending_rows[sink_name].append(result.final_data)
                            break
                elif result.outcome == "failed":
                    rows_failed += 1

        # Write to sinks
        from elspeth.engine.executors import SinkExecutor
        sink_executor = SinkExecutor(recorder, self._span_factory, run_id)

        for sink_name, rows in pending_rows.items():
            if rows and sink_name in config.sinks:
                sink = config.sinks[sink_name]
                sink.write(rows, ctx)

        # Close source
        config.source.close()

        return RunResult(
            run_id=run_id,
            status="running",  # Will be updated
            rows_processed=rows_processed,
            rows_succeeded=rows_succeeded,
            rows_failed=rows_failed,
            rows_routed=rows_routed,
        )
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/engine/test_orchestrator.py -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/elspeth/engine/orchestrator.py tests/engine/test_orchestrator.py
git commit -m "feat(engine): add Orchestrator for full run lifecycle"
```

---

## Task 20: Engine Module Exports and Final Verification

**Files:**
- Modify: `src/elspeth/engine/__init__.py`
- Create: `tests/engine/test_integration.py`

### Step 1: Write the integration test

```python
# tests/engine/test_integration.py
"""Integration tests for SDA Engine."""

import pytest


class TestEngineIntegration:
    """Full engine integration tests."""

    def test_can_import_all_components(self) -> None:
        from elspeth.engine import (
            Orchestrator,
            PipelineConfig,
            RowProcessor,
            RowResult,
            RunResult,
            SpanFactory,
            TokenInfo,
            TokenManager,
        )

        assert Orchestrator is not None
        assert RowProcessor is not None
        assert SpanFactory is not None

    def test_full_pipeline_with_audit(self) -> None:
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine import Orchestrator, PipelineConfig
        from elspeth.plugins.results import TransformResult
        from elspeth.plugins.schemas import PluginSchema

        db = LandscapeDB.in_memory()

        class RowSchema(PluginSchema):
            value: int

        class ListSource:
            name = "list_source"
            output_schema = RowSchema

            def __init__(self, data):
                self._data = data

            def load(self, ctx):
                yield from self._data

            def close(self):
                pass

        class IncrementTransform:
            name = "increment"
            input_schema = RowSchema
            output_schema = RowSchema

            def process(self, row, ctx):
                return TransformResult.success({"value": row["value"] + 1})

        class MemorySink:
            name = "memory"
            results = []

            def write(self, rows, ctx):
                self.results.extend(rows)
                return {"path": "memory", "size_bytes": 0, "content_hash": ""}

        source = ListSource([{"value": i} for i in range(5)])
        transform = IncrementTransform()
        sink = MemorySink()

        config = PipelineConfig(
            source=source,
            transforms=[transform],
            sinks={"default": sink},
        )

        orchestrator = Orchestrator(db)
        result = orchestrator.run(config)

        # Check result
        assert result.status == "completed"
        assert result.rows_processed == 5
        assert result.rows_succeeded == 5

        # Check sink received transformed data
        assert len(sink.results) == 5
        assert sink.results[0] == {"value": 1}  # 0 + 1
        assert sink.results[4] == {"value": 5}  # 4 + 1

        # Check audit trail
        recorder = LandscapeRecorder(db)
        run = recorder.get_run(result.run_id)
        assert run is not None
        assert run.status == "completed"

        nodes = recorder.get_nodes(result.run_id)
        assert len(nodes) >= 3  # source, transform, sink
```

### Step 2: Run test to verify it fails

Run: `pytest tests/engine/test_integration.py -v`
Expected: FAIL (ImportError)

### Step 3: Update engine module exports

```python
# src/elspeth/engine/__init__.py
"""SDA Engine: Orchestration with complete audit trails.

This module provides the execution engine for ELSPETH pipelines:
- Orchestrator: Full run lifecycle management
- RowProcessor: Row-by-row processing through transforms
- TokenManager: Token identity through forks/joins
- SpanFactory: OpenTelemetry integration
- RetryManager: Retry logic with tenacity

Example:
    from elspeth.core.landscape import LandscapeDB
    from elspeth.engine import Orchestrator, PipelineConfig

    db = LandscapeDB.from_url("sqlite:///audit.db")

    config = PipelineConfig(
        source=csv_source,
        transforms=[transform1, gate1],
        sinks={"default": output_sink},
    )

    orchestrator = Orchestrator(db)
    result = orchestrator.run(config)
"""

from elspeth.engine.executors import (
    AggregationExecutor,
    GateExecutor,
    SinkExecutor,
    TransformExecutor,
)
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig, RunResult
from elspeth.engine.processor import RowProcessor, RowResult
from elspeth.engine.retry import MaxRetriesExceeded, RetryConfig, RetryManager
from elspeth.engine.spans import SpanFactory
from elspeth.engine.tokens import TokenInfo, TokenManager

__all__ = [
    # Orchestration
    "Orchestrator",
    "PipelineConfig",
    "RunResult",
    # Processing
    "RowProcessor",
    "RowResult",
    # Tokens
    "TokenManager",
    "TokenInfo",
    # Executors
    "TransformExecutor",
    "GateExecutor",
    "AggregationExecutor",
    "SinkExecutor",
    # Retry
    "RetryManager",
    "RetryConfig",
    "MaxRetriesExceeded",
    # Tracing
    "SpanFactory",
]
```

### Step 4: Run all tests

Run: `pytest tests/engine/ tests/core/landscape/ -v`
Expected: ALL PASS

### Step 5: Final commit

```bash
git add src/elspeth/engine/__init__.py tests/engine/test_integration.py
git commit -m "feat(engine): export public API and add integration tests"
```

---

# Phase 3 Complete

**All Tasks:**
1. ✅ LandscapeSchema - SQLAlchemy table definitions
2. ✅ LandscapeDB - Database connection manager
3. ✅ LandscapeRecorder - Run management
4. ✅ LandscapeRecorder - Node and edge registration
5. ✅ LandscapeRecorder - Row and token creation
6. ✅ LandscapeRecorder - NodeState recording
7. ✅ LandscapeRecorder - Routing events
8. ✅ LandscapeRecorder - Batch management
9. ✅ LandscapeRecorder - Artifact registration
10. ✅ Landscape module exports
11. ✅ SpanFactory - OpenTelemetry integration
12. ✅ TokenManager - High-level token operations
13. ✅ TransformExecutor - Audit-wrapped transform execution
14. ✅ GateExecutor - Routing with audit
15. ✅ AggregationExecutor - Batch tracking
16. ✅ SinkExecutor - Artifact recording
17. ✅ RetryManager - tenacity integration
18. ✅ RowProcessor - Row processing orchestration
19. ✅ Orchestrator - Full run lifecycle
20. ✅ Engine module exports and integration tests

---

## Final Verification

```bash
# Run all Phase 3 tests
pytest tests/core/landscape/ tests/engine/ -v

# Run full test suite
pytest -v

# Type check
mypy src/elspeth/core/landscape src/elspeth/engine

# Lint
ruff check src/elspeth/core/landscape src/elspeth/engine
```

---

**Final commit:**

```bash
git add docs/plans/2026-01-12-phase3-sda-engine.md
git commit -m "docs: complete Phase 3 SDA Engine implementation plan"
```
