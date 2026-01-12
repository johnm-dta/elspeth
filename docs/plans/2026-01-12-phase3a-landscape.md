# Phase 3A: Landscape - Audit Infrastructure (Tasks 1-10)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the SDA Engine that orchestrates plugin execution while recording complete audit trails to Landscape and emitting OpenTelemetry spans.

**Architecture:** The engine wraps plugin calls (transform, gate, aggregation, sink) to add audit behavior without modifying plugin code. LandscapeRecorder provides the high-level API for audit recording. TokenManager handles row instance identity through forks/joins. The Orchestrator coordinates the full run lifecycle.

**Tech Stack:** Python 3.11+, SQLAlchemy Core (database), OpenTelemetry (tracing), tenacity (retries), structlog (logging)

**Dependencies:**
- Phase 1: `elspeth.core.canonical`, `elspeth.core.config`, `elspeth.core.dag`, `elspeth.core.payload_store`, `elspeth.core.landscape.models`
- Phase 2: `elspeth.plugins` (protocols, results, context, schemas, manager, enums, PluginSpec)

**Phase 2 Additions (use these):**
- `NodeType` enum: Use instead of string literals like `"transform"`
- `Determinism` enum: For reproducibility grading
- `PluginSpec.from_plugin()`: For plugin registration metadata
- Note: `TransformResult.status` is now `"success" | "error"` only (no "route")

---

## Task 1: Complete Landscape Schema - Add Missing Tables

**Context:** Phase 1 created `schema.py` with 9 tables. The architecture requires 13 tables. This task adds the 4 missing tables (routing_events, batches, batch_members, batch_outputs), adds indexes to existing tables, and adds corresponding model classes.

**Files:**
- Modify: `src/elspeth/core/landscape/schema.py` (add 4 tables + indexes)
- Modify: `src/elspeth/core/landscape/models.py` (add 4 model classes)
- Modify: `tests/core/landscape/test_schema.py` (add tests for new tables)

### Step 1: Write the failing test for new tables

```python
# Add to tests/core/landscape/test_schema.py
"""Tests for new Landscape schema tables added in Phase 3A."""

import pytest
from sqlalchemy import create_engine, inspect


class TestPhase3ASchemaAdditions:
    """Tests for tables added in Phase 3A."""

    def test_routing_events_table_exists(self) -> None:
        from elspeth.core.landscape.schema import routing_events_table

        assert routing_events_table.name == "routing_events"
        columns = {c.name for c in routing_events_table.columns}
        assert "event_id" in columns
        assert "state_id" in columns
        assert "edge_id" in columns
        assert "routing_group_id" in columns

    def test_batches_table_exists(self) -> None:
        from elspeth.core.landscape.schema import batches_table

        assert batches_table.name == "batches"
        columns = {c.name for c in batches_table.columns}
        assert "batch_id" in columns
        assert "aggregation_node_id" in columns
        assert "status" in columns

    def test_batch_members_table_exists(self) -> None:
        from elspeth.core.landscape.schema import batch_members_table

        assert batch_members_table.name == "batch_members"

    def test_batch_outputs_table_exists(self) -> None:
        from elspeth.core.landscape.schema import batch_outputs_table

        assert batch_outputs_table.name == "batch_outputs"

    def test_all_13_tables_exist(self) -> None:
        from elspeth.core.landscape.schema import metadata

        table_names = set(metadata.tables.keys())
        expected = {
            "runs", "nodes", "edges", "rows", "tokens", "token_parents",
            "node_states", "routing_events", "calls", "batches",
            "batch_members", "batch_outputs", "artifacts",
        }
        assert expected.issubset(table_names), f"Missing: {expected - table_names}"


class TestPhase3AModels:
    """Tests for model classes added in Phase 3A."""

    def test_routing_event_model(self) -> None:
        from elspeth.core.landscape.models import RoutingEvent

        event = RoutingEvent(
            event_id="evt1",
            state_id="state1",
            edge_id="edge1",
            routing_group_id="grp1",
            ordinal=0,
            mode="move",
            created_at=None,  # Will be set in real use
        )
        assert event.event_id == "evt1"

    def test_batch_model(self) -> None:
        from elspeth.core.landscape.models import Batch

        batch = Batch(
            batch_id="batch1",
            run_id="run1",
            aggregation_node_id="node1",
            attempt=0,
            status="draft",
            created_at=None,
        )
        assert batch.status == "draft"
```

### Step 2: Run test to verify it fails

Run: `pytest tests/core/landscape/test_schema.py::TestPhase3ASchemaAdditions -v`
Expected: FAIL (ImportError - routing_events_table not found)

### Step 3: Add missing tables to schema.py

Add the following to `src/elspeth/core/landscape/schema.py` after `calls_table`:

```python
from sqlalchemy import Index  # Add to imports if not present

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
    Column("reason_hash", String(64)),
    Column("reason_ref", String(256)),
    Column("created_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("routing_group_id", "ordinal"),
)

# === Batches (Aggregation) ===

batches_table = Table(
    "batches",
    metadata,
    Column("batch_id", String(64), primary_key=True),
    Column("run_id", String(64), ForeignKey("runs.run_id"), nullable=False),
    Column("aggregation_node_id", String(64), ForeignKey("nodes.node_id"), nullable=False),
    Column("aggregation_state_id", String(64), ForeignKey("node_states.state_id")),
    Column("trigger_reason", String(128)),
    Column("attempt", Integer, nullable=False, default=0),
    Column("status", String(32), nullable=False),  # draft, executing, completed, failed
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("completed_at", DateTime(timezone=True)),
)

batch_members_table = Table(
    "batch_members",
    metadata,
    Column("batch_id", String(64), ForeignKey("batches.batch_id"), nullable=False),
    Column("token_id", String(64), ForeignKey("tokens.token_id"), nullable=False),
    Column("ordinal", Integer, nullable=False),
    UniqueConstraint("batch_id", "ordinal"),
    UniqueConstraint("batch_id", "token_id"),  # Prevent duplicate token in same batch
)

batch_outputs_table = Table(
    "batch_outputs",
    metadata,
    Column("batch_output_id", String(64), primary_key=True),  # Surrogate PK
    Column("batch_id", String(64), ForeignKey("batches.batch_id"), nullable=False),
    Column("output_type", String(32), nullable=False),  # token, artifact
    Column("output_id", String(64), nullable=False),
    UniqueConstraint("batch_id", "output_type", "output_id"),  # Prevent duplicates
)

# === Indexes for Query Performance ===
# Add these after all table definitions

Index("ix_routing_events_state", routing_events_table.c.state_id)
Index("ix_routing_events_group", routing_events_table.c.routing_group_id)
Index("ix_batches_run_status", batches_table.c.run_id, batches_table.c.status)
Index("ix_batch_members_batch", batch_members_table.c.batch_id)
Index("ix_batch_outputs_batch", batch_outputs_table.c.batch_id)

# Also add indexes to existing Phase 1 tables (in same file):
Index("ix_nodes_run_id", nodes_table.c.run_id)
Index("ix_edges_run_id", edges_table.c.run_id)
Index("ix_rows_run_id", rows_table.c.run_id)
Index("ix_tokens_row_id", tokens_table.c.row_id)
Index("ix_token_parents_parent", token_parents_table.c.parent_token_id)
Index("ix_node_states_token", node_states_table.c.token_id)
Index("ix_node_states_node", node_states_table.c.node_id)
Index("ix_calls_state", calls_table.c.state_id)
Index("ix_artifacts_run", artifacts_table.c.run_id)
```

### Step 4: Add missing model classes to models.py

Add the following to `src/elspeth/core/landscape/models.py`:

```python
@dataclass
class RoutingEvent:
    """A routing decision at a gate node."""

    event_id: str
    state_id: str
    edge_id: str
    routing_group_id: str
    ordinal: int
    mode: str  # move, copy
    created_at: datetime
    reason_hash: str | None = None
    reason_ref: str | None = None


@dataclass
class Batch:
    """An aggregation batch collecting tokens."""

    batch_id: str
    run_id: str
    aggregation_node_id: str
    attempt: int
    status: str  # draft, executing, completed, failed
    created_at: datetime
    aggregation_state_id: str | None = None
    trigger_reason: str | None = None
    completed_at: datetime | None = None


@dataclass
class BatchMember:
    """A token belonging to a batch."""

    batch_id: str
    token_id: str
    ordinal: int


@dataclass
class BatchOutput:
    """An output produced by a batch."""

    batch_id: str
    output_type: str  # token, artifact
    output_id: str
```

### Step 5: Update landscape __init__.py exports

Add to `src/elspeth/core/landscape/__init__.py`:

```python
from elspeth.core.landscape.models import (
    # ... existing imports ...
    Batch,
    BatchMember,
    BatchOutput,
    RoutingEvent,
)
from elspeth.core.landscape.schema import (
    # ... existing imports ...
    batch_members_table,
    batch_outputs_table,
    batches_table,
    routing_events_table,
)
```

### Step 6: Run tests to verify they pass

Run: `pytest tests/core/landscape/test_schema.py -v`
Expected: PASS

### Step 7: Commit

```bash
git add -u
git commit -m "feat(landscape): add routing_events, batches tables and models"
```

---

## Task 2: Extend LandscapeDB - Add Factory Methods

**Context:** Phase 1 created `database.py` with basic connection handling. Phase 3A needs `in_memory()` and `connection()` methods for the LandscapeRecorder.

**Files:**
- Modify: `src/elspeth/core/landscape/database.py` (add factory methods)
- Modify: `tests/core/landscape/test_database.py` (add tests for new methods)

### Step 1: Write the failing test for new methods

```python
# Add to tests/core/landscape/test_database.py
"""Tests for LandscapeDB factory methods added in Phase 3A."""

import pytest
from sqlalchemy import inspect, text


class TestPhase3ADBMethods:
    """Tests for methods added in Phase 3A."""

    def test_in_memory_factory(self) -> None:
        from elspeth.core.landscape.database import LandscapeDB

        db = LandscapeDB.in_memory()
        assert db.engine is not None
        inspector = inspect(db.engine)
        assert "runs" in inspector.get_table_names()

    def test_connection_context_manager(self) -> None:
        from elspeth.core.landscape.database import LandscapeDB

        db = LandscapeDB.in_memory()
        with db.connection() as conn:
            result = conn.execute(text("SELECT 1"))
            assert result.scalar() == 1

    def test_from_url_factory(self, tmp_path) -> None:
        from elspeth.core.landscape.database import LandscapeDB

        db_path = tmp_path / "test.db"
        db = LandscapeDB.from_url(f"sqlite:///{db_path}")
        assert db_path.exists()
        inspector = inspect(db.engine)
        assert "runs" in inspector.get_table_names()

    def test_from_url_skip_table_creation(self, tmp_path) -> None:
        """Test that create_tables=False doesn't create tables."""
        from elspeth.core.landscape.database import LandscapeDB

        db_path = tmp_path / "empty.db"
        # First create an empty database file (no tables)
        empty_engine = create_engine(f"sqlite:///{db_path}")
        empty_engine.dispose()

        # Connect with create_tables=False - should NOT create tables
        db = LandscapeDB.from_url(f"sqlite:///{db_path}", create_tables=False)
        inspector = inspect(db.engine)
        assert "runs" not in inspector.get_table_names()  # No tables!
```

### Step 2: Run test to verify it fails

Run: `pytest tests/core/landscape/test_database.py::TestPhase3ADBMethods -v`
Expected: FAIL (AttributeError: type object 'LandscapeDB' has no attribute 'in_memory')

### Step 3: Add factory methods to database.py

Modify `src/elspeth/core/landscape/database.py` to add these methods:

```python
# Add these imports at the top
from contextlib import contextmanager
from typing import Iterator, Self

from sqlalchemy import Connection, text

# Add these methods to the LandscapeDB class:

    @classmethod
    def in_memory(cls) -> Self:
        """Create an in-memory SQLite database for testing.

        Tables are created automatically.

        Returns:
            LandscapeDB instance with in-memory SQLite
        """
        engine = create_engine("sqlite:///:memory:", echo=False)
        metadata.create_all(engine)
        instance = cls.__new__(cls)
        instance.connection_string = "sqlite:///:memory:"
        instance._engine = engine
        return instance

    @classmethod
    def from_url(cls, url: str, *, create_tables: bool = True) -> Self:
        """Create database from connection URL.

        Args:
            url: SQLAlchemy connection URL
            create_tables: Whether to create tables if they don't exist.
                           Set to False when connecting to an existing database.

        Returns:
            LandscapeDB instance
        """
        # Bypass __init__ to avoid automatic table creation
        # (same pattern as in_memory())
        engine = create_engine(url, echo=False)
        if create_tables:
            metadata.create_all(engine)
        instance = cls.__new__(cls)
        instance.connection_string = url
        instance._engine = engine
        return instance

    @contextmanager
    def connection(self) -> Iterator[Connection]:
        """Get a database connection with automatic transaction handling.

        Uses engine.begin() for proper transaction semantics:
        - Auto-commits on successful block exit
        - Auto-rolls back on exception

        Usage:
            with db.connection() as conn:
                conn.execute(runs_table.insert().values(...))
            # Committed automatically if no exception raised
        """
        with self.engine.begin() as conn:
            yield conn
        # No explicit commit needed - begin() handles it
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/core/landscape/test_database.py -v`
Expected: PASS

### Step 5: Commit

```bash
git add -u
git commit -m "feat(landscape): add LandscapeDB factory methods for Phase 3A"
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
from enum import Enum
from typing import Any, TypeVar

from sqlalchemy import select

from elspeth.core.canonical import canonical_json, stable_hash
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.models import Run
from elspeth.core.landscape.schema import runs_table
from elspeth.plugins.enums import Determinism, NodeType

E = TypeVar("E", bound=Enum)


def _now() -> datetime:
    """Get current UTC timestamp."""
    return datetime.now(timezone.utc)


def _generate_id() -> str:
    """Generate a unique ID."""
    return uuid.uuid4().hex


def _coerce_enum(value: str | E, enum_type: type[E]) -> E:
    """Coerce a string or enum value to the target enum type.

    Args:
        value: String value or enum instance
        enum_type: Target enum class

    Returns:
        Enum instance

    Raises:
        ValueError: If string doesn't match any enum value

    Example:
        >>> _coerce_enum("transform", NodeType)
        <NodeType.TRANSFORM: 'transform'>
        >>> _coerce_enum(NodeType.TRANSFORM, NodeType)
        <NodeType.TRANSFORM: 'transform'>
    """
    if isinstance(value, enum_type):
        return value
    # str-based enums use value lookup
    return enum_type(value)


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

    def test_register_node_with_enum(self) -> None:
        """Test that NodeType enum is accepted and coerced."""
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder
        from elspeth.plugins.enums import NodeType

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        # Both enum and string should work
        node_from_enum = recorder.register_node(
            run_id=run.run_id,
            plugin_name="transform1",
            node_type=NodeType.TRANSFORM,  # Enum
            plugin_version="1.0.0",
            config={},
        )
        node_from_str = recorder.register_node(
            run_id=run.run_id,
            plugin_name="transform2",
            node_type="transform",  # String
            plugin_version="1.0.0",
            config={},
        )

        # Both should store the same string value
        assert node_from_enum.node_type == "transform"
        assert node_from_str.node_type == "transform"

    def test_register_node_invalid_type_raises(self) -> None:
        """Test that invalid node_type string raises ValueError."""
        import pytest
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        with pytest.raises(ValueError, match="transfom"):  # Note typo
            recorder.register_node(
                run_id=run.run_id,
                plugin_name="bad",
                node_type="transfom",  # Typo! Should fail fast
                plugin_version="1.0.0",
                config={},
            )

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
        node_type: NodeType | str,
        plugin_version: str,
        config: dict[str, Any],
        *,
        node_id: str | None = None,
        sequence: int | None = None,
        schema_hash: str | None = None,
        determinism: Determinism | str = Determinism.PURE,
    ) -> Node:
        """Register a plugin instance (node) in the execution graph.

        Args:
            run_id: Run this node belongs to
            plugin_name: Name of the plugin
            node_type: Type (source, transform, gate, aggregation, coalesce, sink)
                       Accepts NodeType enum or string (will be validated)
            plugin_version: Version of the plugin
            config: Plugin configuration
            node_id: Optional node ID (generated if not provided)
            sequence: Position in pipeline
            schema_hash: Optional input/output schema hash
            determinism: Reproducibility grade (Determinism enum or string)

        Returns:
            Node model

        Raises:
            ValueError: If node_type or determinism string is not a valid enum value
        """
        # Validate and coerce enums early - fail fast on typos
        node_type_enum = _coerce_enum(node_type, NodeType)
        determinism_enum = _coerce_enum(determinism, Determinism)

        node_id = node_id or _generate_id()
        config_json = canonical_json(config)
        config_hash = stable_hash(config)
        now = _now()

        node = Node(
            node_id=node_id,
            run_id=run_id,
            plugin_name=plugin_name,
            node_type=node_type_enum.value,  # Store string in DB
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
            List of Node models, ordered by sequence (NULL sequences last)
        """
        with self._db.connection() as conn:
            result = conn.execute(
                select(nodes_table)
                .where(nodes_table.c.run_id == run_id)
                # Use nullslast() for consistent NULL handling across databases
                # Nodes without sequence (e.g., dynamically added) sort last
                .order_by(nodes_table.c.sequence_in_pipeline.nullslast())
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

# Add imports (RoutingEvent already defined in models.py - DO NOT REDEFINE)
from elspeth.core.landscape.models import RoutingEvent
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

# Add imports (Batch and BatchMember already defined in models.py - DO NOT REDEFINE)
from elspeth.core.landscape.models import Batch, BatchMember
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
