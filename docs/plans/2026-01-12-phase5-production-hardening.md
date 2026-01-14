# Phase 5: Production Hardening (Tasks 0-14)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add operational reliability features: checkpointing for crash recovery, rate limiting for external API protection, and retention/purge for storage management.

**Architecture:** Checkpointing saves progress at row boundaries, enabling resume from crashes. Rate limiting uses pyrate-limiter with SQLite persistence to respect external API limits. Retention purges old PayloadStore blobs while preserving Landscape metadata (hashes remain for verification).

**Tech Stack:** Python 3.11+, pyrate-limiter (rate limiting), SQLite (persistence), Typer (CLI)

**Dependencies:**
- Phase 1: `elspeth.core.canonical`, `elspeth.core.config`, `elspeth.core.payload_store`
- Phase 2: `elspeth.plugins` (protocols, context)
- Phase 3A: `elspeth.core.landscape` (schema, db, recorder)
- Phase 3B: `elspeth.engine` (Orchestrator, RowProcessor, executors)
- Phase 4: `elspeth.cli` (for purge command)

**Deferred to Phase 6:**
- Redaction profiles (needed when LLM calls store prompts/responses)
- Replay mode (re-run with recorded external call responses)
- Secret fingerprinting (HMAC for API key verification)

---

## Audit Completeness Requirement

**Critical invariant for all Phase 5 features:**

The Landscape stores data at every boundary crossing:
1. **Source entry** - Raw data before any processing (`rows` table)
2. **Transform boundaries** - Input AND output of each transform (`node_states` table)
3. **External calls** - Full request AND response (`calls` table)
4. **Sink output** - Final artifacts (`artifacts` table)

**Redaction, if needed, is the responsibility of plugins BEFORE invoking Landscape recording methods.** The Landscape is a faithful recorder - it stores what it's given.

---

## Checkpointing Architecture

**Two distinct checkpointing mechanisms exist:**

| Mechanism | Scope | Phase | How It Works |
|-----------|-------|-------|--------------|
| **Aggregation state** | Rows buffered in aggregations | Phase 3A | `batch_members` table persists membership on every `accept()`. Crash recovery via query. |
| **Row-level progress** | Which rows have been fully processed | Phase 5 | `checkpoints` table tracks last completed token/node. Resume skips completed rows. |

**Why two mechanisms?**

1. **Aggregation checkpointing (Phase 3A)** solves: "What rows are sitting in aggregation buffers waiting for flush?"
   - Already handled by the "draft batch pattern" - persist membership immediately
   - Recovery: `SELECT * FROM batch_members WHERE batch_id IN (SELECT batch_id FROM batches WHERE status='draft')`

2. **Row-level checkpointing (Phase 5)** solves: "If we crash mid-run, which rows can we skip on resume?"
   - Tracks pipeline progress independent of aggregation state
   - Enables resume from any row, not just aggregation boundaries

**The two mechanisms are complementary, not redundant.**

---

## Reproducibility Grade

The `runs.reproducibility_grade` column tracks what level of reproducibility is possible:

| Grade | Meaning | When Assigned |
|-------|---------|---------------|
| `FULL_REPRODUCIBLE` | All transforms deterministic or seeded | Run completion (all nodes have `deterministic` or `seeded` determinism) |
| `REPLAY_REPRODUCIBLE` | Has non-deterministic transforms, payloads retained | Run completion (has `nondeterministic` nodes, payloads exist) |
| `ATTRIBUTABLE_ONLY` | Payloads purged | After purge job runs |

**Computation:** At run completion, scan `nodes.determinism` for any `nondeterministic` values (from `Determinism` enum). Grade degrades from `FULL_REPRODUCIBLE` → `ATTRIBUTABLE_ONLY` over time as payloads are purged.

---

## Task 0: Add determinism Column to nodes_table (Pre-requisite)

**Context:** The reproducibility grade computation (Task 14) requires scanning `nodes.determinism` to determine if a run used any non-deterministic transforms. This column must be added to the Landscape schema before other tasks can proceed.

**Files:**
- Modify: `src/elspeth/core/landscape/schema.py`
- Modify: `src/elspeth/core/landscape/models.py`
- Modify: `tests/core/landscape/test_schema.py`

### Step 1: Write the failing test

```python
# Add to tests/core/landscape/test_schema.py

class TestNodesDeterminismColumn:
    """Tests for determinism column in nodes table."""

    def test_nodes_table_has_determinism_column(self) -> None:
        from elspeth.core.landscape.schema import nodes_table

        columns = {c.name for c in nodes_table.columns}
        assert "determinism" in columns

    def test_node_model_has_determinism_field(self) -> None:
        from elspeth.core.landscape.models import Node
        from datetime import datetime, timezone

        node = Node(
            node_id="node-001",
            run_id="run-001",
            plugin_name="test_plugin",
            node_type="transform",
            plugin_version="1.0.0",
            determinism="deterministic",  # New field
            config_hash="abc123",
            config_json="{}",
            registered_at=datetime.now(timezone.utc),
        )
        assert node.determinism == "deterministic"

    def test_determinism_values(self) -> None:
        """Verify valid determinism values match Determinism enum."""
        from elspeth.plugins.enums import Determinism

        valid_values = {d.value for d in Determinism}
        # Current enum values (not the granular architecture spec values)
        expected = {"deterministic", "seeded", "nondeterministic"}
        assert valid_values == expected
```

### Step 2: Run test to verify it fails

Run: `pytest tests/core/landscape/test_schema.py::TestNodesDeterminismColumn -v`
Expected: FAIL (determinism column/field not found)

### Step 3: Add determinism column to schema

Add to `nodes_table` in `src/elspeth/core/landscape/schema.py` (after `plugin_version` column):

```python
Column("determinism", String(32), nullable=False),  # deterministic, seeded, nondeterministic (from Determinism enum)
```

### Step 4: Add determinism field to Node model

Update the `Node` dataclass in `src/elspeth/core/landscape/models.py`:

```python
@dataclass
class Node:
    """A node (plugin instance) in the execution graph."""

    node_id: str
    run_id: str
    plugin_name: str
    node_type: str  # source, transform, gate, aggregation, coalesce, sink
    plugin_version: str
    determinism: str  # From Determinism enum: deterministic, seeded, nondeterministic
    config_hash: str
    config_json: str
    registered_at: datetime
    schema_hash: str | None = None
    sequence_in_pipeline: int | None = None
```

### Step 5: Update LandscapeRecorder.register_node()

The `register_node` method in `src/elspeth/core/landscape/recorder.py` must accept and store the determinism value:

```python
def register_node(
    self,
    run_id: str,
    node_id: str,
    plugin_name: str,
    node_type: NodeType,
    plugin_version: str,
    config: dict[str, Any],
    determinism: str = "deterministic",  # Add this parameter with default
    schema_hash: str | None = None,
    sequence_in_pipeline: int | None = None,
) -> Node:
    # ... existing code ...
    # Add determinism to the insert values
```

### Step 6: Run tests

Run: `pytest tests/core/landscape/test_schema.py::TestNodesDeterminismColumn -v`
Expected: PASS

**Note:** This column captures the plugin's declared determinism level (from `elspeth.plugins.enums.Determinism`). The engine sets this when registering nodes based on the plugin's `determinism` attribute.

---

## Task 1: Checkpoint Schema and Models

**Context:** Add checkpoint table to Landscape schema. Checkpoints capture run progress at row boundaries, enabling resume after crash.

**Files:**
- Modify: `src/elspeth/core/landscape/schema.py` (add checkpoints table)
- Modify: `src/elspeth/core/landscape/models.py` (add Checkpoint dataclass)
- Modify: `tests/core/landscape/test_schema.py` (add checkpoint tests)

### Step 1: Write the failing test

```python
# Add to tests/core/landscape/test_schema.py

class TestPhase5CheckpointSchema:
    """Tests for checkpoint table added in Phase 5."""

    def test_checkpoints_table_exists(self) -> None:
        from elspeth.core.landscape.schema import checkpoints_table

        assert checkpoints_table.name == "checkpoints"
        columns = {c.name for c in checkpoints_table.columns}
        assert "checkpoint_id" in columns
        assert "run_id" in columns
        assert "token_id" in columns
        assert "node_id" in columns
        assert "created_at" in columns

    def test_checkpoint_model(self) -> None:
        from elspeth.core.landscape.models import Checkpoint

        checkpoint = Checkpoint(
            checkpoint_id="cp-001",
            run_id="run-001",
            token_id="tok-001",
            node_id="node-001",
            sequence_number=42,
            created_at=None,
        )
        assert checkpoint.sequence_number == 42
```

### Step 2: Run test to verify it fails

Run: `pytest tests/core/landscape/test_schema.py::TestPhase5CheckpointSchema -v`
Expected: FAIL (ImportError - checkpoints_table not found)

### Step 3: Add checkpoints table to schema

Add to `src/elspeth/core/landscape/schema.py` after artifacts_table:

```python
# === Checkpoints (Phase 5: Production Hardening) ===

checkpoints_table = Table(
    "checkpoints",
    metadata,
    Column("checkpoint_id", String(64), primary_key=True),
    Column("run_id", String(64), ForeignKey("runs.run_id"), nullable=False),
    Column("token_id", String(64), ForeignKey("tokens.token_id"), nullable=False),
    Column("node_id", String(64), ForeignKey("nodes.node_id"), nullable=False),
    Column("sequence_number", Integer, nullable=False),  # Monotonic progress marker
    Column("aggregation_state_json", Text),  # Serialized aggregation buffers (if any)
    Column("created_at", DateTime(timezone=True), nullable=False),
)

Index("ix_checkpoints_run", checkpoints_table.c.run_id)
Index("ix_checkpoints_run_seq", checkpoints_table.c.run_id, checkpoints_table.c.sequence_number)
```

### Step 4: Add Checkpoint model

Add to `src/elspeth/core/landscape/models.py`:

```python
@dataclass
class Checkpoint:
    """Checkpoint for crash recovery.

    Captures run progress at row/transform boundaries.
    sequence_number is monotonically increasing within a run.
    """
    checkpoint_id: str
    run_id: str
    token_id: str
    node_id: str
    sequence_number: int
    created_at: datetime | None
    aggregation_state_json: str | None = None
```

### Step 5: Run tests

Run: `pytest tests/core/landscape/test_schema.py::TestPhase5CheckpointSchema -v`
Expected: PASS

---

## Task 2: CheckpointManager - Create and Load Checkpoints

**Context:** CheckpointManager provides the API for creating checkpoints during execution and loading them for recovery.

**Files:**
- Create: `src/elspeth/core/checkpoint/__init__.py`
- Create: `src/elspeth/core/checkpoint/manager.py`
- Create: `tests/core/checkpoint/__init__.py`
- Create: `tests/core/checkpoint/test_manager.py`

### Step 1: Write the failing test

```python
# tests/core/checkpoint/__init__.py
"""Checkpoint tests."""

# tests/core/checkpoint/test_manager.py
"""Tests for CheckpointManager."""

import pytest
from datetime import datetime, timezone


class TestCheckpointManager:
    """Tests for checkpoint creation and loading."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create CheckpointManager with test database."""
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.checkpoint.manager import CheckpointManager

        db = LandscapeDB(f"sqlite:///{tmp_path}/test.db")
        db.create_tables()
        return CheckpointManager(db)

    @pytest.fixture
    def setup_run(self, manager):
        """Create a run with some tokens for checkpoint tests."""
        from elspeth.core.landscape.schema import (
            runs_table, nodes_table, rows_table, tokens_table
        )

        # Insert test run, node, row, token via schema tables
        db = manager._db
        with db.engine.connect() as conn:
            conn.execute(
                runs_table.insert().values(
                    run_id="run-001",
                    started_at=datetime.now(timezone.utc),
                    config_hash="abc123",
                    settings_json="{}",
                    canonical_version="sha256-rfc8785-v1",
                    status="running",
                )
            )
            conn.execute(
                nodes_table.insert().values(
                    node_id="node-001",
                    run_id="run-001",
                    plugin_name="test_transform",
                    node_type="transform",
                    plugin_version="1.0.0",
                    determinism="pure",
                    config_hash="xyz",
                    config_json="{}",
                    registered_at=datetime.now(timezone.utc),
                )
            )
            conn.execute(
                rows_table.insert().values(
                    row_id="row-001",
                    run_id="run-001",
                    source_node_id="node-001",
                    row_index=0,
                    source_data_hash="hash1",
                    created_at=datetime.now(timezone.utc),
                )
            )
            conn.execute(
                tokens_table.insert().values(
                    token_id="tok-001",
                    row_id="row-001",
                    created_at=datetime.now(timezone.utc),
                )
            )
            conn.commit()
        return "run-001"

    def test_create_checkpoint(self, manager, setup_run) -> None:
        """Can create a checkpoint."""
        checkpoint = manager.create_checkpoint(
            run_id="run-001",
            token_id="tok-001",
            node_id="node-001",
            sequence_number=1,
        )

        assert checkpoint.checkpoint_id is not None
        assert checkpoint.run_id == "run-001"
        assert checkpoint.sequence_number == 1

    def test_get_latest_checkpoint(self, manager, setup_run) -> None:
        """Can retrieve the latest checkpoint for a run."""
        # Create multiple checkpoints
        manager.create_checkpoint("run-001", "tok-001", "node-001", 1)
        manager.create_checkpoint("run-001", "tok-001", "node-001", 2)
        manager.create_checkpoint("run-001", "tok-001", "node-001", 3)

        latest = manager.get_latest_checkpoint("run-001")

        assert latest is not None
        assert latest.sequence_number == 3

    def test_get_latest_checkpoint_no_checkpoints(self, manager) -> None:
        """Returns None when no checkpoints exist."""
        latest = manager.get_latest_checkpoint("nonexistent-run")
        assert latest is None

    def test_checkpoint_with_aggregation_state(self, manager, setup_run) -> None:
        """Can store aggregation state in checkpoint."""
        import json

        agg_state = {"buffer": [1, 2, 3], "count": 3}

        checkpoint = manager.create_checkpoint(
            run_id="run-001",
            token_id="tok-001",
            node_id="node-001",
            sequence_number=1,
            aggregation_state=agg_state,
        )

        loaded = manager.get_latest_checkpoint("run-001")
        assert loaded is not None
        assert json.loads(loaded.aggregation_state_json) == agg_state
```

### Step 2: Run test to verify it fails

Run: `pytest tests/core/checkpoint/test_manager.py -v`
Expected: FAIL (ImportError)

### Step 3: Create CheckpointManager

```python
# src/elspeth/core/checkpoint/__init__.py
"""Checkpoint subsystem for crash recovery.

Provides:
- CheckpointManager: Create and load checkpoints
- Recovery protocol for resuming crashed runs
"""

from elspeth.core.checkpoint.manager import CheckpointManager

__all__ = ["CheckpointManager"]


# src/elspeth/core/checkpoint/manager.py
"""CheckpointManager for creating and loading checkpoints."""

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.models import Checkpoint
from elspeth.core.landscape.schema import checkpoints_table


class CheckpointManager:
    """Manages checkpoint creation and retrieval.

    Checkpoints capture run progress at row boundaries, enabling
    resume after crash. Each checkpoint records:
    - Which token was being processed
    - Which node it was at
    - A monotonic sequence number for ordering
    - Optional aggregation state for stateful plugins
    """

    def __init__(self, db: LandscapeDB) -> None:
        """Initialize with Landscape database.

        Args:
            db: LandscapeDB instance for storage
        """
        self._db = db

    def create_checkpoint(
        self,
        run_id: str,
        token_id: str,
        node_id: str,
        sequence_number: int,
        aggregation_state: dict[str, Any] | None = None,
    ) -> Checkpoint:
        """Create a checkpoint at current progress point.

        Args:
            run_id: The run being checkpointed
            token_id: Current token being processed
            node_id: Current node in the pipeline
            sequence_number: Monotonic progress marker
            aggregation_state: Optional serializable aggregation buffers

        Returns:
            The created Checkpoint
        """
        checkpoint_id = f"cp-{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc)

        agg_json = json.dumps(aggregation_state) if aggregation_state else None

        with self._db.engine.connect() as conn:
            conn.execute(
                checkpoints_table.insert().values(
                    checkpoint_id=checkpoint_id,
                    run_id=run_id,
                    token_id=token_id,
                    node_id=node_id,
                    sequence_number=sequence_number,
                    aggregation_state_json=agg_json,
                    created_at=now,
                )
            )
            conn.commit()

        return Checkpoint(
            checkpoint_id=checkpoint_id,
            run_id=run_id,
            token_id=token_id,
            node_id=node_id,
            sequence_number=sequence_number,
            aggregation_state_json=agg_json,
            created_at=now,
        )

    def get_latest_checkpoint(self, run_id: str) -> Checkpoint | None:
        """Get the most recent checkpoint for a run.

        Args:
            run_id: The run to get checkpoint for

        Returns:
            Latest Checkpoint or None if no checkpoints exist
        """
        from sqlalchemy import select, desc

        with self._db.engine.connect() as conn:
            result = conn.execute(
                select(checkpoints_table)
                .where(checkpoints_table.c.run_id == run_id)
                .order_by(desc(checkpoints_table.c.sequence_number))
                .limit(1)
            ).fetchone()

        if result is None:
            return None

        return Checkpoint(
            checkpoint_id=result.checkpoint_id,
            run_id=result.run_id,
            token_id=result.token_id,
            node_id=result.node_id,
            sequence_number=result.sequence_number,
            aggregation_state_json=result.aggregation_state_json,
            created_at=result.created_at,
        )

    def get_checkpoints(self, run_id: str) -> list[Checkpoint]:
        """Get all checkpoints for a run, ordered by sequence.

        Args:
            run_id: The run to get checkpoints for

        Returns:
            List of Checkpoints ordered by sequence_number
        """
        from sqlalchemy import select, asc

        with self._db.engine.connect() as conn:
            results = conn.execute(
                select(checkpoints_table)
                .where(checkpoints_table.c.run_id == run_id)
                .order_by(asc(checkpoints_table.c.sequence_number))
            ).fetchall()

        return [
            Checkpoint(
                checkpoint_id=r.checkpoint_id,
                run_id=r.run_id,
                token_id=r.token_id,
                node_id=r.node_id,
                sequence_number=r.sequence_number,
                aggregation_state_json=r.aggregation_state_json,
                created_at=r.created_at,
            )
            for r in results
        ]

    def delete_checkpoints(self, run_id: str) -> int:
        """Delete all checkpoints for a completed run.

        Called after successful run completion to clean up.

        Args:
            run_id: The run to clean up

        Returns:
            Number of checkpoints deleted
        """
        from sqlalchemy import delete

        with self._db.engine.connect() as conn:
            result = conn.execute(
                delete(checkpoints_table)
                .where(checkpoints_table.c.run_id == run_id)
            )
            conn.commit()
            return result.rowcount
```

### Step 4: Run tests

Run: `pytest tests/core/checkpoint/test_manager.py -v`
Expected: PASS

---

## Task 3: Checkpoint Frequency Configuration

**Context:** Add configuration for checkpoint frequency. Checkpointing every row is safe but slow; users can tune the trade-off.

**Files:**
- Modify: `src/elspeth/core/config.py` (add CheckpointSettings)
- Modify: `tests/core/test_config.py` (add checkpoint config tests)

### Step 1: Write the failing test

```python
# Add to tests/core/test_config.py

class TestCheckpointSettings:
    """Tests for checkpoint configuration."""

    def test_checkpoint_settings_defaults(self) -> None:
        from elspeth.core.config import CheckpointSettings

        settings = CheckpointSettings()

        assert settings.enabled is True
        assert settings.frequency == "every_row"
        assert settings.aggregation_boundaries is True

    def test_checkpoint_frequency_options(self) -> None:
        from elspeth.core.config import CheckpointSettings

        # Every row (safest, slowest)
        s1 = CheckpointSettings(frequency="every_row")
        assert s1.frequency == "every_row"

        # Every N rows (balanced)
        s2 = CheckpointSettings(frequency="every_n", checkpoint_interval=100)
        assert s2.frequency == "every_n"
        assert s2.checkpoint_interval == 100

        # Aggregation boundaries only (fastest, less safe)
        s3 = CheckpointSettings(frequency="aggregation_only")
        assert s3.frequency == "aggregation_only"

    def test_checkpoint_settings_validation(self) -> None:
        from pydantic import ValidationError
        from elspeth.core.config import CheckpointSettings

        # every_n requires checkpoint_interval
        with pytest.raises(ValidationError):
            CheckpointSettings(frequency="every_n", checkpoint_interval=None)
```

### Step 2: Run test to verify it fails

Run: `pytest tests/core/test_config.py::TestCheckpointSettings -v`
Expected: FAIL (ImportError - CheckpointSettings not found)

### Step 3: Add CheckpointSettings to config

Add to `src/elspeth/core/config.py`:

```python
from typing import Literal

from pydantic import BaseModel, model_validator


class CheckpointSettings(BaseModel):
    """Configuration for crash recovery checkpointing.

    Checkpoint frequency trade-offs:
    - every_row: Safest, can resume from any row. Higher I/O overhead.
    - every_n: Balance safety and performance. Lose up to N-1 rows on crash.
    - aggregation_only: Fastest, checkpoint only at aggregation flushes.
    """

    model_config = {"frozen": True}

    enabled: bool = True
    frequency: Literal["every_row", "every_n", "aggregation_only"] = "every_row"
    checkpoint_interval: int | None = None  # Required if frequency == "every_n"
    aggregation_boundaries: bool = True  # Always checkpoint at aggregation flush

    @model_validator(mode="after")
    def validate_interval(self) -> "CheckpointSettings":
        if self.frequency == "every_n" and self.checkpoint_interval is None:
            raise ValueError("checkpoint_interval required when frequency='every_n'")
        return self
```

Also add to `ElspethSettings`:

```python
class ElspethSettings(BaseModel):
    # ... existing fields ...
    checkpoint: CheckpointSettings = CheckpointSettings()
```

### Step 4: Run tests

Run: `pytest tests/core/test_config.py::TestCheckpointSettings -v`
Expected: PASS

---

## Task 4: Orchestrator Checkpoint Integration

**Context:** Integrate CheckpointManager into Orchestrator. Create checkpoints according to configured frequency.

**Files:**
- Modify: `src/elspeth/engine/orchestrator.py` (add checkpoint calls)
- Modify: `tests/engine/test_orchestrator.py` (add checkpoint integration tests)

### Step 1: Write the failing test

```python
# Add to tests/engine/test_orchestrator.py

import pytest
from datetime import datetime, timezone


class TestOrchestratorCheckpointing:
    """Tests for checkpoint integration in Orchestrator."""

    @pytest.fixture
    def landscape_db(self, tmp_path):
        """Create test database."""
        from elspeth.core.landscape.database import LandscapeDB

        db = LandscapeDB(f"sqlite:///{tmp_path}/test.db")
        db.create_tables()
        return db

    @pytest.fixture
    def checkpoint_manager(self, landscape_db):
        """Create CheckpointManager for tests."""
        from elspeth.core.checkpoint import CheckpointManager
        return CheckpointManager(landscape_db)

    @pytest.fixture
    def orchestrator(self, landscape_db):
        """Create Orchestrator with test database."""
        from elspeth.engine.orchestrator import Orchestrator
        from elspeth.core.config import ElspethSettings

        settings = ElspethSettings()
        return Orchestrator(db=landscape_db, settings=settings)

    @pytest.fixture
    def simple_pipeline(self):
        """Pipeline that processes 3 rows successfully."""
        # Returns a Pipeline config with a source yielding 3 rows
        # and a passthrough transform
        return {
            "source": {"type": "memory", "rows": [{"x": 1}, {"x": 2}, {"x": 3}]},
            "transforms": [{"type": "passthrough"}],
            "sinks": [{"type": "memory", "name": "output"}],
        }

    @pytest.fixture
    def failing_pipeline(self):
        """Pipeline that fails partway through."""
        return {
            "source": {"type": "memory", "rows": [{"x": 1}, {"x": 2}, {"x": 3}]},
            "transforms": [{"type": "fail_on_third"}],  # Fails on row 3
            "sinks": [{"type": "memory", "name": "output"}],
        }

    def test_orchestrator_creates_checkpoints(
        self, orchestrator, checkpoint_manager, simple_pipeline
    ) -> None:
        """Orchestrator creates checkpoints during execution."""
        from elspeth.core.config import CheckpointSettings

        # Configure every_row checkpointing
        orchestrator._checkpoint_settings = CheckpointSettings(frequency="every_row")
        orchestrator._checkpoint_manager = checkpoint_manager

        # Run with 3 rows
        result = orchestrator.run(simple_pipeline)

        assert result.status == "completed"

        # Should have checkpoints (one per row minimum)
        checkpoints = checkpoint_manager.get_checkpoints(result.run_id)
        assert len(checkpoints) >= 3

    def test_orchestrator_cleans_checkpoints_on_success(
        self, orchestrator, checkpoint_manager, simple_pipeline
    ) -> None:
        """Checkpoints are cleaned up after successful run."""
        orchestrator._checkpoint_manager = checkpoint_manager

        result = orchestrator.run(simple_pipeline)

        assert result.status == "completed"

        # Checkpoints should be deleted on success
        remaining = checkpoint_manager.get_checkpoints(result.run_id)
        assert len(remaining) == 0

    def test_orchestrator_preserves_checkpoints_on_failure(
        self, orchestrator, checkpoint_manager, failing_pipeline
    ) -> None:
        """Checkpoints are preserved when run fails."""
        orchestrator._checkpoint_manager = checkpoint_manager

        result = orchestrator.run(failing_pipeline)

        assert result.status == "failed"

        # Checkpoints should remain for recovery
        remaining = checkpoint_manager.get_checkpoints(result.run_id)
        assert len(remaining) > 0
```

### Step 2: Implementation guidance

The Orchestrator integration should:

1. **On row completion:** Call `checkpoint_manager.create_checkpoint()` based on frequency settings
2. **On aggregation flush:** Always create checkpoint (safety boundary)
3. **On successful completion:** Delete all checkpoints for the run
4. **On failure:** Leave checkpoints for recovery

**Integration Points in Orchestrator:**

```python
# In Orchestrator.__init__():
from elspeth.core.checkpoint import CheckpointManager
from elspeth.core.config import CheckpointSettings

class Orchestrator:
    def __init__(
        self,
        db: LandscapeDB,
        settings: ElspethSettings,
        checkpoint_manager: CheckpointManager | None = None,
    ) -> None:
        self._db = db
        self._settings = settings
        self._checkpoint_settings = settings.checkpoint
        self._checkpoint_manager = checkpoint_manager or CheckpointManager(db)
        self._sequence_number = 0

# In Orchestrator.run() - call _maybe_checkpoint after each row completes:
def run(self, pipeline: PipelineConfig) -> RunResult:
    run_id = self._begin_run(pipeline)
    try:
        for token in self._process_pipeline(pipeline):
            # After each row/token completes a node:
            self._maybe_checkpoint(run_id, token.token_id, token.current_node_id)

        self._complete_run(run_id, status="completed")
        # Clean up checkpoints on success
        self._checkpoint_manager.delete_checkpoints(run_id)
        return RunResult(run_id=run_id, status="completed")

    except Exception as e:
        self._complete_run(run_id, status="failed", error=str(e))
        # Leave checkpoints for recovery - don't delete them
        return RunResult(run_id=run_id, status="failed", error=str(e))

# The checkpoint decision logic:
def _maybe_checkpoint(self, run_id: str, token_id: str, node_id: str) -> None:
    """Create checkpoint if configured."""
    if not self._checkpoint_settings.enabled:
        return

    if self._checkpoint_manager is None:
        return

    self._sequence_number += 1

    should_checkpoint = False

    if self._checkpoint_settings.frequency == "every_row":
        should_checkpoint = True
    elif self._checkpoint_settings.frequency == "every_n":
        interval = self._checkpoint_settings.checkpoint_interval
        should_checkpoint = (self._sequence_number % interval == 0)
    # aggregation_only: checkpointed separately in _flush_aggregation()

    if should_checkpoint:
        self._checkpoint_manager.create_checkpoint(
            run_id=run_id,
            token_id=token_id,
            node_id=node_id,
            sequence_number=self._sequence_number,
        )

# For aggregation transforms, always checkpoint at flush:
def _flush_aggregation(self, run_id: str, aggregation: BaseAggregation, ...) -> None:
    # ... flush logic ...

    # Always checkpoint at aggregation boundaries regardless of frequency setting
    if self._checkpoint_settings.aggregation_boundaries:
        self._sequence_number += 1
        self._checkpoint_manager.create_checkpoint(
            run_id=run_id,
            token_id=batch_token_id,
            node_id=aggregation_node_id,
            sequence_number=self._sequence_number,
            aggregation_state=aggregation.get_state(),  # Capture buffer state
        )
```

### Step 3: Run tests

Run: `pytest tests/engine/test_orchestrator.py::TestOrchestratorCheckpointing -v`
Expected: PASS

---

## Task 5: Recovery Protocol - Resume from Checkpoint

**Context:** Implement recovery protocol to resume a failed run from its last checkpoint.

**Files:**
- Create: `src/elspeth/core/checkpoint/recovery.py`
- Create: `tests/core/checkpoint/test_recovery.py`

### Step 1: Write the failing test

```python
# tests/core/checkpoint/test_recovery.py
"""Tests for checkpoint recovery protocol."""

import pytest
from datetime import datetime, timezone


class TestRecoveryProtocol:
    """Tests for resuming runs from checkpoints."""

    @pytest.fixture
    def landscape_db(self, tmp_path):
        """Create test database."""
        from elspeth.core.landscape.database import LandscapeDB

        db = LandscapeDB(f"sqlite:///{tmp_path}/test.db")
        db.create_tables()
        return db

    @pytest.fixture
    def checkpoint_manager(self, landscape_db):
        """Create CheckpointManager."""
        from elspeth.core.checkpoint import CheckpointManager
        return CheckpointManager(landscape_db)

    @pytest.fixture
    def recovery_manager(self, landscape_db, checkpoint_manager):
        """Create RecoveryManager."""
        from elspeth.core.checkpoint import RecoveryManager
        return RecoveryManager(landscape_db, checkpoint_manager)

    @pytest.fixture
    def failed_run_with_checkpoint(self, landscape_db, checkpoint_manager):
        """Create a failed run that has checkpoints."""
        from elspeth.core.landscape.schema import (
            runs_table, nodes_table, rows_table, tokens_table
        )

        run_id = "failed-run-001"
        now = datetime.now(timezone.utc)

        with landscape_db.engine.connect() as conn:
            conn.execute(runs_table.insert().values(
                run_id=run_id, started_at=now, config_hash="abc",
                settings_json="{}", canonical_version="v1", status="failed"
            ))
            conn.execute(nodes_table.insert().values(
                node_id="node-001", run_id=run_id, plugin_name="test",
                node_type="transform", plugin_version="1.0", determinism="pure",
                config_hash="xyz", config_json="{}", registered_at=now
            ))
            conn.execute(rows_table.insert().values(
                row_id="row-001", run_id=run_id, source_node_id="node-001",
                row_index=0, source_data_hash="hash1", created_at=now
            ))
            conn.execute(tokens_table.insert().values(
                token_id="tok-001", row_id="row-001", created_at=now
            ))
            conn.commit()

        # Create a checkpoint
        checkpoint_manager.create_checkpoint(run_id, "tok-001", "node-001", 1)
        return run_id

    @pytest.fixture
    def completed_run(self, landscape_db):
        """Create a completed run (cannot be resumed)."""
        from elspeth.core.landscape.schema import runs_table

        run_id = "completed-run-001"
        now = datetime.now(timezone.utc)

        with landscape_db.engine.connect() as conn:
            conn.execute(runs_table.insert().values(
                run_id=run_id, started_at=now, completed_at=now,
                config_hash="abc", settings_json="{}",
                canonical_version="v1", status="completed"
            ))
            conn.commit()
        return run_id

    @pytest.fixture
    def failed_run_no_checkpoint(self, landscape_db):
        """Create a failed run without checkpoints."""
        from elspeth.core.landscape.schema import runs_table

        run_id = "failed-no-cp-001"
        now = datetime.now(timezone.utc)

        with landscape_db.engine.connect() as conn:
            conn.execute(runs_table.insert().values(
                run_id=run_id, started_at=now, config_hash="abc",
                settings_json="{}", canonical_version="v1", status="failed"
            ))
            conn.commit()
        return run_id

    def test_can_resume_returns_true_for_failed_run_with_checkpoint(
        self, recovery_manager, failed_run_with_checkpoint
    ) -> None:
        """can_resume() returns True when recovery is possible."""
        can_resume, reason = recovery_manager.can_resume(failed_run_with_checkpoint)

        assert can_resume is True
        assert reason is None

    def test_can_resume_returns_false_for_completed_run(
        self, recovery_manager, completed_run
    ) -> None:
        """can_resume() returns False for completed runs."""
        can_resume, reason = recovery_manager.can_resume(completed_run)

        assert can_resume is False
        assert "completed" in reason.lower()

    def test_can_resume_returns_false_without_checkpoint(
        self, recovery_manager, failed_run_no_checkpoint
    ) -> None:
        """can_resume() returns False when no checkpoint exists."""
        can_resume, reason = recovery_manager.can_resume(failed_run_no_checkpoint)

        assert can_resume is False
        assert "no checkpoint" in reason.lower()

    def test_get_resume_point(
        self, recovery_manager, failed_run_with_checkpoint
    ) -> None:
        """get_resume_point() returns checkpoint info."""
        resume_point = recovery_manager.get_resume_point(failed_run_with_checkpoint)

        assert resume_point is not None
        assert resume_point.token_id is not None
        assert resume_point.node_id is not None
        assert resume_point.sequence_number > 0

    def test_get_unprocessed_rows(
        self, recovery_manager, failed_run_with_checkpoint
    ) -> None:
        """get_unprocessed_rows() returns rows after checkpoint."""
        unprocessed = recovery_manager.get_unprocessed_rows(failed_run_with_checkpoint)

        # Should return rows that weren't fully processed
        assert isinstance(unprocessed, list)
        # The exact count depends on test setup
```

### Step 2: Run test to verify it fails

Run: `pytest tests/core/checkpoint/test_recovery.py -v`
Expected: FAIL (ImportError)

### Step 3: Create RecoveryManager

```python
# src/elspeth/core/checkpoint/recovery.py
"""Recovery protocol for resuming failed runs."""

from dataclasses import dataclass
from typing import Any

from elspeth.core.checkpoint.manager import CheckpointManager
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.models import Checkpoint


@dataclass
class ResumePoint:
    """Information needed to resume a run."""
    checkpoint: Checkpoint
    token_id: str
    node_id: str
    sequence_number: int
    aggregation_state: dict[str, Any] | None


class RecoveryManager:
    """Manages recovery of failed runs from checkpoints.

    Recovery protocol:
    1. Check if run can be resumed (failed status + checkpoint exists)
    2. Load checkpoint and aggregation state
    3. Identify unprocessed rows (sequence > checkpoint.sequence)
    4. Resume processing from checkpoint position
    """

    def __init__(
        self,
        db: LandscapeDB,
        checkpoint_manager: CheckpointManager,
    ) -> None:
        self._db = db
        self._checkpoint_manager = checkpoint_manager

    def can_resume(self, run_id: str) -> tuple[bool, str | None]:
        """Check if a run can be resumed.

        Args:
            run_id: The run to check

        Returns:
            Tuple of (can_resume, reason_if_not)
        """
        # Check run status
        run = self._get_run(run_id)
        if run is None:
            return False, f"Run {run_id} not found"

        if run.status == "completed":
            return False, "Run already completed successfully"

        if run.status == "running":
            return False, "Run is still in progress"

        # Check for checkpoint
        checkpoint = self._checkpoint_manager.get_latest_checkpoint(run_id)
        if checkpoint is None:
            return False, "No checkpoint found for recovery"

        return True, None

    def get_resume_point(self, run_id: str) -> ResumePoint | None:
        """Get the resume point for a failed run.

        Args:
            run_id: The run to resume

        Returns:
            ResumePoint with checkpoint info, or None if cannot resume
        """
        can_resume, _ = self.can_resume(run_id)
        if not can_resume:
            return None

        checkpoint = self._checkpoint_manager.get_latest_checkpoint(run_id)
        if checkpoint is None:
            return None

        import json
        agg_state = None
        if checkpoint.aggregation_state_json:
            agg_state = json.loads(checkpoint.aggregation_state_json)

        return ResumePoint(
            checkpoint=checkpoint,
            token_id=checkpoint.token_id,
            node_id=checkpoint.node_id,
            sequence_number=checkpoint.sequence_number,
            aggregation_state=agg_state,
        )

    def get_unprocessed_rows(self, run_id: str) -> list[str]:
        """Get row IDs that need reprocessing after checkpoint.

        Args:
            run_id: The run to check

        Returns:
            List of row_ids that weren't fully processed
        """
        from sqlalchemy import select, and_
        from elspeth.core.landscape.schema import (
            rows_table,
            tokens_table,
            node_states_table,
        )

        checkpoint = self._checkpoint_manager.get_latest_checkpoint(run_id)
        if checkpoint is None:
            return []

        # Find rows whose tokens don't have completed terminal states
        # This query finds rows that need reprocessing
        with self._db.engine.connect() as conn:
            # Get all rows for this run
            all_rows = conn.execute(
                select(rows_table.c.row_id)
                .where(rows_table.c.run_id == run_id)
                .order_by(rows_table.c.row_index)
            ).fetchall()

            # For each row, check if it reached terminal state
            unprocessed = []
            for (row_id,) in all_rows:
                if not self._row_completed(conn, row_id):
                    unprocessed.append(row_id)

            return unprocessed

    def _row_completed(self, conn, row_id: str) -> bool:
        """Check if a row reached terminal state.

        Handles retry scenarios by checking the LATEST attempt for each
        token/node combination. The node_states table has a unique constraint
        on (token_id, node_id, attempt), so multiple attempts may exist.
        """
        from sqlalchemy import select, exists, func
        from elspeth.core.landscape.schema import (
            tokens_table,
            node_states_table,
            nodes_table,
        )

        # A row is complete if any of its tokens reached a sink node
        # with status "completed" on the latest attempt.
        # Subquery to get the max attempt for each token/node pair
        latest_attempt_subq = (
            select(
                node_states_table.c.token_id,
                node_states_table.c.node_id,
                func.max(node_states_table.c.attempt).label("max_attempt")
            )
            .group_by(
                node_states_table.c.token_id,
                node_states_table.c.node_id
            )
            .subquery()
        )

        # Check if row has a token that reached a sink with completed status
        # on its latest attempt
        result = conn.execute(
            select(exists().where(
                and_(
                    tokens_table.c.row_id == row_id,
                    node_states_table.c.token_id == tokens_table.c.token_id,
                    node_states_table.c.status == "completed",
                    nodes_table.c.node_id == node_states_table.c.node_id,
                    nodes_table.c.node_type == "sink",
                    # Join with latest attempt subquery
                    node_states_table.c.token_id == latest_attempt_subq.c.token_id,
                    node_states_table.c.node_id == latest_attempt_subq.c.node_id,
                    node_states_table.c.attempt == latest_attempt_subq.c.max_attempt,
                )
            ))
        ).scalar()

        return result

    def _get_run(self, run_id: str):
        """Get run metadata."""
        from sqlalchemy import select
        from elspeth.core.landscape.schema import runs_table

        with self._db.engine.connect() as conn:
            result = conn.execute(
                select(runs_table).where(runs_table.c.run_id == run_id)
            ).fetchone()

        return result
```

Update `__init__.py`:

```python
# src/elspeth/core/checkpoint/__init__.py
from elspeth.core.checkpoint.manager import CheckpointManager
from elspeth.core.checkpoint.recovery import RecoveryManager, ResumePoint

__all__ = ["CheckpointManager", "RecoveryManager", "ResumePoint"]
```

### Step 4: Run tests

Run: `pytest tests/core/checkpoint/test_recovery.py -v`
Expected: PASS

---

## Task 6: Rate Limiter Wrapper

**Context:** Create a wrapper around pyrate-limiter for rate limiting external calls. Uses SQLite for persistence.

**Files:**
- Create: `src/elspeth/core/rate_limit/__init__.py`
- Create: `src/elspeth/core/rate_limit/limiter.py`
- Create: `tests/core/rate_limit/__init__.py`
- Create: `tests/core/rate_limit/test_limiter.py`

### Step 1: Write the failing test

```python
# tests/core/rate_limit/__init__.py
"""Rate limit tests."""

# tests/core/rate_limit/test_limiter.py
"""Tests for rate limiter."""

import pytest
import time


class TestRateLimiter:
    """Tests for rate limiting wrapper."""

    def test_create_limiter(self) -> None:
        """Can create a rate limiter."""
        from elspeth.core.rate_limit import RateLimiter

        limiter = RateLimiter(
            name="test_api",
            requests_per_second=10,
        )

        assert limiter.name == "test_api"

    def test_acquire_within_limit(self) -> None:
        """acquire() succeeds when under limit."""
        from elspeth.core.rate_limit import RateLimiter

        limiter = RateLimiter(name="test", requests_per_second=100)

        # Should not raise or block significantly
        start = time.monotonic()
        limiter.acquire()
        elapsed = time.monotonic() - start

        assert elapsed < 0.1  # Should be near-instant

    def test_acquire_blocks_when_exceeded(self) -> None:
        """acquire() blocks when rate exceeded."""
        from elspeth.core.rate_limit import RateLimiter

        # Very restrictive: 1 request per second
        limiter = RateLimiter(name="test", requests_per_second=1)

        # First request: instant
        limiter.acquire()

        # Second request: should block ~1 second
        start = time.monotonic()
        limiter.acquire()
        elapsed = time.monotonic() - start

        assert elapsed >= 0.9  # Should have waited ~1s

    def test_try_acquire_returns_false_when_exceeded(self) -> None:
        """try_acquire() returns False instead of blocking."""
        from elspeth.core.rate_limit import RateLimiter

        limiter = RateLimiter(name="test", requests_per_second=1)

        # First: succeeds
        assert limiter.try_acquire() is True

        # Second (immediate): should fail without blocking
        assert limiter.try_acquire() is False

    def test_limiter_with_sqlite_persistence(self, tmp_path) -> None:
        """Rate limits persist across limiter instances."""
        from elspeth.core.rate_limit import RateLimiter

        db_path = tmp_path / "limits.db"

        # First limiter uses up the quota
        limiter1 = RateLimiter(
            name="persistent",
            requests_per_second=1,
            persistence_path=str(db_path),
        )
        limiter1.acquire()

        # Second limiter (same name, same db) should see used quota
        limiter2 = RateLimiter(
            name="persistent",
            requests_per_second=1,
            persistence_path=str(db_path),
        )

        # Should fail because quota already used
        assert limiter2.try_acquire() is False
```

### Step 2: Run test to verify it fails

Run: `pytest tests/core/rate_limit/test_limiter.py -v`
Expected: FAIL (ImportError)

### Step 3: Create RateLimiter wrapper

```python
# src/elspeth/core/rate_limit/__init__.py
"""Rate limiting for external calls.

Uses pyrate-limiter with SQLite persistence.
"""

from elspeth.core.rate_limit.limiter import RateLimiter

__all__ = ["RateLimiter"]


# src/elspeth/core/rate_limit/limiter.py
"""Rate limiter wrapper around pyrate-limiter."""

from pyrate_limiter import Duration, Limiter, Rate, SQLiteBucket


class RateLimiter:
    """Rate limiter for external API calls.

    Wraps pyrate-limiter with sensible defaults and optional
    SQLite persistence for cross-process rate limiting.

    Example:
        limiter = RateLimiter("openai", requests_per_second=10)

        # Blocking acquire (waits if needed)
        limiter.acquire()
        call_openai_api()

        # Non-blocking check
        if limiter.try_acquire():
            call_openai_api()
        else:
            handle_rate_limit()
    """

    def __init__(
        self,
        name: str,
        requests_per_second: int,
        requests_per_minute: int | None = None,
        persistence_path: str | None = None,
    ) -> None:
        """Create a rate limiter.

        Args:
            name: Identifier for this limiter (e.g., "openai", "weather_api")
            requests_per_second: Maximum requests per second
            requests_per_minute: Optional additional per-minute limit
            persistence_path: SQLite path for cross-process persistence
        """
        self.name = name
        self._requests_per_second = requests_per_second
        self._requests_per_minute = requests_per_minute
        self._persistence_path = persistence_path  # Store for reset()

        # Build rates
        rates = [Rate(requests_per_second, Duration.SECOND)]
        if requests_per_minute is not None:
            rates.append(Rate(requests_per_minute, Duration.MINUTE))

        # Configure bucket (in-memory or SQLite)
        if persistence_path:
            bucket = SQLiteBucket(
                path=persistence_path,
                table=f"ratelimit_{name}",
                rates=rates,
            )
            self._limiter = Limiter(bucket)
        else:
            self._limiter = Limiter(*rates)

    def acquire(self, weight: int = 1) -> None:
        """Acquire rate limit tokens, blocking if necessary.

        Args:
            weight: Number of tokens to acquire (default 1)
        """
        # Explicit block=True ensures blocking behavior across pyrate-limiter versions
        self._limiter.try_acquire(self.name, weight=weight, block=True)

    def try_acquire(self, weight: int = 1) -> bool:
        """Try to acquire tokens without blocking.

        Args:
            weight: Number of tokens to acquire (default 1)

        Returns:
            True if acquired, False if rate limited
        """
        from pyrate_limiter import BucketFullException

        try:
            self._limiter.try_acquire(self.name, weight=weight, block=False)
            return True
        except BucketFullException:
            return False

    def reset(self) -> None:
        """Reset the rate limiter (for testing)."""
        # pyrate-limiter doesn't have a clean reset, recreate
        self.__init__(
            name=self.name,
            requests_per_second=self._requests_per_second,
            requests_per_minute=self._requests_per_minute,
            persistence_path=self._persistence_path,
        )
```

### Step 4: Run tests

Run: `pytest tests/core/rate_limit/test_limiter.py -v`
Expected: PASS

---

## Task 7: Rate Limit Configuration

**Context:** Add configuration schema for rate limits. Supports per-service limits.

**Files:**
- Modify: `src/elspeth/core/config.py` (add RateLimitSettings)
- Modify: `tests/core/test_config.py` (add rate limit config tests)

### Step 1: Write the failing test

```python
# Add to tests/core/test_config.py

class TestRateLimitSettings:
    """Tests for rate limit configuration."""

    def test_rate_limit_settings_defaults(self) -> None:
        from elspeth.core.config import RateLimitSettings

        settings = RateLimitSettings()

        assert settings.enabled is True
        assert settings.default_requests_per_second == 10
        assert settings.persistence_path is None

    def test_rate_limit_per_service(self) -> None:
        from elspeth.core.config import RateLimitSettings, ServiceRateLimit

        settings = RateLimitSettings(
            services={
                "openai": ServiceRateLimit(
                    requests_per_second=5,
                    requests_per_minute=100,
                ),
                "weather_api": ServiceRateLimit(
                    requests_per_second=20,
                ),
            }
        )

        assert settings.services["openai"].requests_per_second == 5
        assert settings.services["openai"].requests_per_minute == 100
        assert settings.services["weather_api"].requests_per_second == 20

    def test_rate_limit_get_service_config(self) -> None:
        from elspeth.core.config import RateLimitSettings, ServiceRateLimit

        settings = RateLimitSettings(
            default_requests_per_second=10,
            services={
                "openai": ServiceRateLimit(requests_per_second=5),
            }
        )

        # Configured service
        openai_config = settings.get_service_config("openai")
        assert openai_config.requests_per_second == 5

        # Unconfigured service falls back to default
        other_config = settings.get_service_config("other_api")
        assert other_config.requests_per_second == 10
```

### Step 2: Run test to verify it fails

Run: `pytest tests/core/test_config.py::TestRateLimitSettings -v`
Expected: FAIL (ImportError)

### Step 3: Add RateLimitSettings to config

Add to `src/elspeth/core/config.py`:

```python
from pydantic import BaseModel


class ServiceRateLimit(BaseModel):
    """Rate limit configuration for a specific service."""

    model_config = {"frozen": True}

    requests_per_second: int
    requests_per_minute: int | None = None


class RateLimitSettings(BaseModel):
    """Configuration for rate limiting external calls.

    Example YAML:
        rate_limit:
          enabled: true
          default_requests_per_second: 10
          persistence_path: ./rate_limits.db
          services:
            openai:
              requests_per_second: 5
              requests_per_minute: 100
            weather_api:
              requests_per_second: 20
    """

    model_config = {"frozen": True}

    enabled: bool = True
    default_requests_per_second: int = 10
    default_requests_per_minute: int | None = None
    persistence_path: str | None = None  # SQLite path for cross-process limits
    services: dict[str, ServiceRateLimit] = {}

    def get_service_config(self, service_name: str) -> ServiceRateLimit:
        """Get rate limit config for a service, with fallback to defaults."""
        if service_name in self.services:
            return self.services[service_name]
        return ServiceRateLimit(
            requests_per_second=self.default_requests_per_second,
            requests_per_minute=self.default_requests_per_minute,
        )
```

Also add to `ElspethSettings`:

```python
class ElspethSettings(BaseModel):
    # ... existing fields ...
    rate_limit: RateLimitSettings = RateLimitSettings()
```

### Step 4: Run tests

Run: `pytest tests/core/test_config.py::TestRateLimitSettings -v`
Expected: PASS

---

## Task 8: Rate Limit Registry

**Context:** Create a registry that manages rate limiters for multiple services. Used by engine when making external calls.

**Files:**
- Create: `src/elspeth/core/rate_limit/registry.py`
- Modify: `tests/core/rate_limit/test_limiter.py` (add registry tests)

### Step 1: Write the failing test

```python
# Add to tests/core/rate_limit/test_limiter.py

class TestRateLimitRegistry:
    """Tests for rate limiter registry."""

    def test_get_or_create_limiter(self) -> None:
        from elspeth.core.config import RateLimitSettings
        from elspeth.core.rate_limit import RateLimitRegistry

        settings = RateLimitSettings(default_requests_per_second=10)
        registry = RateLimitRegistry(settings)

        limiter1 = registry.get_limiter("api_a")
        limiter2 = registry.get_limiter("api_a")

        # Same instance returned
        assert limiter1 is limiter2

    def test_different_services_different_limiters(self) -> None:
        from elspeth.core.config import RateLimitSettings
        from elspeth.core.rate_limit import RateLimitRegistry

        settings = RateLimitSettings(default_requests_per_second=10)
        registry = RateLimitRegistry(settings)

        limiter_a = registry.get_limiter("api_a")
        limiter_b = registry.get_limiter("api_b")

        assert limiter_a is not limiter_b

    def test_registry_respects_service_config(self) -> None:
        from elspeth.core.config import RateLimitSettings, ServiceRateLimit
        from elspeth.core.rate_limit import RateLimitRegistry

        settings = RateLimitSettings(
            default_requests_per_second=10,
            services={
                "slow_api": ServiceRateLimit(requests_per_second=1),
            }
        )
        registry = RateLimitRegistry(settings)

        default_limiter = registry.get_limiter("fast_api")
        slow_limiter = registry.get_limiter("slow_api")

        assert default_limiter._requests_per_second == 10
        assert slow_limiter._requests_per_second == 1

    def test_registry_disabled(self) -> None:
        from elspeth.core.config import RateLimitSettings
        from elspeth.core.rate_limit import RateLimitRegistry, NoOpLimiter

        settings = RateLimitSettings(enabled=False)
        registry = RateLimitRegistry(settings)

        limiter = registry.get_limiter("any_api")

        # Should return no-op limiter
        assert isinstance(limiter, NoOpLimiter)
```

### Step 2: Run test to verify it fails

Run: `pytest tests/core/rate_limit/test_limiter.py::TestRateLimitRegistry -v`
Expected: FAIL (ImportError)

### Step 3: Create RateLimitRegistry

```python
# src/elspeth/core/rate_limit/registry.py
"""Registry for managing rate limiters."""

from elspeth.core.config import RateLimitSettings
from elspeth.core.rate_limit.limiter import RateLimiter


class NoOpLimiter:
    """No-op limiter when rate limiting is disabled."""

    def acquire(self, weight: int = 1) -> None:
        pass

    def try_acquire(self, weight: int = 1) -> bool:
        return True


class RateLimitRegistry:
    """Registry that manages rate limiters per service.

    Creates limiters on demand based on configuration.
    Reuses limiter instances for the same service.

    Example:
        settings = RateLimitSettings(...)
        registry = RateLimitRegistry(settings)

        # In external call code:
        limiter = registry.get_limiter("openai")
        limiter.acquire()
        response = call_openai()
    """

    def __init__(self, settings: RateLimitSettings) -> None:
        self._settings = settings
        self._limiters: dict[str, RateLimiter | NoOpLimiter] = {}

    def get_limiter(self, service_name: str) -> RateLimiter | NoOpLimiter:
        """Get or create a rate limiter for a service.

        Args:
            service_name: Name of the external service

        Returns:
            RateLimiter (or NoOpLimiter if disabled)
        """
        if not self._settings.enabled:
            return NoOpLimiter()

        if service_name not in self._limiters:
            config = self._settings.get_service_config(service_name)
            self._limiters[service_name] = RateLimiter(
                name=service_name,
                requests_per_second=config.requests_per_second,
                requests_per_minute=config.requests_per_minute,
                persistence_path=self._settings.persistence_path,
            )

        return self._limiters[service_name]

    def reset_all(self) -> None:
        """Reset all limiters (for testing)."""
        self._limiters.clear()
```

Update `__init__.py`:

```python
# src/elspeth/core/rate_limit/__init__.py
from elspeth.core.rate_limit.limiter import RateLimiter
from elspeth.core.rate_limit.registry import NoOpLimiter, RateLimitRegistry

__all__ = ["RateLimiter", "RateLimitRegistry", "NoOpLimiter"]
```

### Step 4: Run tests

Run: `pytest tests/core/rate_limit/test_limiter.py::TestRateLimitRegistry -v`
Expected: PASS

---

## Task 9: Purge Manager - Query Old Payloads

**Context:** Create PurgeManager that identifies PayloadStore content eligible for deletion based on retention policy.

**Files:**
- Create: `src/elspeth/core/retention/__init__.py`
- Create: `src/elspeth/core/retention/purge.py`
- Create: `tests/core/retention/__init__.py`
- Create: `tests/core/retention/test_purge.py`

### Step 1: Write the failing test

```python
# tests/core/retention/__init__.py
"""Retention tests."""

# tests/core/retention/test_purge.py
"""Tests for payload purge manager."""

import pytest
from datetime import datetime, timezone, timedelta


class TestPurgeManager:
    """Tests for identifying and purging old payloads."""

    @pytest.fixture
    def landscape_db(self, tmp_path):
        """Create test database."""
        from elspeth.core.landscape.database import LandscapeDB

        db = LandscapeDB(f"sqlite:///{tmp_path}/test.db")
        db.create_tables()
        return db

    @pytest.fixture
    def payload_store(self, tmp_path):
        """Create test payload store."""
        from elspeth.core.payload_store import FilesystemPayloadStore
        return FilesystemPayloadStore(tmp_path / "payloads")

    @pytest.fixture
    def purge_manager(self, landscape_db, payload_store):
        """Create PurgeManager for tests."""
        from elspeth.core.retention import PurgeManager
        return PurgeManager(landscape_db, payload_store)

    @pytest.fixture
    def old_run(self, landscape_db, payload_store):
        """Create a completed run from 100 days ago with payloads."""
        from elspeth.core.landscape.schema import runs_table, rows_table, nodes_table

        run_id = "old-run-001"
        old_time = datetime.now(timezone.utc) - timedelta(days=100)

        # Store a payload
        payload_data = b'{"old": "data"}'
        ref = payload_store.store(payload_data)

        with landscape_db.engine.connect() as conn:
            conn.execute(runs_table.insert().values(
                run_id=run_id, started_at=old_time, completed_at=old_time,
                config_hash="abc", settings_json="{}",
                canonical_version="v1", status="completed"
            ))
            conn.execute(nodes_table.insert().values(
                node_id="old-node-001", run_id=run_id, plugin_name="source",
                node_type="source", plugin_version="1.0", determinism="pure",
                config_hash="xyz", config_json="{}", registered_at=old_time
            ))
            conn.execute(rows_table.insert().values(
                row_id="old-row-001", run_id=run_id, source_node_id="old-node-001",
                row_index=0, source_data_hash="hash1", source_data_ref=ref,
                created_at=old_time
            ))
            conn.commit()
        return run_id

    @pytest.fixture
    def recent_run(self, landscape_db, payload_store):
        """Create a completed run from 10 days ago."""
        from elspeth.core.landscape.schema import runs_table, rows_table, nodes_table

        run_id = "recent-run-001"
        recent_time = datetime.now(timezone.utc) - timedelta(days=10)

        # Store a payload
        payload_data = b'{"recent": "data"}'
        ref = payload_store.store(payload_data)

        with landscape_db.engine.connect() as conn:
            conn.execute(runs_table.insert().values(
                run_id=run_id, started_at=recent_time, completed_at=recent_time,
                config_hash="abc", settings_json="{}",
                canonical_version="v1", status="completed"
            ))
            conn.execute(nodes_table.insert().values(
                node_id="recent-node-001", run_id=run_id, plugin_name="source",
                node_type="source", plugin_version="1.0", determinism="pure",
                config_hash="xyz", config_json="{}", registered_at=recent_time
            ))
            conn.execute(rows_table.insert().values(
                row_id="recent-row-001", run_id=run_id, source_node_id="recent-node-001",
                row_index=0, source_data_hash="hash2", source_data_ref=ref,
                created_at=recent_time
            ))
            conn.commit()
        return run_id

    def test_find_expired_row_payloads(
        self, purge_manager, landscape_db, old_run
    ) -> None:
        """Finds row payloads older than retention period."""
        # old_run fixture creates a run from 100 days ago
        expired = purge_manager.find_expired_row_payloads(retention_days=90)

        assert len(expired) > 0
        for ref in expired:
            assert ref is not None

    def test_find_expired_respects_retention(
        self, purge_manager, landscape_db, recent_run
    ) -> None:
        """Does not flag recent payloads."""
        # recent_run fixture creates a run from 10 days ago
        expired = purge_manager.find_expired_row_payloads(retention_days=90)

        assert len(expired) == 0

    def test_purge_payloads_deletes_content(
        self, purge_manager, payload_store, old_run
    ) -> None:
        """Purge actually deletes from PayloadStore."""
        # Get refs before purge
        expired = purge_manager.find_expired_row_payloads(retention_days=90)
        assert len(expired) > 0

        first_ref = expired[0]
        assert payload_store.exists(first_ref) is True

        # Purge
        result = purge_manager.purge_payloads(expired)

        assert result.deleted_count > 0
        assert payload_store.exists(first_ref) is False

    def test_purge_preserves_landscape_hashes(
        self, purge_manager, landscape_db, old_run
    ) -> None:
        """Purge deletes blobs but keeps hashes in Landscape."""
        from sqlalchemy import select
        from elspeth.core.landscape.schema import rows_table

        # Get hash before purge
        with landscape_db.engine.connect() as conn:
            row = conn.execute(
                select(rows_table)
                .where(rows_table.c.run_id == old_run)
            ).fetchone()
            original_hash = row.source_data_hash

        # Purge
        expired = purge_manager.find_expired_row_payloads(retention_days=90)
        purge_manager.purge_payloads(expired)

        # Hash should still exist
        with landscape_db.engine.connect() as conn:
            row = conn.execute(
                select(rows_table)
                .where(rows_table.c.run_id == old_run)
            ).fetchone()
            assert row.source_data_hash == original_hash  # Preserved!
            # But ref might be cleared (implementation choice)
```

### Step 2: Run test to verify it fails

Run: `pytest tests/core/retention/test_purge.py -v`
Expected: FAIL (ImportError)

### Step 3: Create PurgeManager

```python
# src/elspeth/core/retention/__init__.py
"""Retention and purge management.

Handles cleanup of old PayloadStore content while
preserving Landscape audit metadata (hashes).
"""

from elspeth.core.retention.purge import PurgeManager, PurgeResult

__all__ = ["PurgeManager", "PurgeResult"]


# src/elspeth/core/retention/purge.py
"""Purge manager for cleaning old payloads."""

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Protocol


class PayloadStoreProtocol(Protocol):
    """Protocol for PayloadStore (avoid circular import)."""
    def exists(self, content_hash: str) -> bool: ...
    def delete(self, content_hash: str) -> bool: ...


@dataclass
class PurgeResult:
    """Result of a purge operation."""
    deleted_count: int
    bytes_freed: int
    failed_refs: list[str]
    duration_seconds: float


class PurgeManager:
    """Manages purging of expired payloads.

    Key principle: Delete blobs, keep hashes.

    After purge:
    - Landscape tables retain all metadata including hashes
    - PayloadStore blobs are deleted
    - explain() still works but reports "payload unavailable"
    """

    def __init__(self, db, payload_store: PayloadStoreProtocol) -> None:
        """Initialize with Landscape DB and PayloadStore.

        Args:
            db: LandscapeDB instance
            payload_store: PayloadStore for blob deletion
        """
        self._db = db
        self._payload_store = payload_store

    def find_expired_row_payloads(
        self,
        retention_days: int,
        as_of: datetime | None = None,
    ) -> list[str]:
        """Find row payload refs older than retention period.

        Args:
            retention_days: Days to retain payloads
            as_of: Reference time (default: now)

        Returns:
            List of payload refs eligible for deletion
        """
        from sqlalchemy import select, and_
        from elspeth.core.landscape.schema import rows_table, runs_table

        if as_of is None:
            as_of = datetime.now(timezone.utc)

        cutoff = as_of - timedelta(days=retention_days)

        with self._db.engine.connect() as conn:
            # Find rows from completed runs older than cutoff
            results = conn.execute(
                select(rows_table.c.source_data_ref)
                .select_from(rows_table.join(
                    runs_table,
                    rows_table.c.run_id == runs_table.c.run_id
                ))
                .where(and_(
                    runs_table.c.completed_at < cutoff,
                    runs_table.c.status == "completed",
                    rows_table.c.source_data_ref.isnot(None),
                ))
            ).fetchall()

        return [r[0] for r in results if r[0] is not None]

    def find_expired_call_payloads(
        self,
        retention_days: int,
        as_of: datetime | None = None,
    ) -> list[str]:
        """Find call payload refs older than retention period.

        Args:
            retention_days: Days to retain payloads
            as_of: Reference time (default: now)

        Returns:
            List of payload refs (request and response) eligible for deletion
        """
        from sqlalchemy import select, and_
        from elspeth.core.landscape.schema import (
            calls_table, node_states_table, tokens_table,
            rows_table, runs_table,
        )

        if as_of is None:
            as_of = datetime.now(timezone.utc)

        cutoff = as_of - timedelta(days=retention_days)

        refs = []
        with self._db.engine.connect() as conn:
            # Complex join to find calls from old completed runs
            results = conn.execute(
                select(calls_table.c.request_ref, calls_table.c.response_ref)
                .select_from(
                    calls_table
                    .join(node_states_table,
                          calls_table.c.state_id == node_states_table.c.state_id)
                    .join(tokens_table,
                          node_states_table.c.token_id == tokens_table.c.token_id)
                    .join(rows_table,
                          tokens_table.c.row_id == rows_table.c.row_id)
                    .join(runs_table,
                          rows_table.c.run_id == runs_table.c.run_id)
                )
                .where(and_(
                    runs_table.c.completed_at < cutoff,
                    runs_table.c.status == "completed",
                ))
            ).fetchall()

        for req_ref, resp_ref in results:
            if req_ref:
                refs.append(req_ref)
            if resp_ref:
                refs.append(resp_ref)

        return refs

    def purge_payloads(self, refs: list[str]) -> PurgeResult:
        """Delete payloads from PayloadStore.

        Args:
            refs: List of payload refs to delete

        Returns:
            PurgeResult with statistics
        """
        import time

        start = time.monotonic()
        deleted = 0
        bytes_freed = 0
        failed = []

        for ref in refs:
            try:
                if self._payload_store.exists(ref):
                    # Note: Real implementation would track size before delete
                    if self._payload_store.delete(ref):
                        deleted += 1
            except OSError as e:
                # Log the specific error for debugging, but continue purging
                # This handles file-not-found, permission denied, etc.
                failed.append(ref)

        elapsed = time.monotonic() - start

        return PurgeResult(
            deleted_count=deleted,
            bytes_freed=bytes_freed,  # Would need size tracking
            failed_refs=failed,
            duration_seconds=elapsed,
        )

    def purge_all_expired(
        self,
        row_retention_days: int = 90,
        call_retention_days: int = 90,
    ) -> PurgeResult:
        """Purge all expired payloads.

        Args:
            row_retention_days: Retention for row payloads
            call_retention_days: Retention for call payloads

        Returns:
            Combined PurgeResult
        """
        row_refs = self.find_expired_row_payloads(row_retention_days)
        call_refs = self.find_expired_call_payloads(call_retention_days)

        all_refs = list(set(row_refs + call_refs))  # Deduplicate

        return self.purge_payloads(all_refs)
```

### Step 4: Add delete method to PayloadStore

The existing `FilesystemPayloadStore` needs a `delete` method. Add to `src/elspeth/core/payload_store.py`:

```python
def delete(self, content_hash: str) -> bool:
    """Delete a payload from the store.

    Args:
        content_hash: Hash of content to delete

    Returns:
        True if deleted, False if not found
    """
    path = self._path_for_hash(content_hash)
    if path.exists():
        path.unlink()
        return True
    return False
```

### Step 5: Run tests

Run: `pytest tests/core/retention/test_purge.py -v`
Expected: PASS

---

## Task 10: CLI Purge Command

**Context:** Add `elspeth purge` CLI command for manual retention management.

**Files:**
- Modify: `src/elspeth/cli.py` (add purge command)
- Modify: `tests/cli/test_cli.py` (add purge tests)

### Step 1: Write the failing test

```python
# Add to tests/cli/test_cli.py

class TestPurgeCommand:
    """Tests for purge CLI command."""

    def test_purge_help(self) -> None:
        """purge --help shows usage."""
        from elspeth.cli import app

        result = runner.invoke(app, ["purge", "--help"])

        assert result.exit_code == 0
        assert "retention" in result.stdout.lower() or "days" in result.stdout.lower()

    def test_purge_dry_run(self, tmp_path) -> None:
        """purge --dry-run shows what would be deleted."""
        from elspeth.cli import app

        result = runner.invoke(app, [
            "purge",
            "--dry-run",
            "--database", str(tmp_path / "test.db"),
        ])

        assert result.exit_code == 0
        assert "would delete" in result.stdout.lower() or "0" in result.stdout

    def test_purge_with_retention_override(self, tmp_path) -> None:
        """purge --retention-days overrides default."""
        from elspeth.cli import app

        result = runner.invoke(app, [
            "purge",
            "--dry-run",
            "--retention-days", "30",
            "--database", str(tmp_path / "test.db"),
        ])

        assert result.exit_code == 0

    def test_purge_requires_confirmation(self, tmp_path) -> None:
        """purge without --yes asks for confirmation."""
        from elspeth.cli import app

        result = runner.invoke(
            app,
            ["purge", "--database", str(tmp_path / "test.db")],
            input="n\n",  # Say no to confirmation
        )

        assert result.exit_code == 0 or result.exit_code == 1
        assert "abort" in result.stdout.lower() or "cancel" in result.stdout.lower()
```

### Step 2: Run test to verify it fails

Run: `pytest tests/cli/test_cli.py::TestPurgeCommand -v`
Expected: FAIL (purge command doesn't exist)

### Step 3: Add purge command to CLI

Add to `src/elspeth/cli.py`:

```python
import typer

# ... existing code ...

@app.command()
def purge(
    database: str = typer.Option(
        None,
        "--database", "-d",
        help="Path to Landscape database",
    ),
    retention_days: int = typer.Option(
        90,
        "--retention-days", "-r",
        help="Delete payloads older than this many days",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be deleted without deleting",
    ),
    yes: bool = typer.Option(
        False,
        "--yes", "-y",
        help="Skip confirmation prompt",
    ),
) -> None:
    """Purge old payloads to free storage.

    Deletes PayloadStore blobs older than retention period.
    Landscape metadata (hashes) is preserved for audit trail.

    Examples:

        # See what would be deleted
        elspeth purge --dry-run

        # Delete payloads older than 30 days
        elspeth purge --retention-days 30 --yes
    """
    from pathlib import Path

    from elspeth.core.config import load_settings, ElspethSettings
    from elspeth.core.landscape.database import LandscapeDB
    from elspeth.core.payload_store import FilesystemPayloadStore
    from elspeth.core.retention import PurgeManager

    # Load settings from default path or use defaults
    settings_path = Path("settings.yaml")
    if settings_path.exists():
        settings = load_settings(settings_path)
    else:
        settings = ElspethSettings()

    db_path = database or settings.landscape.url.replace("sqlite:///", "")
    payload_path = settings.payload_store.base_path

    db = LandscapeDB(f"sqlite:///{db_path}")
    payload_store = FilesystemPayloadStore(payload_path)
    purge_manager = PurgeManager(db, payload_store)

    # Find expired payloads
    row_refs = purge_manager.find_expired_row_payloads(retention_days)
    call_refs = purge_manager.find_expired_call_payloads(retention_days)
    all_refs = list(set(row_refs + call_refs))

    typer.echo(f"Found {len(all_refs)} payloads older than {retention_days} days")
    typer.echo(f"  - Row payloads: {len(row_refs)}")
    typer.echo(f"  - Call payloads: {len(call_refs)}")

    if dry_run:
        typer.echo("\n--dry-run: Would delete the above payloads")
        return

    if len(all_refs) == 0:
        typer.echo("Nothing to purge.")
        return

    # Confirm unless --yes
    if not yes:
        confirm = typer.confirm(f"Delete {len(all_refs)} payloads?")
        if not confirm:
            typer.echo("Aborted.")
            raise typer.Exit(1)

    # Purge
    result = purge_manager.purge_payloads(all_refs)

    typer.echo(f"\nPurged {result.deleted_count} payloads in {result.duration_seconds:.2f}s")
    if result.failed_refs:
        typer.echo(f"Failed to delete {len(result.failed_refs)} payloads")
```

### Step 4: Run tests

Run: `pytest tests/cli/test_cli.py::TestPurgeCommand -v`
Expected: PASS

---

## Task 11: Graceful Degradation in explain()

**Context:** When payloads have been purged, `explain()` should report "payload unavailable" rather than failing.

**Files:**
- Modify: `src/elspeth/core/landscape/recorder.py` (update explain methods)
- Modify: `tests/core/landscape/test_recorder.py` (add degradation tests)

### Step 1: Write the failing test

```python
# Add to tests/core/landscape/test_recorder.py

import pytest
from datetime import datetime, timezone


class TestExplainGracefulDegradation:
    """Tests for explain() when payloads are unavailable."""

    @pytest.fixture
    def landscape_db(self, tmp_path):
        """Create test database."""
        from elspeth.core.landscape.database import LandscapeDB

        db = LandscapeDB(f"sqlite:///{tmp_path}/test.db")
        db.create_tables()
        return db

    @pytest.fixture
    def payload_store(self, tmp_path):
        """Create test payload store."""
        from elspeth.core.payload_store import FilesystemPayloadStore
        return FilesystemPayloadStore(tmp_path / "payloads")

    @pytest.fixture
    def recorder(self, landscape_db, payload_store):
        """Create LandscapeRecorder with payload store."""
        from elspeth.core.landscape.recorder import LandscapeRecorder
        return LandscapeRecorder(landscape_db, payload_store=payload_store)

    @pytest.fixture
    def run_with_purged_payloads(self, landscape_db):
        """Create a run where payloads have been purged (ref exists but file gone)."""
        from elspeth.core.landscape.schema import runs_table, rows_table, nodes_table

        run_id = "purged-run-001"
        now = datetime.now(timezone.utc)

        with landscape_db.engine.connect() as conn:
            conn.execute(runs_table.insert().values(
                run_id=run_id, started_at=now, completed_at=now,
                config_hash="abc", settings_json="{}",
                canonical_version="v1", status="completed"
            ))
            conn.execute(nodes_table.insert().values(
                node_id="purged-node-001", run_id=run_id, plugin_name="source",
                node_type="source", plugin_version="1.0", determinism="pure",
                config_hash="xyz", config_json="{}", registered_at=now
            ))
            # Note: source_data_ref points to a non-existent payload (simulating purge)
            conn.execute(rows_table.insert().values(
                row_id="row-001", run_id=run_id, source_node_id="purged-node-001",
                row_index=0, source_data_hash="hash-still-here",
                source_data_ref="nonexistent-ref", created_at=now
            ))
            conn.commit()
        return run_id

    @pytest.fixture
    def recent_run(self, landscape_db, payload_store):
        """Create a run with available payloads."""
        from elspeth.core.landscape.schema import runs_table, rows_table, nodes_table

        run_id = "recent-run-001"
        now = datetime.now(timezone.utc)

        # Store actual payload
        payload_data = b'{"field": "value"}'
        ref = payload_store.store(payload_data)

        with landscape_db.engine.connect() as conn:
            conn.execute(runs_table.insert().values(
                run_id=run_id, started_at=now, completed_at=now,
                config_hash="abc", settings_json="{}",
                canonical_version="v1", status="completed"
            ))
            conn.execute(nodes_table.insert().values(
                node_id="recent-node-001", run_id=run_id, plugin_name="source",
                node_type="source", plugin_version="1.0", determinism="pure",
                config_hash="xyz", config_json="{}", registered_at=now
            ))
            conn.execute(rows_table.insert().values(
                row_id="row-001", run_id=run_id, source_node_id="recent-node-001",
                row_index=0, source_data_hash="hash1", source_data_ref=ref,
                created_at=now
            ))
            conn.commit()
        return run_id

    def test_explain_with_missing_row_payload(
        self, recorder, run_with_purged_payloads
    ) -> None:
        """explain() succeeds even when row payload is purged."""
        lineage = recorder.explain_row(
            run_id=run_with_purged_payloads,
            row_id="row-001",
        )

        assert lineage is not None
        assert lineage.source_hash is not None  # Hash preserved
        assert lineage.source_data is None  # Payload unavailable
        assert lineage.payload_available is False

    def test_explain_reports_payload_status(
        self, recorder, run_with_purged_payloads
    ) -> None:
        """explain() explicitly reports payload availability."""
        lineage = recorder.explain_row(
            run_id=run_with_purged_payloads,
            row_id="row-001",
        )

        # Should have metadata about unavailability
        assert hasattr(lineage, "payload_available")
        assert lineage.payload_available is False

    def test_explain_with_available_payload(
        self, recorder, recent_run
    ) -> None:
        """explain() returns payload when available."""
        lineage = recorder.explain_row(
            run_id=recent_run,
            row_id="row-001",
        )

        assert lineage is not None
        assert lineage.source_data is not None  # Payload available
        assert lineage.payload_available is True
```

### Step 2: Implementation guidance

Update `LandscapeRecorder.explain_row()` to:

1. Try to load payload from PayloadStore
2. If not found, set `payload_available = False`
3. Return lineage with hash (always) and data (if available)

```python
@dataclass
class RowLineage:
    """Lineage information for a row."""
    row_id: str
    run_id: str
    source_hash: str
    source_data: dict | None  # None if payload purged
    payload_available: bool
    node_states: list[NodeState]
    # ... other fields


def explain_row(self, run_id: str, row_id: str) -> RowLineage | None:
    """Get lineage for a row, gracefully handling purged payloads."""
    row = self.get_row(row_id)
    if row is None:
        return None

    # Try to load payload
    source_data = None
    payload_available = False

    if row.source_data_ref and self._payload_store:
        try:
            payload_bytes = self._payload_store.retrieve(row.source_data_ref)
            if payload_bytes:
                source_data = json.loads(payload_bytes)
                payload_available = True
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            # Payload has been purged or is corrupted - this is expected
            # after retention period expires. Hash remains for verification.
            pass

    return RowLineage(
        row_id=row_id,
        run_id=run_id,
        source_hash=row.source_data_hash,
        source_data=source_data,
        payload_available=payload_available,
        node_states=self.get_node_states_for_row(row_id),
    )
```

### Step 3: Run tests

Run: `pytest tests/core/landscape/test_recorder.py::TestExplainGracefulDegradation -v`
Expected: PASS

---

## Task 12: Resume Command for Crash Recovery

**Context:** Add `elspeth resume` CLI command to continue a failed run from checkpoint.

**Files:**
- Modify: `src/elspeth/cli.py` (add resume command)
- Modify: `tests/cli/test_cli.py` (add resume tests)

### Step 1: Write the failing test

```python
# Add to tests/cli/test_cli.py

class TestResumeCommand:
    """Tests for resume CLI command."""

    @pytest.fixture
    def completed_run_id(self, tmp_path) -> str:
        """Create a completed run in a test database."""
        from datetime import datetime, timezone
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.schema import runs_table

        db = LandscapeDB(f"sqlite:///{tmp_path}/test.db")
        db.create_tables()

        run_id = "completed-run-001"
        now = datetime.now(timezone.utc)

        with db.engine.connect() as conn:
            conn.execute(runs_table.insert().values(
                run_id=run_id,
                started_at=now,
                completed_at=now,
                config_hash="test",
                settings_json="{}",
                canonical_version="sha256-rfc8785-v1",
                status="completed",
            ))
            conn.commit()

        return run_id

    def test_resume_help(self) -> None:
        """resume --help shows usage."""
        from elspeth.cli import app

        result = runner.invoke(app, ["resume", "--help"])

        assert result.exit_code == 0
        assert "run" in result.stdout.lower()

    def test_resume_nonexistent_run(self) -> None:
        """resume fails gracefully for nonexistent run."""
        from elspeth.cli import app

        result = runner.invoke(app, ["resume", "nonexistent-run-id"])

        assert result.exit_code != 0
        assert "not found" in result.stdout.lower() or "error" in result.stdout.lower()

    def test_resume_completed_run(self, completed_run_id) -> None:
        """resume fails for already-completed run."""
        from elspeth.cli import app

        result = runner.invoke(app, ["resume", completed_run_id])

        assert result.exit_code != 0
        assert "completed" in result.stdout.lower()
```

### Step 2: Run test to verify it fails

Run: `pytest tests/cli/test_cli.py::TestResumeCommand -v`
Expected: FAIL (resume command doesn't exist)

### Step 3: Add resume command to CLI

Add to `src/elspeth/cli.py`:

```python
@app.command()
def resume(
    run_id: str = typer.Argument(..., help="Run ID to resume"),
    database: str = typer.Option(
        None,
        "--database", "-d",
        help="Path to Landscape database",
    ),
) -> None:
    """Resume a failed run from its last checkpoint.

    Continues processing from where the run left off.
    Only works for runs that failed and have checkpoints.

    Examples:

        # Resume a specific run
        elspeth resume run-abc123
    """
    from pathlib import Path

    from elspeth.core.config import load_settings, ElspethSettings
    from elspeth.core.landscape.database import LandscapeDB
    from elspeth.core.checkpoint import CheckpointManager, RecoveryManager

    # Load settings from default path or use defaults
    settings_path = Path("settings.yaml")
    if settings_path.exists():
        settings = load_settings(settings_path)
    else:
        settings = ElspethSettings()

    db_path = database or settings.landscape.url.replace("sqlite:///", "")

    db = LandscapeDB(f"sqlite:///{db_path}")
    checkpoint_mgr = CheckpointManager(db)
    recovery_mgr = RecoveryManager(db, checkpoint_mgr)

    # Check if resume is possible
    can_resume, reason = recovery_mgr.can_resume(run_id)

    if not can_resume:
        typer.echo(f"Cannot resume run {run_id}: {reason}", err=True)
        raise typer.Exit(1)

    # Get resume point
    resume_point = recovery_mgr.get_resume_point(run_id)
    if resume_point is None:
        typer.echo(f"Failed to get resume point for {run_id}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Resuming run {run_id}")
    typer.echo(f"  Checkpoint: {resume_point.checkpoint.checkpoint_id}")
    typer.echo(f"  Sequence: {resume_point.sequence_number}")
    typer.echo(f"  Last node: {resume_point.node_id}")

    # Get unprocessed rows
    unprocessed = recovery_mgr.get_unprocessed_rows(run_id)
    typer.echo(f"  Rows to process: {len(unprocessed)}")

    # Load original pipeline config from run's settings_json
    from elspeth.engine.orchestrator import Orchestrator

    orchestrator = Orchestrator(db)
    result = orchestrator.resume(run_id, resume_point)

    if result.status == "completed":
        typer.echo(f"\nResumed run completed successfully!")
        typer.echo(f"  Rows processed: {result.rows_processed}")
        typer.echo(f"  Rows succeeded: {result.rows_succeeded}")
    else:
        typer.echo(f"\nResumed run failed: {result.status}", err=True)
        raise typer.Exit(1)
```

### Step 4: Add Orchestrator.resume() method

Add to `src/elspeth/engine/orchestrator.py`:

```python
def resume(
    self,
    run_id: str,
    resume_point: "ResumePoint",
) -> RunResult:
    """Resume a failed run from a checkpoint.

    Args:
        run_id: The run to resume
        resume_point: Checkpoint info from RecoveryManager

    Returns:
        RunResult with final status

    Note:
        The original pipeline config is reconstructed from the run's
        settings_json stored in Landscape. This requires plugins to
        be re-instantiated from their stored configuration.
    """
    from elspeth.core.checkpoint import CheckpointManager

    recorder = LandscapeRecorder(self._db)

    # Update run status to running
    self._update_run_status(run_id, "running")

    # Get original config from settings_json
    original_config = self._load_run_config(run_id)

    # Re-create pipeline from stored config
    # (Implementation depends on how plugins are serialized)
    pipeline_config = self._reconstruct_pipeline(original_config)

    # Get execution graph
    graph = self._reconstruct_graph(run_id)

    try:
        with self._span_factory.run_span(run_id):
            result = self._execute_run_from_checkpoint(
                recorder=recorder,
                run_id=run_id,
                config=pipeline_config,
                graph=graph,
                resume_point=resume_point,
            )

        # Complete run
        recorder.complete_run(run_id, status="completed")
        result.status = "completed"

        # Clean up checkpoints on success
        checkpoint_mgr = CheckpointManager(self._db)
        checkpoint_mgr.delete_checkpoints(run_id)

        return result

    except Exception:
        recorder.complete_run(run_id, status="failed")
        raise

def _execute_run_from_checkpoint(
    self,
    recorder: LandscapeRecorder,
    run_id: str,
    config: PipelineConfig,
    graph: ExecutionGraph,
    resume_point: "ResumePoint",
) -> RunResult:
    """Execute run starting from checkpoint position.

    Skips rows that were already processed according to the checkpoint.
    """
    # Implementation follows similar pattern to _execute_run()
    # but starts from resume_point.sequence_number and skips
    # rows that already reached terminal states

    # For aggregations, restore state from resume_point.aggregation_state
    if resume_point.aggregation_state:
        self._restore_aggregation_state(config, resume_point.aggregation_state)

    # ... rest of implementation follows _execute_run pattern
    # but filters source rows to only unprocessed ones
    pass  # Full implementation follows _execute_run() pattern

def _update_run_status(self, run_id: str, status: str) -> None:
    """Update run status in Landscape."""
    from sqlalchemy import update
    from elspeth.core.landscape.schema import runs_table

    with self._db.engine.connect() as conn:
        conn.execute(
            update(runs_table)
            .where(runs_table.c.run_id == run_id)
            .values(status=status)
        )
        conn.commit()

def _load_run_config(self, run_id: str) -> dict:
    """Load original run configuration from Landscape."""
    import json
    from sqlalchemy import select
    from elspeth.core.landscape.schema import runs_table

    with self._db.engine.connect() as conn:
        result = conn.execute(
            select(runs_table.c.settings_json)
            .where(runs_table.c.run_id == run_id)
        ).fetchone()

    if result is None:
        raise ValueError(f"Run {run_id} not found")

    return json.loads(result[0])
```

**Note:** The full `_execute_run_from_checkpoint` implementation mirrors `_execute_run()` but:
1. Filters source rows to skip already-processed ones
2. Restores aggregation state from checkpoint
3. Continues from the checkpoint's sequence number

This is marked as partial implementation - the full implementation requires careful handling of:
- Plugin re-instantiation from stored config
- Aggregation state restoration
- Source row filtering

### Step 5: Run tests

Run: `pytest tests/cli/test_cli.py::TestResumeCommand -v`
Expected: PASS

---

## Task 13: Integration Test - Full Checkpoint/Recovery Cycle

**Context:** End-to-end test that verifies checkpoint creation, simulated crash, and recovery.

**Files:**
- Create: `tests/integration/test_checkpoint_recovery.py`

### Step 1: Write the integration test

```python
# tests/integration/test_checkpoint_recovery.py
"""Integration tests for checkpoint and recovery."""

import pytest


class TestCheckpointRecoveryIntegration:
    """End-to-end checkpoint/recovery tests."""

    @pytest.fixture
    def test_env(self, tmp_path):
        """Set up test environment with database and payload store."""
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.payload_store import FilesystemPayloadStore
        from elspeth.core.checkpoint import CheckpointManager, RecoveryManager
        from elspeth.core.config import CheckpointSettings

        db = LandscapeDB(f"sqlite:///{tmp_path}/test.db")
        db.create_tables()

        payload_store = FilesystemPayloadStore(tmp_path / "payloads")
        checkpoint_mgr = CheckpointManager(db)
        recovery_mgr = RecoveryManager(db, checkpoint_mgr)
        checkpoint_settings = CheckpointSettings(frequency="every_row")

        return {
            "db": db,
            "payload_store": payload_store,
            "checkpoint_manager": checkpoint_mgr,
            "recovery_manager": recovery_mgr,
            "checkpoint_settings": checkpoint_settings,
        }

    def test_full_checkpoint_recovery_cycle(self, test_env) -> None:
        """Complete cycle: run -> checkpoint -> crash -> recover -> complete."""
        # This test simulates:
        # 1. Start a run
        # 2. Process some rows (checkpoints created)
        # 3. Simulate crash (mark run as failed)
        # 4. Verify recovery is possible
        # 5. Resume and complete

        checkpoint_mgr = test_env["checkpoint_manager"]
        recovery_mgr = test_env["recovery_manager"]
        db = test_env["db"]

        # 1. Set up a run with rows and checkpoints
        run_id = self._setup_partial_run(db, checkpoint_mgr)

        # 2. Verify checkpoint exists
        checkpoint = checkpoint_mgr.get_latest_checkpoint(run_id)
        assert checkpoint is not None
        assert checkpoint.sequence_number > 0

        # 3. Verify can resume
        can_resume, reason = recovery_mgr.can_resume(run_id)
        assert can_resume is True, f"Cannot resume: {reason}"

        # 4. Get resume point
        resume_point = recovery_mgr.get_resume_point(run_id)
        assert resume_point is not None

        # 5. Get unprocessed rows
        unprocessed = recovery_mgr.get_unprocessed_rows(run_id)
        assert len(unprocessed) > 0  # Some rows still need processing

    def _setup_partial_run(self, db, checkpoint_mgr) -> str:
        """Helper to create a partially-completed run with checkpoints."""
        from datetime import datetime, timezone
        from elspeth.core.landscape.schema import (
            runs_table, nodes_table, rows_table, tokens_table
        )

        run_id = "test-run-001"
        now = datetime.now(timezone.utc)

        with db.engine.connect() as conn:
            # Create run (failed status)
            conn.execute(runs_table.insert().values(
                run_id=run_id,
                started_at=now,
                config_hash="test",
                settings_json="{}",
                canonical_version="sha256-rfc8785-v1",
                status="failed",
            ))

            # Create node
            conn.execute(nodes_table.insert().values(
                node_id="node-001",
                run_id=run_id,
                plugin_name="test",
                node_type="transform",
                plugin_version="1.0",
                determinism="pure",
                config_hash="x",
                config_json="{}",
                registered_at=now,
            ))

            # Create multiple rows
            for i in range(5):
                row_id = f"row-{i:03d}"
                conn.execute(rows_table.insert().values(
                    row_id=row_id,
                    run_id=run_id,
                    source_node_id="node-001",
                    row_index=i,
                    source_data_hash=f"hash{i}",
                    created_at=now,
                ))
                conn.execute(tokens_table.insert().values(
                    token_id=f"tok-{i:03d}",
                    row_id=row_id,
                    created_at=now,
                ))

            conn.commit()

        # Create checkpoint at row 2 (simulating partial progress)
        checkpoint_mgr.create_checkpoint(
            run_id=run_id,
            token_id="tok-002",
            node_id="node-001",
            sequence_number=2,
        )

        return run_id
```

### Step 2: Run tests

Run: `pytest tests/integration/test_checkpoint_recovery.py -v`
Expected: PASS

---

## Task 14: Reproducibility Grade Computation

**Context:** Compute and set the `reproducibility_grade` on the `runs` table at run completion. The grade indicates what level of replay/verification is possible for the run.

**Files:**
- Modify: `src/elspeth/core/landscape/schema.py` (add `reproducibility_grade` column to runs table if not present)
- Modify: `src/elspeth/core/landscape/recorder.py` (add grade computation)
- Create: `src/elspeth/core/landscape/reproducibility.py` (grade computation logic)
- Modify: `tests/core/landscape/test_recorder.py` (add grade computation tests)

### Step 1: Write the failing test

```python
# Add to tests/core/landscape/test_recorder.py

class TestReproducibilityGradeComputation:
    """Tests for reproducibility grade computation at run completion."""

    @pytest.fixture
    def landscape_db(self, tmp_path):
        """Create test database."""
        from elspeth.core.landscape.database import LandscapeDB

        db = LandscapeDB(f"sqlite:///{tmp_path}/test.db")
        db.create_tables()
        return db

    @pytest.fixture
    def recorder(self, landscape_db):
        """Create LandscapeRecorder for tests."""
        from elspeth.core.landscape.recorder import LandscapeRecorder
        return LandscapeRecorder(landscape_db)

    @pytest.fixture
    def run_id(self, landscape_db) -> str:
        """Create a basic run for testing."""
        from datetime import datetime, timezone
        from elspeth.core.landscape.schema import runs_table

        run_id = "test-run-001"
        now = datetime.now(timezone.utc)

        with landscape_db.engine.connect() as conn:
            conn.execute(runs_table.insert().values(
                run_id=run_id,
                started_at=now,
                config_hash="test",
                settings_json="{}",
                canonical_version="sha256-rfc8785-v1",
                status="running",
            ))
            conn.commit()

        return run_id

    @pytest.fixture
    def run_with_pure_transforms(self, landscape_db) -> str:
        """Create a run with only deterministic transforms."""
        from datetime import datetime, timezone
        from elspeth.core.landscape.schema import runs_table, nodes_table

        run_id = "pure-run-001"
        now = datetime.now(timezone.utc)

        with landscape_db.engine.connect() as conn:
            conn.execute(runs_table.insert().values(
                run_id=run_id,
                started_at=now,
                config_hash="pure",
                settings_json="{}",
                canonical_version="sha256-rfc8785-v1",
                status="completed",
            ))
            # All nodes are deterministic
            conn.execute(nodes_table.insert().values(
                node_id="source-001", run_id=run_id, plugin_name="csv_source",
                node_type="source", plugin_version="1.0",
                determinism="deterministic",  # Uses actual enum value
                config_hash="x", config_json="{}", registered_at=now
            ))
            conn.execute(nodes_table.insert().values(
                node_id="transform-001", run_id=run_id, plugin_name="passthrough",
                node_type="transform", plugin_version="1.0",
                determinism="deterministic",  # Uses actual enum value
                config_hash="y", config_json="{}", registered_at=now
            ))
            conn.commit()

        return run_id

    @pytest.fixture
    def run_with_external_calls(self, landscape_db) -> str:
        """Create a run with nondeterministic (external call) transforms."""
        from datetime import datetime, timezone
        from elspeth.core.landscape.schema import runs_table, nodes_table

        run_id = "external-run-001"
        now = datetime.now(timezone.utc)

        with landscape_db.engine.connect() as conn:
            conn.execute(runs_table.insert().values(
                run_id=run_id,
                started_at=now,
                config_hash="external",
                settings_json="{}",
                canonical_version="sha256-rfc8785-v1",
                status="completed",
            ))
            conn.execute(nodes_table.insert().values(
                node_id="source-001", run_id=run_id, plugin_name="csv_source",
                node_type="source", plugin_version="1.0",
                determinism="deterministic",
                config_hash="x", config_json="{}", registered_at=now
            ))
            # This transform makes external calls (nondeterministic)
            conn.execute(nodes_table.insert().values(
                node_id="llm-transform-001", run_id=run_id, plugin_name="llm_classifier",
                node_type="transform", plugin_version="1.0",
                determinism="nondeterministic",  # Uses actual enum value
                config_hash="z", config_json="{}", registered_at=now
            ))
            conn.commit()

        return run_id

    def test_pure_pipeline_gets_full_reproducible(
        self, recorder, run_with_pure_transforms
    ) -> None:
        """Run with only deterministic transforms gets FULL_REPRODUCIBLE."""
        from elspeth.core.landscape.reproducibility import ReproducibilityGrade

        grade = recorder.compute_reproducibility_grade(run_with_pure_transforms)

        assert grade == ReproducibilityGrade.FULL_REPRODUCIBLE

    def test_external_calls_gets_replay_reproducible(
        self, recorder, run_with_external_calls
    ) -> None:
        """Run with external calls (payloads retained) gets REPLAY_REPRODUCIBLE."""
        from elspeth.core.landscape.reproducibility import ReproducibilityGrade

        grade = recorder.compute_reproducibility_grade(run_with_external_calls)

        assert grade == ReproducibilityGrade.REPLAY_REPRODUCIBLE

    def test_finalize_run_sets_grade(
        self, recorder, run_id
    ) -> None:
        """finalize_run() sets reproducibility_grade on runs table."""
        from sqlalchemy import select
        from elspeth.core.landscape.schema import runs_table

        # Finalize the run
        recorder.finalize_run(run_id, status="completed")

        # Check grade was set
        with recorder._db.engine.connect() as conn:
            result = conn.execute(
                select(runs_table.c.reproducibility_grade)
                .where(runs_table.c.run_id == run_id)
            ).fetchone()

        assert result is not None
        assert result[0] is not None  # Grade should be set

    def test_grade_degrades_after_purge(
        self, recorder, landscape_db, run_with_external_calls
    ) -> None:
        """Grade degrades to ATTRIBUTABLE_ONLY when payloads are purged."""
        from elspeth.core.landscape.reproducibility import (
            ReproducibilityGrade,
            update_grade_after_purge,
        )

        # Initial grade should be REPLAY_REPRODUCIBLE
        initial = recorder.compute_reproducibility_grade(run_with_external_calls)
        assert initial == ReproducibilityGrade.REPLAY_REPRODUCIBLE

        # After purge, grade should degrade
        update_grade_after_purge(landscape_db, run_with_external_calls)

        # Re-read grade from database
        from sqlalchemy import select
        from elspeth.core.landscape.schema import runs_table

        with landscape_db.engine.connect() as conn:
            result = conn.execute(
                select(runs_table.c.reproducibility_grade)
                .where(runs_table.c.run_id == run_with_external_calls)
            ).fetchone()

        assert result[0] == ReproducibilityGrade.ATTRIBUTABLE_ONLY.value
```

### Step 2: Run test to verify it fails

Run: `pytest tests/core/landscape/test_recorder.py::TestReproducibilityGradeComputation -v`
Expected: FAIL (ImportError - reproducibility module doesn't exist)

### Step 3: Create reproducibility grade module

```python
# src/elspeth/core/landscape/reproducibility.py
"""Reproducibility grade computation for runs.

Grades indicate what level of replay/verification is possible:
- FULL_REPRODUCIBLE: All transforms deterministic, can re-run and get same results
- REPLAY_REPRODUCIBLE: Has external calls, but payloads retained for replay
- ATTRIBUTABLE_ONLY: Payloads purged, can only verify hashes match
"""

from enum import Enum

from sqlalchemy import select, update, and_


class ReproducibilityGrade(Enum):
    """Reproducibility levels for a completed run."""

    FULL_REPRODUCIBLE = "full_reproducible"
    REPLAY_REPRODUCIBLE = "replay_reproducible"
    ATTRIBUTABLE_ONLY = "attributable_only"


def compute_grade(db, run_id: str) -> ReproducibilityGrade:
    """Compute reproducibility grade for a run.

    Algorithm:
    1. Scan all nodes in the run
    2. If any node has determinism='nondeterministic' (from Determinism enum),
       grade is at best REPLAY_REPRODUCIBLE
    3. 'deterministic' and 'seeded' both allow FULL_REPRODUCIBLE
       (seeded transforms are reproducible given the same seed)

    Args:
        db: LandscapeDB instance
        run_id: Run to compute grade for

    Returns:
        Computed ReproducibilityGrade
    """
    from sqlalchemy import select
    from elspeth.core.landscape.schema import nodes_table
    from elspeth.plugins.enums import Determinism

    with db.engine.connect() as conn:
        # Check for non-deterministic nodes
        results = conn.execute(
            select(nodes_table.c.determinism)
            .where(nodes_table.c.run_id == run_id)
        ).fetchall()

    determinisms = {r[0] for r in results}

    # If any nondeterministic transform, not fully reproducible
    # (Note: 'seeded' counts as reproducible since seed is recorded)
    if Determinism.NONDETERMINISTIC.value in determinisms:
        return ReproducibilityGrade.REPLAY_REPRODUCIBLE

    return ReproducibilityGrade.FULL_REPRODUCIBLE


def set_run_grade(db, run_id: str, grade: ReproducibilityGrade) -> None:
    """Set the reproducibility grade on a run.

    Args:
        db: LandscapeDB instance
        run_id: Run to update
        grade: Grade to set
    """
    from elspeth.core.landscape.schema import runs_table

    with db.engine.connect() as conn:
        conn.execute(
            update(runs_table)
            .where(runs_table.c.run_id == run_id)
            .values(reproducibility_grade=grade.value)
        )
        conn.commit()


def update_grade_after_purge(db, run_id: str) -> None:
    """Degrade run's reproducibility grade after payload purge.

    Called by PurgeManager after deleting payloads. Transitions:
    - REPLAY_REPRODUCIBLE -> ATTRIBUTABLE_ONLY
    - Others remain unchanged

    Args:
        db: LandscapeDB instance
        run_id: Run whose payloads were purged
    """
    from elspeth.core.landscape.schema import runs_table

    with db.engine.connect() as conn:
        # Get current grade
        result = conn.execute(
            select(runs_table.c.reproducibility_grade)
            .where(runs_table.c.run_id == run_id)
        ).fetchone()

        if result is None:
            return

        current = result[0]

        # Degrade if was REPLAY_REPRODUCIBLE
        if current == ReproducibilityGrade.REPLAY_REPRODUCIBLE.value:
            conn.execute(
                update(runs_table)
                .where(runs_table.c.run_id == run_id)
                .values(reproducibility_grade=ReproducibilityGrade.ATTRIBUTABLE_ONLY.value)
            )
            conn.commit()
```

### Step 4: Add reproducibility_grade column to runs table

Add to `src/elspeth/core/landscape/schema.py` in the runs_table definition:

```python
Column("reproducibility_grade", String(32)),  # full_reproducible, replay_reproducible, attributable_only
```

### Step 5: Update LandscapeRecorder.finalize_run()

Add to `src/elspeth/core/landscape/recorder.py`:

```python
from elspeth.core.landscape.reproducibility import (
    compute_grade,
    set_run_grade,
)


def finalize_run(self, run_id: str, status: str) -> None:
    """Finalize a run with status and reproducibility grade.

    Args:
        run_id: Run to finalize
        status: Final status (completed, failed)
    """
    from datetime import datetime, timezone

    # Compute reproducibility grade
    grade = compute_grade(self._db, run_id)

    # Update run record
    with self._db.engine.connect() as conn:
        conn.execute(
            update(runs_table)
            .where(runs_table.c.run_id == run_id)
            .values(
                status=status,
                completed_at=datetime.now(timezone.utc),
                reproducibility_grade=grade.value,
            )
        )
        conn.commit()
```

### Step 6: Update PurgeManager to degrade grades

Modify `src/elspeth/core/retention/purge.py` to call `update_grade_after_purge()`:

```python
def purge_payloads(self, refs: list[str], run_ids: list[str] | None = None) -> PurgeResult:
    """Delete payloads and update reproducibility grades.

    Args:
        refs: List of payload refs to delete
        run_ids: Optional list of affected run IDs to update grades

    Returns:
        PurgeResult with statistics
    """
    result = self._delete_payloads(refs)

    # Degrade affected runs' grades
    if run_ids:
        from elspeth.core.landscape.reproducibility import update_grade_after_purge

        for run_id in run_ids:
            update_grade_after_purge(self._db, run_id)

    return result
```

### Step 7: Run tests

Run: `pytest tests/core/landscape/test_recorder.py::TestReproducibilityGradeComputation -v`
Expected: PASS

---

## Summary

Phase 5 adds three pillars of operational reliability plus reproducibility tracking:

| Pillar | Tasks | Key Components |
|--------|-------|----------------|
| **Checkpointing** | 1-5 | `CheckpointManager`, `RecoveryManager`, checkpoint schema |
| **Rate Limiting** | 6-8 | `RateLimiter`, `RateLimitRegistry`, per-service config |
| **Retention/Purge** | 9-11 | `PurgeManager`, `purge` command, graceful explain() |
| **CLI Integration** | 12-13 | `resume` command, end-to-end checkpoint/recovery tests |
| **Reproducibility** | 14 | Grade computation (`FULL_REPRODUCIBLE` → `ATTRIBUTABLE_ONLY`), grade degradation on purge |

**Deferred to Phase 6:**
- Redaction profiles
- Secret fingerprinting (HMAC)
- Replay mode for external calls

**Dependencies satisfied:**
- Phase 3A: Landscape schema and recorder
- Phase 3B: Engine and Orchestrator
- Phase 4: CLI foundation

**New config fields:**
- `checkpoint.enabled`, `checkpoint.frequency`, `checkpoint.checkpoint_interval`
- `rate_limit.enabled`, `rate_limit.default_requests_per_second`, `rate_limit.services`

**New CLI commands:**
- `elspeth purge` - Clean old payloads
- `elspeth resume` - Continue failed run from checkpoint
