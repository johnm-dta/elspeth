# Phase 7: Advanced Features (Tasks 1-14)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add advanced capabilities: A/B testing infrastructure for comparing transform variants, multi-destination routing with copy semantics, and Azure cloud integration.

**Architecture:** A/B testing uses deterministic variant assignment based on row_id for reproducibility. Multi-destination routing creates token copies that flow through parallel paths. Azure pack provides blob storage sources/sinks.

**Tech Stack:** Python 3.11+, azure-storage-blob, azure-identity, NetworkX (variant path validation)

**Dependencies:**
- Phase 3A: `elspeth.core.landscape` (schema, tokens, routing_events)
- Phase 3B: `elspeth.engine` (Orchestrator, RowProcessor)
- Phase 4: `elspeth.cli`
- Phase 6: `elspeth.core.calls` (for external call recording in Azure)

---

## Multi-Destination Routing: Copy vs Move

**Key concept:** When a gate routes to multiple destinations, it can use `move` or `copy` semantics:

| Mode | Behavior | Token Handling | Use Case |
|------|----------|----------------|----------|
| `move` | Token goes to ONE destination | Original token continues on selected edge | Mutually exclusive routing (A OR B) |
| `copy` | Token is duplicated to MULTIPLE destinations | New child tokens created, parent marked FORKED | Fan-out (A AND B simultaneously) |

**Existing schema support (from Phase 3A):**

```sql
-- edges.default_mode indicates move or copy
CREATE TABLE edges (
    ...
    default_mode TEXT NOT NULL,  -- move, copy
);

-- routing_events link decisions to edges
CREATE TABLE routing_events (
    ...
    routing_group_id TEXT NOT NULL,  -- Links events from same decision
    mode TEXT NOT NULL,              -- move, copy (may override default)
);

-- tokens track fork lineage
CREATE TABLE tokens (
    ...
    fork_group_id TEXT,  -- Links tokens from same fork
);

-- token_parents for multi-parent joins
CREATE TABLE token_parents (
    token_id TEXT NOT NULL,
    parent_token_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
);
```

---

## A/B Testing: Experimental Orchestration

**Key concept:** Run the same row through different transform variants to compare outcomes.

**Design principles:**
1. **Deterministic assignment** - Same row_id always gets same variant (for reproducibility)
2. **Parallel execution** - Variants run as parallel paths, not sequential
3. **Metrics collection** - Capture latency, token usage, quality scores per variant
4. **Statistical analysis** - Compare variants with significance testing

---

## Task 1: Experiment Configuration Schema

**Context:** Define configuration schema for A/B experiments.

**Files:**
- Modify: `src/elspeth/core/config.py` (add ExperimentConfig)
- Create: `tests/core/test_experiment_config.py`

### Step 1: Write the failing test

```python
# tests/core/test_experiment_config.py
"""Tests for experiment configuration."""

import pytest


class TestExperimentConfig:
    """Tests for A/B experiment configuration."""

    def test_experiment_config_basic(self) -> None:
        """Can create experiment config with variants."""
        from elspeth.core.config import ExperimentConfig, VariantConfig

        config = ExperimentConfig(
            experiment_id="exp-001",
            name="Compare GPT-4 vs Claude",
            variants=[
                VariantConfig(
                    variant_id="control",
                    weight=50,
                    transform_overrides={"model": "gpt-4"},
                ),
                VariantConfig(
                    variant_id="treatment",
                    weight=50,
                    transform_overrides={"model": "claude-3-opus"},
                ),
            ],
        )

        assert len(config.variants) == 2
        assert config.total_weight == 100

    def test_experiment_config_validates_weights(self) -> None:
        """Weights must sum to 100."""
        from pydantic import ValidationError
        from elspeth.core.config import ExperimentConfig, VariantConfig

        with pytest.raises(ValidationError):
            ExperimentConfig(
                experiment_id="exp-001",
                name="Invalid",
                variants=[
                    VariantConfig(variant_id="a", weight=30, transform_overrides={}),
                    VariantConfig(variant_id="b", weight=30, transform_overrides={}),
                ],  # Only sums to 60
            )

    def test_variant_assignment_deterministic(self) -> None:
        """Same row_id always gets same variant."""
        from elspeth.core.config import ExperimentConfig, VariantConfig

        config = ExperimentConfig(
            experiment_id="exp-001",
            name="Test",
            variants=[
                VariantConfig(variant_id="a", weight=50, transform_overrides={}),
                VariantConfig(variant_id="b", weight=50, transform_overrides={}),
            ],
        )

        # Same row_id should always get same variant
        v1 = config.assign_variant("row-123")
        v2 = config.assign_variant("row-123")
        v3 = config.assign_variant("row-123")

        assert v1 == v2 == v3

    def test_variant_distribution_roughly_correct(self) -> None:
        """Variant distribution should roughly match weights."""
        from elspeth.core.config import ExperimentConfig, VariantConfig

        config = ExperimentConfig(
            experiment_id="exp-001",
            name="Test",
            variants=[
                VariantConfig(variant_id="a", weight=70, transform_overrides={}),
                VariantConfig(variant_id="b", weight=30, transform_overrides={}),
            ],
        )

        # Assign many rows
        assignments = [config.assign_variant(f"row-{i}") for i in range(1000)]

        a_count = sum(1 for v in assignments if v.variant_id == "a")
        b_count = sum(1 for v in assignments if v.variant_id == "b")

        # Should be roughly 70/30 (allow 10% tolerance)
        assert 600 < a_count < 800
        assert 200 < b_count < 400
```

### Step 2: Create ExperimentConfig

```python
# Add to src/elspeth/core/config.py

import hashlib
from dataclasses import dataclass


@dataclass
class VariantConfig:
    """Configuration for an experiment variant."""
    variant_id: str
    weight: int  # 0-100, percentage of traffic
    transform_overrides: dict[str, Any]  # Config overrides for this variant
    description: str = ""


class ExperimentConfig(BaseSettings):
    """Configuration for A/B experiment.

    Example YAML:
        experiment:
          experiment_id: exp-001
          name: "Compare models"
          variants:
            - variant_id: control
              weight: 50
              transform_overrides:
                model: gpt-4
            - variant_id: treatment
              weight: 50
              transform_overrides:
                model: claude-3-opus
    """

    experiment_id: str
    name: str
    description: str = ""
    variants: list[VariantConfig]
    enabled: bool = True

    @property
    def total_weight(self) -> int:
        return sum(v.weight for v in self.variants)

    @model_validator(mode="after")
    def validate_weights(self) -> "ExperimentConfig":
        if self.total_weight != 100:
            raise ValueError(
                f"Variant weights must sum to 100, got {self.total_weight}"
            )
        return self

    def assign_variant(self, row_id: str) -> VariantConfig:
        """Deterministically assign a row to a variant.

        Uses hash of (experiment_id, row_id) for reproducibility.
        Same row_id always gets same variant for this experiment.

        Args:
            row_id: Row identifier

        Returns:
            Assigned VariantConfig
        """
        # Create deterministic hash
        hash_input = f"{self.experiment_id}:{row_id}".encode()
        hash_value = int(hashlib.sha256(hash_input).hexdigest(), 16)

        # Map to 0-99
        bucket = hash_value % 100

        # Find variant for this bucket
        cumulative = 0
        for variant in self.variants:
            cumulative += variant.weight
            if bucket < cumulative:
                return variant

        # Fallback (shouldn't happen if weights sum to 100)
        return self.variants[-1]
```

### Step 3: Run tests

Run: `pytest tests/core/test_experiment_config.py -v`
Expected: PASS

---

## Task 2: Experiment Recording in Landscape

**Context:** Record experiment assignments in Landscape for analysis.

**Files:**
- Modify: `src/elspeth/core/landscape/schema.py` (add experiment_assignments table)
- Modify: `src/elspeth/core/landscape/recorder.py` (add record_experiment_assignment)
- Create: `tests/core/landscape/test_experiments.py`

### Step 1: Write the failing test

```python
# tests/core/landscape/test_experiments.py
"""Tests for experiment recording."""

import pytest


class TestExperimentRecording:
    """Tests for recording experiment assignments."""

    def test_record_experiment_assignment(self, recorder, run_id, row_id) -> None:
        """Can record which variant a row was assigned to."""
        assignment = recorder.record_experiment_assignment(
            run_id=run_id,
            row_id=row_id,
            experiment_id="exp-001",
            variant_id="treatment",
            transform_overrides={"model": "claude-3-opus"},
        )

        assert assignment.assignment_id is not None
        assert assignment.variant_id == "treatment"

    def test_get_assignments_for_experiment(self, recorder, run_id) -> None:
        """Can query assignments for an experiment."""
        # Record multiple assignments
        for i in range(10):
            recorder.record_experiment_assignment(
                run_id=run_id,
                row_id=f"row-{i}",
                experiment_id="exp-001",
                variant_id="control" if i % 2 == 0 else "treatment",
                transform_overrides={},
            )

        assignments = recorder.get_experiment_assignments(
            run_id=run_id,
            experiment_id="exp-001",
        )

        assert len(assignments) == 10

    def test_get_variant_metrics(self, recorder, run_with_experiment) -> None:
        """Can aggregate metrics by variant."""
        metrics = recorder.get_variant_metrics(
            run_id=run_with_experiment,
            experiment_id="exp-001",
        )

        assert "control" in metrics
        assert "treatment" in metrics
        assert "count" in metrics["control"]
```

### Step 2: Add experiment_assignments table

Add to `src/elspeth/core/landscape/schema.py`:

```python
# === Experiment Assignments (Phase 7: A/B Testing) ===

experiment_assignments_table = Table(
    "experiment_assignments",
    metadata,
    Column("assignment_id", String(64), primary_key=True),
    Column("run_id", String(64), ForeignKey("runs.run_id"), nullable=False),
    Column("row_id", String(64), ForeignKey("rows.row_id"), nullable=False),
    Column("experiment_id", String(64), nullable=False),
    Column("variant_id", String(64), nullable=False),
    Column("transform_overrides_json", Text),  # JSON of applied overrides
    Column("created_at", DateTime(timezone=True), nullable=False),
    # Unique constraint: one assignment per (row, experiment)
    UniqueConstraint("run_id", "row_id", "experiment_id"),
)

Index("ix_exp_assignments_experiment", experiment_assignments_table.c.experiment_id)
Index("ix_exp_assignments_variant", experiment_assignments_table.c.variant_id)
```

### Step 3: Add recording methods to LandscapeRecorder

```python
# Add to src/elspeth/core/landscape/recorder.py

def record_experiment_assignment(
    self,
    run_id: str,
    row_id: str,
    experiment_id: str,
    variant_id: str,
    transform_overrides: dict[str, Any],
) -> ExperimentAssignment:
    """Record that a row was assigned to an experiment variant.

    Args:
        run_id: Current run
        row_id: Row being processed
        experiment_id: Experiment identifier
        variant_id: Assigned variant
        transform_overrides: Config overrides applied

    Returns:
        Recorded assignment
    """
    import uuid
    import json
    from datetime import datetime, timezone
    from elspeth.core.landscape.schema import experiment_assignments_table

    assignment_id = f"assign-{uuid.uuid4().hex[:12]}"
    now = datetime.now(timezone.utc)

    with self._db.engine.connect() as conn:
        conn.execute(
            experiment_assignments_table.insert().values(
                assignment_id=assignment_id,
                run_id=run_id,
                row_id=row_id,
                experiment_id=experiment_id,
                variant_id=variant_id,
                transform_overrides_json=json.dumps(transform_overrides),
                created_at=now,
            )
        )
        conn.commit()

    return ExperimentAssignment(
        assignment_id=assignment_id,
        run_id=run_id,
        row_id=row_id,
        experiment_id=experiment_id,
        variant_id=variant_id,
        transform_overrides=transform_overrides,
        created_at=now,
    )

def get_experiment_assignments(
    self,
    run_id: str,
    experiment_id: str,
) -> list[ExperimentAssignment]:
    """Get all assignments for an experiment in a run."""
    from sqlalchemy import select, and_
    from elspeth.core.landscape.schema import experiment_assignments_table

    with self._db.engine.connect() as conn:
        results = conn.execute(
            select(experiment_assignments_table)
            .where(and_(
                experiment_assignments_table.c.run_id == run_id,
                experiment_assignments_table.c.experiment_id == experiment_id,
            ))
        ).fetchall()

    return [self._row_to_assignment(r) for r in results]

def get_variant_metrics(
    self,
    run_id: str,
    experiment_id: str,
) -> dict[str, dict[str, Any]]:
    """Get aggregated metrics by variant.

    Returns metrics like count, avg_latency, success_rate per variant.
    """
    from sqlalchemy import select, func, and_
    from elspeth.core.landscape.schema import (
        experiment_assignments_table as ea,
        node_states_table as ns,
        tokens_table as t,
    )

    metrics = {}

    with self._db.engine.connect() as conn:
        # Count per variant
        counts = conn.execute(
            select(
                ea.c.variant_id,
                func.count(ea.c.assignment_id).label("count"),
            )
            .where(and_(
                ea.c.run_id == run_id,
                ea.c.experiment_id == experiment_id,
            ))
            .group_by(ea.c.variant_id)
        ).fetchall()

        for row in counts:
            metrics[row.variant_id] = {"count": row.count}

    return metrics
```

### Step 4: Run tests

Run: `pytest tests/core/landscape/test_experiments.py -v`
Expected: PASS

---

## Task 3: Copy Mode Token Forking

**Context:** Implement token forking for copy mode routing (one row → multiple parallel paths).

**Files:**
- Modify: `src/elspeth/engine/token_manager.py` (add fork_token)
- Create: `tests/engine/test_token_forking.py`

### Step 1: Write the failing test

```python
# tests/engine/test_token_forking.py
"""Tests for token forking."""

import pytest


class TestTokenForking:
    """Tests for copy mode token forking."""

    def test_fork_creates_child_tokens(self, token_manager, parent_token) -> None:
        """Fork creates child tokens for each destination."""
        children = token_manager.fork_token(
            parent_token_id=parent_token.token_id,
            destinations=["path_a", "path_b", "path_c"],
        )

        assert len(children) == 3
        assert all(c.parent_token_id == parent_token.token_id for c in children)

    def test_fork_sets_branch_names(self, token_manager, parent_token) -> None:
        """Forked tokens have branch names."""
        children = token_manager.fork_token(
            parent_token_id=parent_token.token_id,
            destinations=["sentiment", "classification"],
        )

        branch_names = {c.branch_name for c in children}
        assert branch_names == {"sentiment", "classification"}

    def test_fork_links_to_fork_group(self, token_manager, parent_token) -> None:
        """Forked tokens share a fork_group_id."""
        children = token_manager.fork_token(
            parent_token_id=parent_token.token_id,
            destinations=["a", "b"],
        )

        # All children have same fork_group_id
        fork_groups = {c.fork_group_id for c in children}
        assert len(fork_groups) == 1
        assert fork_groups.pop() is not None

    def test_fork_records_parent_relationship(
        self, token_manager, parent_token, landscape_db
    ) -> None:
        """Fork creates token_parents entries."""
        children = token_manager.fork_token(
            parent_token_id=parent_token.token_id,
            destinations=["a", "b"],
        )

        from sqlalchemy import select
        from elspeth.core.landscape.schema import token_parents_table

        with landscape_db.engine.connect() as conn:
            for child in children:
                result = conn.execute(
                    select(token_parents_table)
                    .where(token_parents_table.c.token_id == child.token_id)
                ).fetchone()

                assert result is not None
                assert result.parent_token_id == parent_token.token_id

    def test_parent_token_marked_forked(
        self, token_manager, parent_token
    ) -> None:
        """Parent token can be queried as FORKED terminal state."""
        token_manager.fork_token(
            parent_token_id=parent_token.token_id,
            destinations=["a", "b"],
        )

        state = token_manager.get_terminal_state(parent_token.token_id)
        assert state == "FORKED"
```

### Step 2: Create TokenManager.fork_token

```python
# src/elspeth/engine/token_manager.py
"""Token lifecycle management."""

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass
class Token:
    """A token flowing through the DAG."""
    token_id: str
    row_id: str
    parent_token_id: str | None
    fork_group_id: str | None
    join_group_id: str | None
    branch_name: str | None
    created_at: datetime


class TokenManager:
    """Manages token lifecycle including forks and joins."""

    def __init__(self, db) -> None:
        self._db = db

    def create_token(self, row_id: str) -> Token:
        """Create initial token for a row."""
        from elspeth.core.landscape.schema import tokens_table

        token_id = f"tok-{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc)

        with self._db.engine.connect() as conn:
            conn.execute(
                tokens_table.insert().values(
                    token_id=token_id,
                    row_id=row_id,
                    created_at=now,
                )
            )
            conn.commit()

        return Token(
            token_id=token_id,
            row_id=row_id,
            parent_token_id=None,
            fork_group_id=None,
            join_group_id=None,
            branch_name=None,
            created_at=now,
        )

    def fork_token(
        self,
        parent_token_id: str,
        destinations: list[str],
    ) -> list[Token]:
        """Fork a token into multiple child tokens.

        Used for copy mode routing where one token becomes many.

        Args:
            parent_token_id: Token to fork
            destinations: Branch names for child tokens

        Returns:
            List of child tokens
        """
        from elspeth.core.landscape.schema import tokens_table, token_parents_table

        # Get parent token's row_id
        parent = self.get_token(parent_token_id)
        if parent is None:
            raise ValueError(f"Parent token not found: {parent_token_id}")

        fork_group_id = f"fork-{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc)
        children = []

        with self._db.engine.connect() as conn:
            for ordinal, branch_name in enumerate(destinations):
                child_id = f"tok-{uuid.uuid4().hex[:12]}"

                # Create child token
                conn.execute(
                    tokens_table.insert().values(
                        token_id=child_id,
                        row_id=parent.row_id,
                        fork_group_id=fork_group_id,
                        branch_name=branch_name,
                        created_at=now,
                    )
                )

                # Link to parent
                conn.execute(
                    token_parents_table.insert().values(
                        token_id=child_id,
                        parent_token_id=parent_token_id,
                        ordinal=ordinal,
                    )
                )

                children.append(Token(
                    token_id=child_id,
                    row_id=parent.row_id,
                    parent_token_id=parent_token_id,
                    fork_group_id=fork_group_id,
                    join_group_id=None,
                    branch_name=branch_name,
                    created_at=now,
                ))

            conn.commit()

        return children

    def get_token(self, token_id: str) -> Token | None:
        """Get token by ID."""
        from sqlalchemy import select
        from elspeth.core.landscape.schema import tokens_table

        with self._db.engine.connect() as conn:
            result = conn.execute(
                select(tokens_table)
                .where(tokens_table.c.token_id == token_id)
            ).fetchone()

        if result is None:
            return None

        return Token(
            token_id=result.token_id,
            row_id=result.row_id,
            parent_token_id=None,  # Loaded from token_parents if needed
            fork_group_id=result.fork_group_id,
            join_group_id=result.join_group_id,
            branch_name=result.branch_name,
            created_at=result.created_at,
        )

    def get_terminal_state(self, token_id: str) -> str:
        """Derive terminal state for a token.

        States are derived, not stored:
        - COMPLETED: Reached sink
        - FORKED: Has child tokens
        - CONSUMED_IN_BATCH: In batch_members
        - etc.
        """
        from sqlalchemy import select, exists
        from elspeth.core.landscape.schema import token_parents_table

        with self._db.engine.connect() as conn:
            # Check if token has children (FORKED)
            has_children = conn.execute(
                select(exists().where(
                    token_parents_table.c.parent_token_id == token_id
                ))
            ).scalar()

            if has_children:
                return "FORKED"

        # TODO: Check other terminal states
        return "UNKNOWN"
```

### Step 3: Run tests

Run: `pytest tests/engine/test_token_forking.py -v`
Expected: PASS

---

## Task 4: Multi-Destination Routing in Gate Executor

**Context:** Update gate executor to handle copy mode routing with token forking.

**Files:**
- Modify: `src/elspeth/engine/executors/gate.py` (add copy mode handling)
- Modify: `tests/engine/executors/test_gate.py` (add copy mode tests)

### Step 1: Write the failing test

```python
# Add to tests/engine/executors/test_gate.py

class TestGateCopyMode:
    """Tests for copy mode routing."""

    def test_copy_mode_forks_token(
        self, gate_executor, token, context
    ) -> None:
        """Copy mode creates forked tokens."""
        # Gate that routes to multiple destinations with copy
        result = gate_executor.execute(
            token=token,
            context=context,
            routing_action=RoutingAction(
                kind="fork_to_paths",
                destinations=["path_a", "path_b"],
                mode="copy",
                reason={"rule": "fan_out"},
            ),
        )

        assert result.action == "forked"
        assert len(result.child_tokens) == 2

    def test_copy_mode_records_routing_events(
        self, gate_executor, token, context, landscape_db
    ) -> None:
        """Copy mode records routing_event per destination."""
        result = gate_executor.execute(
            token=token,
            context=context,
            routing_action=RoutingAction(
                kind="fork_to_paths",
                destinations=["a", "b", "c"],
                mode="copy",
                reason={},
            ),
        )

        from sqlalchemy import select
        from elspeth.core.landscape.schema import routing_events_table

        with landscape_db.engine.connect() as conn:
            events = conn.execute(
                select(routing_events_table)
                .where(routing_events_table.c.routing_group_id == result.routing_group_id)
            ).fetchall()

        assert len(events) == 3
        # All share same routing_group_id
        assert all(e.routing_group_id == result.routing_group_id for e in events)

    def test_move_mode_selects_single_destination(
        self, gate_executor, token, context
    ) -> None:
        """Move mode routes to single destination without forking."""
        result = gate_executor.execute(
            token=token,
            context=context,
            routing_action=RoutingAction(
                kind="route_to_sink",
                destinations=["quarantine"],
                mode="move",
                reason={"suspicious": True},
            ),
        )

        assert result.action == "routed"
        assert result.destination == "quarantine"
        assert result.child_tokens is None
```

### Step 2: Implementation guidance

Update `GateExecutor.execute()` to:

1. Check routing action mode (move vs copy)
2. For `copy` mode with multiple destinations: call `token_manager.fork_token()`
3. Record one `routing_event` per destination with shared `routing_group_id`
4. Return `GateResult` with child tokens for copy, or single destination for move

---

## Task 5: Azure Blob Storage Source Plugin

**Context:** Create source plugin that reads from Azure Blob Storage.

**Files:**
- Create: `src/elspeth/plugins/packs/azure/__init__.py`
- Create: `src/elspeth/plugins/packs/azure/blob_source.py`
- Create: `tests/plugins/packs/azure/__init__.py`
- Create: `tests/plugins/packs/azure/test_blob_source.py`

### Step 1: Write the failing test

```python
# tests/plugins/packs/azure/test_blob_source.py
"""Tests for Azure Blob Storage source."""

import pytest
from unittest.mock import Mock, patch


class TestAzureBlobSource:
    """Tests for Azure Blob source plugin."""

    def test_create_source(self) -> None:
        """Can create Azure Blob source."""
        from elspeth.plugins.packs.azure import AzureBlobSource

        source = AzureBlobSource({
            "connection_string": "DefaultEndpointsProtocol=https;...",
            "container": "data",
            "blob_pattern": "*.json",
        })

        assert source.name == "azure_blob"

    @patch("azure.storage.blob.BlobServiceClient")
    def test_load_yields_rows(self, mock_client_class, context) -> None:
        """load() yields rows from blobs."""
        from elspeth.plugins.packs.azure import AzureBlobSource

        # Mock blob listing
        mock_client = Mock()
        mock_container = Mock()
        mock_client.get_container_client.return_value = mock_container
        mock_container.list_blobs.return_value = [
            Mock(name="data1.json"),
            Mock(name="data2.json"),
        ]

        # Mock blob content
        mock_blob = Mock()
        mock_blob.download_blob.return_value.readall.return_value = b'{"id": 1}'
        mock_container.get_blob_client.return_value = mock_blob

        mock_client_class.from_connection_string.return_value = mock_client

        source = AzureBlobSource({
            "connection_string": "test",
            "container": "data",
            "blob_pattern": "*.json",
        })

        rows = list(source.load(context))

        assert len(rows) == 2
        assert rows[0]["id"] == 1

    def test_source_has_output_schema(self) -> None:
        """Source declares output schema."""
        from elspeth.plugins.packs.azure import AzureBlobSource

        source = AzureBlobSource({
            "connection_string": "test",
            "container": "data",
        })

        assert source.output_schema is not None
```

### Step 2: Create AzureBlobSource

```python
# src/elspeth/plugins/packs/azure/__init__.py
"""Azure plugin pack.

Provides:
- AzureBlobSource: Read from Azure Blob Storage
- AzureBlobSink: Write to Azure Blob Storage
"""

from elspeth.plugins.packs.azure.blob_source import AzureBlobSource
from elspeth.plugins.packs.azure.blob_sink import AzureBlobSink

__all__ = ["AzureBlobSource", "AzureBlobSink"]


# src/elspeth/plugins/packs/azure/blob_source.py
"""Azure Blob Storage source plugin."""

import fnmatch
import json
from collections.abc import Iterator
from typing import Any

from azure.storage.blob import BlobServiceClient

from elspeth.plugins.schemas import PluginSchema


class AzureBlobSchema(PluginSchema):
    """Schema for Azure Blob source output."""
    _blob_name: str
    _blob_content: dict


class AzureBlobSource:
    """Source that reads from Azure Blob Storage.

    Configuration:
        connection_string: Azure Storage connection string
        container: Container name
        blob_pattern: Optional glob pattern (default: *)
        parse_json: Whether to parse JSON blobs (default: true)

    Example:
        source:
          plugin: azure_blob
          options:
            connection_string: ${AZURE_STORAGE_CONNECTION}
            container: input-data
            blob_pattern: "*.json"
    """

    name = "azure_blob"
    output_schema = AzureBlobSchema

    def __init__(self, config: dict[str, Any]) -> None:
        self._connection_string = config["connection_string"]
        self._container = config["container"]
        self._blob_pattern = config.get("blob_pattern", "*")
        self._parse_json = config.get("parse_json", True)
        self._client: BlobServiceClient | None = None

    def on_start(self, ctx) -> None:
        """Initialize Azure client."""
        self._client = BlobServiceClient.from_connection_string(
            self._connection_string
        )

    def load(self, ctx) -> Iterator[dict[str, Any]]:
        """Load and yield rows from blobs."""
        if self._client is None:
            self.on_start(ctx)

        container = self._client.get_container_client(self._container)

        for blob in container.list_blobs():
            if not fnmatch.fnmatch(blob.name, self._blob_pattern):
                continue

            blob_client = container.get_blob_client(blob.name)
            content = blob_client.download_blob().readall()

            if self._parse_json:
                data = json.loads(content)
                if isinstance(data, list):
                    # Blob contains array of rows
                    for row in data:
                        row["_blob_name"] = blob.name
                        yield row
                else:
                    # Single object
                    data["_blob_name"] = blob.name
                    yield data
            else:
                # Yield raw content
                yield {
                    "_blob_name": blob.name,
                    "_blob_content": content.decode("utf-8"),
                }

    def close(self) -> None:
        """Clean up resources."""
        self._client = None
```

### Step 3: Run tests

Run: `pytest tests/plugins/packs/azure/test_blob_source.py -v`
Expected: PASS

---

## Task 6: Azure Blob Storage Sink Plugin

**Context:** Create sink plugin that writes to Azure Blob Storage.

**Files:**
- Create: `src/elspeth/plugins/packs/azure/blob_sink.py`
- Create: `tests/plugins/packs/azure/test_blob_sink.py`

### Step 1: Write the failing test

```python
# tests/plugins/packs/azure/test_blob_sink.py
"""Tests for Azure Blob Storage sink."""

import pytest
from unittest.mock import Mock, patch


class TestAzureBlobSink:
    """Tests for Azure Blob sink plugin."""

    def test_create_sink(self) -> None:
        """Can create Azure Blob sink."""
        from elspeth.plugins.packs.azure import AzureBlobSink

        sink = AzureBlobSink({
            "connection_string": "test",
            "container": "output",
            "blob_name_template": "{run_id}/{row_id}.json",
        })

        assert sink.name == "azure_blob"
        assert sink.idempotent is True  # Overwrites are idempotent

    @patch("azure.storage.blob.BlobServiceClient")
    def test_write_uploads_blob(self, mock_client_class, context) -> None:
        """write() uploads to blob storage."""
        from elspeth.plugins.packs.azure import AzureBlobSink

        mock_client = Mock()
        mock_container = Mock()
        mock_blob = Mock()
        mock_client.get_container_client.return_value = mock_container
        mock_container.get_blob_client.return_value = mock_blob
        mock_client_class.from_connection_string.return_value = mock_client

        sink = AzureBlobSink({
            "connection_string": "test",
            "container": "output",
            "blob_name_template": "{row_id}.json",
        })
        sink.on_start(context)

        sink.write({"id": 123, "data": "test"}, context)

        mock_blob.upload_blob.assert_called_once()

    def test_sink_generates_blob_names(self) -> None:
        """Sink generates blob names from template."""
        from elspeth.plugins.packs.azure import AzureBlobSink

        sink = AzureBlobSink({
            "connection_string": "test",
            "container": "output",
            "blob_name_template": "output/{run_id}/{row_id}.json",
        })

        name = sink._generate_blob_name(
            run_id="run-123",
            row_id="row-456",
            row={"category": "A"},
        )

        assert name == "output/run-123/row-456.json"
```

### Step 2: Create AzureBlobSink

```python
# src/elspeth/plugins/packs/azure/blob_sink.py
"""Azure Blob Storage sink plugin."""

import json
from typing import Any

from azure.storage.blob import BlobServiceClient

from elspeth.plugins.schemas import PluginSchema
from elspeth.plugins.enums import Determinism


class AzureBlobSink:
    """Sink that writes to Azure Blob Storage.

    Configuration:
        connection_string: Azure Storage connection string
        container: Container name
        blob_name_template: Template for blob names
        overwrite: Whether to overwrite existing blobs (default: true)

    Template variables:
        {run_id}: Current run ID
        {row_id}: Row ID
        {token_id}: Token ID
        {timestamp}: ISO timestamp
        {field:name}: Value of row field "name"

    Example:
        sink:
          plugin: azure_blob
          options:
            connection_string: ${AZURE_STORAGE_CONNECTION}
            container: results
            blob_name_template: "{run_id}/{row_id}.json"
    """

    name = "azure_blob"
    input_schema = PluginSchema  # Accepts any schema
    idempotent = True  # Overwrites make it idempotent
    determinism = Determinism.IO_DEPENDENT
    plugin_version = "1.0.0"

    def __init__(self, config: dict[str, Any]) -> None:
        self._connection_string = config["connection_string"]
        self._container = config["container"]
        self._blob_name_template = config.get("blob_name_template", "{row_id}.json")
        self._overwrite = config.get("overwrite", True)
        self._client: BlobServiceClient | None = None
        self._container_client = None

    def on_start(self, ctx) -> None:
        """Initialize Azure client."""
        self._client = BlobServiceClient.from_connection_string(
            self._connection_string
        )
        self._container_client = self._client.get_container_client(self._container)

    def write(self, row: dict[str, Any], ctx) -> None:
        """Write row to blob storage."""
        blob_name = self._generate_blob_name(
            run_id=ctx.run_id,
            row_id=ctx.row_id,
            row=row,
        )

        blob_client = self._container_client.get_blob_client(blob_name)

        content = json.dumps(row, indent=2)

        blob_client.upload_blob(
            content,
            overwrite=self._overwrite,
        )

    def _generate_blob_name(
        self,
        run_id: str,
        row_id: str,
        row: dict[str, Any],
    ) -> str:
        """Generate blob name from template."""
        from datetime import datetime, timezone

        name = self._blob_name_template
        name = name.replace("{run_id}", run_id)
        name = name.replace("{row_id}", row_id)
        name = name.replace("{timestamp}", datetime.now(timezone.utc).isoformat())

        # Handle field references
        import re
        for match in re.finditer(r"\{field:(\w+)\}", name):
            field = match.group(1)
            value = str(row.get(field, "unknown"))
            name = name.replace(match.group(0), value)

        return name

    def flush(self) -> None:
        """No buffering, nothing to flush."""
        pass

    def close(self) -> None:
        """Clean up resources."""
        self._client = None
        self._container_client = None
```

### Step 3: Run tests

Run: `pytest tests/plugins/packs/azure/test_blob_sink.py -v`
Expected: PASS

---

## Tasks 7-14: Remaining Implementation

The remaining tasks cover:

- **Task 7:** Azure Identity integration (DefaultAzureCredential, managed identity)
- **Task 8:** Experiment results analysis (statistical significance, confidence intervals)
- **Task 9:** CLI `experiment` command (list, status, compare)
- **Task 10:** Experiment visualization (TUI with Textual)
- **Task 11:** Copy mode integration in Orchestrator
- **Task 12:** Multi-destination explain() queries
- **Task 13:** Integration test - full A/B experiment cycle
- **Task 14:** Integration test - copy mode fan-out/fan-in

These follow the same TDD pattern established above.

---

## Summary

Phase 7 adds three advanced capabilities:

| Pillar | Tasks | Key Components |
|--------|-------|----------------|
| **A/B Testing** | 1-2, 8-10 | `ExperimentConfig`, `experiment_assignments` table, variant metrics, analysis |
| **Multi-Destination Routing** | 3-4, 11-12 | Token forking, copy mode, `routing_group_id`, parallel paths |
| **Azure Pack** | 5-7 | `AzureBlobSource`, `AzureBlobSink`, DefaultAzureCredential |
| **Integration** | 13-14 | End-to-end experiment and copy mode tests |

**Key design decisions:**

1. **Deterministic variant assignment:** Hash of `(experiment_id, row_id)` ensures same row always gets same variant - critical for reproducibility

2. **Copy vs Move semantics:**
   - Move: Token continues to ONE destination (mutually exclusive)
   - Copy: Token forks to MULTIPLE destinations (parallel execution)

3. **Routing groups:** `routing_group_id` links all `routing_event` records from a single gate decision - enables "what else happened" queries

**New Landscape tables:**
- `experiment_assignments` - Records variant assignments for analysis

**New CLI commands:**
- `elspeth experiment list` - Show experiments
- `elspeth experiment status <id>` - Show variant metrics
- `elspeth experiment compare <id>` - Statistical comparison

**Dependencies:**
- Azure SDK: `azure-storage-blob`, `azure-identity`
- Statistics: `scipy.stats` for significance testing
