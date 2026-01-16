# Contracts Subsystem Implementation Plan - Part 2

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Complete the contracts subsystem migration with config.py, audit.py, data.py, and AST enforcement.

**Architecture:** This continues from Part 1 (Tasks 1-4). Part 2 covers the remaining migrations: configuration contracts, audit trail models with repository layer, plugin schema base, and the enforcement script that prevents future drift.

**Tech Stack:** Python 3.11+, dataclasses, Pydantic, ast module, PyYAML

**Dependencies:**
- Part 1 must be complete (contracts/enums.py, identity.py, routing.py, results.py exist)

**Key Architectural Principle (from Part 1):**
- Dataclasses are strict contracts - no `__post_init__` coercion
- Type enforcement is at mypy time, not runtime
- Repository layer handles DB string→enum conversion
- Per Data Manifesto: audit DB is OUR data (crash on anomaly), source/transform output is THEIR data (validate, record, continue)

---

## Task 5: Create contracts/config.py

**Context:** Configuration types are Pydantic models (not dataclasses) because they need validation at the trust boundary (user-provided YAML). These stay as Pydantic but get re-exported from contracts for consistency.

**Files:**
- Create: `src/elspeth/contracts/config.py`
- Modify: `src/elspeth/contracts/__init__.py`
- Create: `tests/contracts/test_config.py`

### Step 1: Create config.py with re-exports

Create `src/elspeth/contracts/config.py`:

```python
"""Configuration contracts.

Configuration types use Pydantic (not dataclasses) because they validate
user-provided YAML - a legitimate trust boundary per Data Manifesto.

These are re-exports from core/config.py for import consistency.
The actual definitions stay in core/config.py where Pydantic validation logic lives.
"""

from elspeth.core.config import (
    CheckpointSettings,
    ConcurrencySettings,
    DatabaseSettings,
    DatasourceSettings,
    ElspethSettings,
    LandscapeExportSettings,
    LandscapeSettings,
    PayloadStoreSettings,
    RateLimitSettings,
    RetrySettings,
    RowPluginSettings,
    SinkSettings,
)

__all__ = [
    "CheckpointSettings",
    "ConcurrencySettings",
    "DatabaseSettings",
    "DatasourceSettings",
    "ElspethSettings",
    "LandscapeExportSettings",
    "LandscapeSettings",
    "PayloadStoreSettings",
    "RateLimitSettings",
    "RetrySettings",
    "RowPluginSettings",
    "SinkSettings",
]
```

### Step 2: Update contracts __init__.py

Add to `src/elspeth/contracts/__init__.py`:

```python
from elspeth.contracts.config import (
    CheckpointSettings,
    ConcurrencySettings,
    DatabaseSettings,
    DatasourceSettings,
    ElspethSettings,
    LandscapeExportSettings,
    LandscapeSettings,
    PayloadStoreSettings,
    RateLimitSettings,
    RetrySettings,
    RowPluginSettings,
    SinkSettings,
)

# Add to __all__
__all__ = [
    # ... existing exports ...
    # config
    "CheckpointSettings",
    "ConcurrencySettings",
    "DatabaseSettings",
    "DatasourceSettings",
    "ElspethSettings",
    "LandscapeExportSettings",
    "LandscapeSettings",
    "PayloadStoreSettings",
    "RateLimitSettings",
    "RetrySettings",
    "RowPluginSettings",
    "SinkSettings",
]
```

### Step 3: Write tests

Create `tests/contracts/test_config.py`:

```python
"""Tests for configuration contracts."""


class TestConfigReexports:
    """Verify config types are accessible from contracts."""

    def test_can_import_settings_from_contracts(self) -> None:
        """All settings types importable from contracts."""
        from elspeth.contracts import (
            ElspethSettings,
            DatasourceSettings,
            SinkSettings,
            RowPluginSettings,
            LandscapeSettings,
        )

        # Just verify import works
        assert ElspethSettings is not None
        assert DatasourceSettings is not None

    def test_settings_are_pydantic_models(self) -> None:
        """Config types are Pydantic (trust boundary validation)."""
        from pydantic import BaseModel
        from elspeth.contracts import ElspethSettings, DatasourceSettings

        assert issubclass(ElspethSettings, BaseModel)
        assert issubclass(DatasourceSettings, BaseModel)

    def test_settings_are_frozen(self) -> None:
        """Config is immutable after construction."""
        from elspeth.contracts import DatasourceSettings

        settings = DatasourceSettings(plugin="csv_local")

        import pytest
        with pytest.raises(Exception):  # Pydantic raises ValidationError
            settings.plugin = "other"
```

### Step 4: Run tests

Run: `pytest tests/contracts/test_config.py -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/elspeth/contracts/config.py tests/contracts/test_config.py
git commit -m "feat(contracts): add config.py with Pydantic re-exports

Configuration types use Pydantic for trust boundary validation.
Re-exported from contracts for consistent import pattern."
```

---

## Task 6a: Create contracts/audit.py - Core Models

**Context:** The largest migration. Core audit models: Run, Node, Edge, Row, Token, TokenParent. These need repository layer for enum conversion.

**Files:**
- Create: `src/elspeth/contracts/audit.py`
- Create: `src/elspeth/core/landscape/repositories.py`
- Modify: `src/elspeth/contracts/__init__.py`
- Create: `tests/contracts/test_audit.py`
- Create: `tests/core/landscape/test_repositories.py`

### Step 1: Create audit.py with core models

Create `src/elspeth/contracts/audit.py`:

```python
"""Audit trail contracts for Landscape tables.

These are strict contracts - all enum fields use proper enum types.
Repository layer handles string→enum conversion for DB reads.

Per Data Manifesto: The audit database is OUR data. If we read
garbage from it, something catastrophic happened - crash immediately.
"""

from dataclasses import dataclass
from datetime import datetime

from elspeth.contracts.enums import (
    Determinism,
    NodeStateStatus,
    NodeType,
    RoutingMode,
    RunStatus,
    BatchStatus,
    CallStatus,
    CallType,
    ExportStatus,
)


@dataclass
class Run:
    """A single execution of a pipeline.

    Strict contract - status must be RunStatus enum.
    """

    run_id: str
    started_at: datetime
    config_hash: str
    settings_json: str
    canonical_version: str
    status: RunStatus  # Strict: enum only
    completed_at: datetime | None = None
    reproducibility_grade: str | None = None
    export_status: ExportStatus | None = None  # Strict: enum only
    export_error: str | None = None
    exported_at: datetime | None = None
    export_format: str | None = None
    export_sink: str | None = None


@dataclass
class Node:
    """A node (plugin instance) in the execution graph.

    Strict contract - node_type and determinism must be enums.
    """

    node_id: str
    run_id: str
    plugin_name: str
    node_type: NodeType  # Strict: enum only
    plugin_version: str
    determinism: Determinism  # Strict: enum only
    config_hash: str
    config_json: str
    registered_at: datetime
    schema_hash: str | None = None
    sequence_in_pipeline: int | None = None


@dataclass
class Edge:
    """An edge in the execution graph.

    Strict contract - default_mode must be RoutingMode enum.
    """

    edge_id: str
    run_id: str
    from_node_id: str
    to_node_id: str
    label: str
    default_mode: RoutingMode  # Strict: enum only
    created_at: datetime


@dataclass
class Row:
    """A source row loaded into the system."""

    row_id: str
    run_id: str
    source_node_id: str
    row_index: int
    source_data_hash: str
    created_at: datetime
    source_data_ref: str | None = None


@dataclass
class Token:
    """A row instance flowing through a specific DAG path."""

    token_id: str
    row_id: str
    created_at: datetime
    fork_group_id: str | None = None
    join_group_id: str | None = None
    branch_name: str | None = None
    step_in_pipeline: int | None = None


@dataclass
class TokenParent:
    """Parent relationship for tokens (supports multi-parent joins)."""

    token_id: str
    parent_token_id: str
    ordinal: int
```

### Step 2: Create repositories.py

Create `src/elspeth/core/landscape/repositories.py`:

```python
"""Repository layer for Landscape audit models.

Handles the seam between SQLAlchemy rows (strings) and domain objects
(strict enum types). This is NOT a trust boundary - if the database
has bad data, we crash. That's intentional per Data Manifesto.

Per Data Manifesto: The audit database is OUR data. Bad data = crash.
"""

from typing import Any

from elspeth.contracts import (
    Run,
    Node,
    Edge,
    Row,
    Token,
    TokenParent,
    RunStatus,
    NodeType,
    Determinism,
    RoutingMode,
    ExportStatus,
)


class RunRepository:
    """Repository for Run records."""

    def __init__(self, session: Any) -> None:
        self.session = session

    def load(self, row: Any) -> Run:
        """Load Run from database row.

        Converts string fields to enums. Crashes on invalid data.
        """
        return Run(
            run_id=row.run_id,
            started_at=row.started_at,
            config_hash=row.config_hash,
            settings_json=row.settings_json,
            canonical_version=row.canonical_version,
            status=RunStatus(row.status),  # Convert HERE
            completed_at=row.completed_at,
            reproducibility_grade=row.reproducibility_grade,
            export_status=ExportStatus(row.export_status) if row.export_status else None,
            export_error=row.export_error,
            exported_at=row.exported_at,
            export_format=row.export_format,
            export_sink=row.export_sink,
        )


class NodeRepository:
    """Repository for Node records."""

    def __init__(self, session: Any) -> None:
        self.session = session

    def load(self, row: Any) -> Node:
        """Load Node from database row.

        Converts node_type and determinism strings to enums.
        """
        return Node(
            node_id=row.node_id,
            run_id=row.run_id,
            plugin_name=row.plugin_name,
            node_type=NodeType(row.node_type),  # Convert HERE
            plugin_version=row.plugin_version,
            determinism=Determinism(row.determinism),  # Convert HERE
            config_hash=row.config_hash,
            config_json=row.config_json,
            registered_at=row.registered_at,
            schema_hash=row.schema_hash,
            sequence_in_pipeline=row.sequence_in_pipeline,
        )


class EdgeRepository:
    """Repository for Edge records."""

    def __init__(self, session: Any) -> None:
        self.session = session

    def load(self, row: Any) -> Edge:
        """Load Edge from database row.

        Converts default_mode string to RoutingMode enum.
        """
        return Edge(
            edge_id=row.edge_id,
            run_id=row.run_id,
            from_node_id=row.from_node_id,
            to_node_id=row.to_node_id,
            label=row.label,
            default_mode=RoutingMode(row.default_mode),  # Convert HERE
            created_at=row.created_at,
        )


class RowRepository:
    """Repository for Row records."""

    def __init__(self, session: Any) -> None:
        self.session = session

    def load(self, row: Any) -> Row:
        """Load Row from database row.

        No enum conversion needed - all fields are primitives.
        """
        return Row(
            row_id=row.row_id,
            run_id=row.run_id,
            source_node_id=row.source_node_id,
            row_index=row.row_index,
            source_data_hash=row.source_data_hash,
            created_at=row.created_at,
            source_data_ref=row.source_data_ref,
        )


class TokenRepository:
    """Repository for Token records."""

    def __init__(self, session: Any) -> None:
        self.session = session

    def load(self, row: Any) -> Token:
        """Load Token from database row.

        No enum conversion needed - all fields are primitives.
        """
        return Token(
            token_id=row.token_id,
            row_id=row.row_id,
            created_at=row.created_at,
            fork_group_id=row.fork_group_id,
            join_group_id=row.join_group_id,
            branch_name=row.branch_name,
            step_in_pipeline=row.step_in_pipeline,
        )


class TokenParentRepository:
    """Repository for TokenParent records."""

    def __init__(self, session: Any) -> None:
        self.session = session

    def load(self, row: Any) -> TokenParent:
        """Load TokenParent from database row."""
        return TokenParent(
            token_id=row.token_id,
            parent_token_id=row.parent_token_id,
            ordinal=row.ordinal,
        )
```

### Step 3: Update contracts __init__.py

Add core audit exports to `src/elspeth/contracts/__init__.py`:

```python
from elspeth.contracts.audit import (
    Run,
    Node,
    Edge,
    Row,
    Token,
    TokenParent,
)

# Add to __all__
__all__ = [
    # ... existing ...
    # audit - core
    "Run",
    "Node",
    "Edge",
    "Row",
    "Token",
    "TokenParent",
]
```

### Step 4: Write tests for audit models

Create `tests/contracts/test_audit.py`:

```python
"""Tests for audit contracts."""

from datetime import datetime

import pytest


class TestRun:
    """Tests for Run contract."""

    def test_requires_enum_for_status(self) -> None:
        """Run.status must be RunStatus enum."""
        from elspeth.contracts import Run, RunStatus

        run = Run(
            run_id="run-1",
            started_at=datetime.now(),
            config_hash="abc123",
            settings_json="{}",
            canonical_version="1.0.0",
            status=RunStatus.RUNNING,  # Correct: enum
        )
        assert run.status == RunStatus.RUNNING


class TestNode:
    """Tests for Node contract."""

    def test_requires_enums_for_type_and_determinism(self) -> None:
        """Node.node_type and determinism must be enums."""
        from elspeth.contracts import Node, NodeType, Determinism

        node = Node(
            node_id="node-1",
            run_id="run-1",
            plugin_name="csv_source",
            node_type=NodeType.SOURCE,  # Correct: enum
            plugin_version="1.0.0",
            determinism=Determinism.DETERMINISTIC,  # Correct: enum
            config_hash="abc123",
            config_json="{}",
            registered_at=datetime.now(),
        )
        assert node.node_type == NodeType.SOURCE
        assert node.determinism == Determinism.DETERMINISTIC


class TestEdge:
    """Tests for Edge contract."""

    def test_requires_enum_for_mode(self) -> None:
        """Edge.default_mode must be RoutingMode enum."""
        from elspeth.contracts import Edge, RoutingMode

        edge = Edge(
            edge_id="edge-1",
            run_id="run-1",
            from_node_id="node-1",
            to_node_id="node-2",
            label="continue",
            default_mode=RoutingMode.MOVE,  # Correct: enum
            created_at=datetime.now(),
        )
        assert edge.default_mode == RoutingMode.MOVE
```

### Step 5: Write tests for repositories

Create `tests/core/landscape/test_repositories.py`:

```python
"""Tests for Landscape repositories."""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from elspeth.contracts import (
    Run,
    Node,
    Edge,
    RunStatus,
    NodeType,
    Determinism,
    RoutingMode,
)
from elspeth.core.landscape.repositories import (
    RunRepository,
    NodeRepository,
    EdgeRepository,
)


@pytest.fixture
def mock_run_row() -> MagicMock:
    """Mock SQLAlchemy row for Run."""
    row = MagicMock()
    row.run_id = "run-1"
    row.started_at = datetime.now()
    row.config_hash = "abc123"
    row.settings_json = "{}"
    row.canonical_version = "1.0.0"
    row.status = "running"  # String from DB
    row.completed_at = None
    row.reproducibility_grade = None
    row.export_status = None
    row.export_error = None
    row.exported_at = None
    row.export_format = None
    row.export_sink = None
    return row


@pytest.fixture
def mock_node_row() -> MagicMock:
    """Mock SQLAlchemy row for Node."""
    row = MagicMock()
    row.node_id = "node-1"
    row.run_id = "run-1"
    row.plugin_name = "csv_source"
    row.node_type = "source"  # String from DB
    row.plugin_version = "1.0.0"
    row.determinism = "deterministic"  # String from DB
    row.config_hash = "abc123"
    row.config_json = "{}"
    row.registered_at = datetime.now()
    row.schema_hash = None
    row.sequence_in_pipeline = 0
    return row


@pytest.fixture
def mock_edge_row() -> MagicMock:
    """Mock SQLAlchemy row for Edge."""
    row = MagicMock()
    row.edge_id = "edge-1"
    row.run_id = "run-1"
    row.from_node_id = "node-1"
    row.to_node_id = "node-2"
    row.label = "continue"
    row.default_mode = "move"  # String from DB
    row.created_at = datetime.now()
    return row


class TestRunRepository:
    """Tests for RunRepository."""

    def test_load_converts_status_to_enum(self, mock_run_row: MagicMock) -> None:
        """Repository converts status string to RunStatus enum."""
        repo = RunRepository(session=None)
        run = repo.load(mock_run_row)

        assert run.status == RunStatus.RUNNING
        assert isinstance(run.status, RunStatus)

    def test_load_crashes_on_invalid_status(self, mock_run_row: MagicMock) -> None:
        """Invalid status in DB crashes - audit integrity failure."""
        mock_run_row.status = "garbage"
        repo = RunRepository(session=None)

        with pytest.raises(ValueError, match="'garbage' is not a valid RunStatus"):
            repo.load(mock_run_row)


class TestNodeRepository:
    """Tests for NodeRepository."""

    def test_load_converts_enums(self, mock_node_row: MagicMock) -> None:
        """Repository converts node_type and determinism to enums."""
        repo = NodeRepository(session=None)
        node = repo.load(mock_node_row)

        assert node.node_type == NodeType.SOURCE
        assert node.determinism == Determinism.DETERMINISTIC

    def test_load_crashes_on_invalid_node_type(self, mock_node_row: MagicMock) -> None:
        """Invalid node_type in DB crashes."""
        mock_node_row.node_type = "garbage"
        repo = NodeRepository(session=None)

        with pytest.raises(ValueError):
            repo.load(mock_node_row)

    def test_load_crashes_on_invalid_determinism(self, mock_node_row: MagicMock) -> None:
        """Invalid determinism in DB crashes."""
        mock_node_row.determinism = "garbage"
        repo = NodeRepository(session=None)

        with pytest.raises(ValueError):
            repo.load(mock_node_row)


class TestEdgeRepository:
    """Tests for EdgeRepository."""

    def test_load_converts_mode_to_enum(self, mock_edge_row: MagicMock) -> None:
        """Repository converts default_mode string to RoutingMode enum."""
        repo = EdgeRepository(session=None)
        edge = repo.load(mock_edge_row)

        assert edge.default_mode == RoutingMode.MOVE
        assert isinstance(edge.default_mode, RoutingMode)

    def test_load_crashes_on_invalid_mode(self, mock_edge_row: MagicMock) -> None:
        """Invalid mode in DB crashes - audit integrity failure."""
        mock_edge_row.default_mode = "garbage"
        repo = EdgeRepository(session=None)

        with pytest.raises(ValueError, match="'garbage' is not a valid RoutingMode"):
            repo.load(mock_edge_row)
```

### Step 6: Run tests

Run: `pytest tests/contracts/test_audit.py tests/core/landscape/test_repositories.py -v`
Expected: PASS

### Step 7: Commit

```bash
git add src/elspeth/contracts/audit.py src/elspeth/core/landscape/repositories.py
git add tests/contracts/test_audit.py tests/core/landscape/test_repositories.py
git commit -m "feat(contracts): add audit.py core models with repository layer

- Run, Node, Edge, Row, Token, TokenParent with strict enum types
- Repository layer converts DB strings to enums
- Crashes on invalid data per Data Manifesto"
```

---

## Task 6b: Add NodeState Variants to audit.py

**Context:** NodeState is a discriminated union with three variants. Each variant has a `status` field that discriminates it.

**Files:**
- Modify: `src/elspeth/contracts/audit.py`
- Modify: `src/elspeth/contracts/__init__.py`
- Modify: `tests/contracts/test_audit.py`

### Step 1: Add NodeState variants to audit.py

Add to `src/elspeth/contracts/audit.py`:

```python
from typing import Literal


@dataclass(frozen=True)
class NodeStateOpen:
    """A node state currently being processed.

    Invariants:
    - No output_hash (not produced yet)
    - No completed_at (not completed)
    - No duration_ms (not finished timing)
    """

    state_id: str
    token_id: str
    node_id: str
    step_index: int
    attempt: int
    status: Literal[NodeStateStatus.OPEN]
    input_hash: str
    started_at: datetime
    context_before_json: str | None = None


@dataclass(frozen=True)
class NodeStateCompleted:
    """A node state that completed successfully.

    Invariants:
    - Has output_hash (produced output)
    - Has completed_at (finished)
    - Has duration_ms (timing complete)
    """

    state_id: str
    token_id: str
    node_id: str
    step_index: int
    attempt: int
    status: Literal[NodeStateStatus.COMPLETED]
    input_hash: str
    started_at: datetime
    output_hash: str
    completed_at: datetime
    duration_ms: float
    context_before_json: str | None = None
    context_after_json: str | None = None


@dataclass(frozen=True)
class NodeStateFailed:
    """A node state that failed during processing.

    Invariants:
    - Has completed_at (finished, with failure)
    - Has duration_ms (timing complete)
    - May have error_json
    """

    state_id: str
    token_id: str
    node_id: str
    step_index: int
    attempt: int
    status: Literal[NodeStateStatus.FAILED]
    input_hash: str
    started_at: datetime
    completed_at: datetime
    duration_ms: float
    error_json: str | None = None
    output_hash: str | None = None
    context_before_json: str | None = None
    context_after_json: str | None = None


# Discriminated union type
NodeState = NodeStateOpen | NodeStateCompleted | NodeStateFailed
```

### Step 2: Update contracts __init__.py

```python
from elspeth.contracts.audit import (
    # ... existing ...
    NodeStateOpen,
    NodeStateCompleted,
    NodeStateFailed,
    NodeState,
)
```

### Step 3: Add tests

Add to `tests/contracts/test_audit.py`:

```python
class TestNodeStateVariants:
    """Tests for NodeState discriminated union."""

    def test_open_state_has_literal_status(self) -> None:
        """NodeStateOpen.status is Literal[OPEN]."""
        from elspeth.contracts import NodeStateOpen, NodeStateStatus

        state = NodeStateOpen(
            state_id="state-1",
            token_id="token-1",
            node_id="node-1",
            step_index=0,
            attempt=1,
            status=NodeStateStatus.OPEN,
            input_hash="abc123",
            started_at=datetime.now(),
        )
        assert state.status == NodeStateStatus.OPEN

    def test_completed_state_requires_output(self) -> None:
        """NodeStateCompleted requires output_hash and completed_at."""
        from elspeth.contracts import NodeStateCompleted, NodeStateStatus

        state = NodeStateCompleted(
            state_id="state-1",
            token_id="token-1",
            node_id="node-1",
            step_index=0,
            attempt=1,
            status=NodeStateStatus.COMPLETED,
            input_hash="abc123",
            started_at=datetime.now(),
            output_hash="def456",  # Required
            completed_at=datetime.now(),  # Required
            duration_ms=100.0,  # Required
        )
        assert state.output_hash == "def456"

    def test_union_type_annotation(self) -> None:
        """NodeState is union of all variants."""
        from elspeth.contracts import NodeState, NodeStateOpen

        # Type checker accepts any variant
        state: NodeState = NodeStateOpen(
            state_id="state-1",
            token_id="token-1",
            node_id="node-1",
            step_index=0,
            attempt=1,
            status=NodeStateStatus.OPEN,
            input_hash="abc123",
            started_at=datetime.now(),
        )
        assert state is not None
```

### Step 4: Run tests and commit

Run: `pytest tests/contracts/test_audit.py -v`

```bash
git add -u
git commit -m "feat(contracts): add NodeState discriminated union variants"
```

---

## Task 6c: Add Events and Calls to audit.py

**Context:** Call, Artifact, RoutingEvent, Batch, BatchMember, BatchOutput. Some have enum fields needing repository conversion.

**Files:**
- Modify: `src/elspeth/contracts/audit.py`
- Modify: `src/elspeth/core/landscape/repositories.py`
- Modify: `src/elspeth/contracts/__init__.py`
- Modify: `tests/contracts/test_audit.py`

### Step 1: Add to audit.py

```python
@dataclass
class Call:
    """An external call made during node processing.

    Strict contract - call_type and status must be enums.
    """

    call_id: str
    state_id: str
    call_index: int
    call_type: CallType  # Strict: enum only
    status: CallStatus  # Strict: enum only
    request_hash: str
    created_at: datetime
    request_ref: str | None = None
    response_hash: str | None = None
    response_ref: str | None = None
    error_json: str | None = None
    latency_ms: float | None = None


@dataclass
class Artifact:
    """An artifact produced by a sink."""

    artifact_id: str
    run_id: str
    produced_by_state_id: str
    sink_node_id: str
    artifact_type: str  # Not enum - user-defined (csv, json, webhook, etc.)
    path_or_uri: str
    content_hash: str
    size_bytes: int
    created_at: datetime


@dataclass
class RoutingEvent:
    """A routing decision at a gate node.

    Strict contract - mode must be RoutingMode enum.
    """

    event_id: str
    state_id: str
    edge_id: str
    routing_group_id: str
    ordinal: int
    mode: RoutingMode  # Strict: enum only
    created_at: datetime
    reason_hash: str | None = None
    reason_ref: str | None = None


@dataclass
class Batch:
    """An aggregation batch collecting tokens.

    Strict contract - status must be BatchStatus enum.
    """

    batch_id: str
    run_id: str
    aggregation_node_id: str
    attempt: int
    status: BatchStatus  # Strict: enum only
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

### Step 2: Add repositories

Add to `src/elspeth/core/landscape/repositories.py`:

```python
from elspeth.contracts import (
    Call,
    RoutingEvent,
    Batch,
    CallType,
    CallStatus,
    BatchStatus,
)


class CallRepository:
    """Repository for Call records."""

    def __init__(self, session: Any) -> None:
        self.session = session

    def load(self, row: Any) -> Call:
        """Load Call from database row."""
        return Call(
            call_id=row.call_id,
            state_id=row.state_id,
            call_index=row.call_index,
            call_type=CallType(row.call_type),  # Convert HERE
            status=CallStatus(row.status),  # Convert HERE
            request_hash=row.request_hash,
            created_at=row.created_at,
            request_ref=row.request_ref,
            response_hash=row.response_hash,
            response_ref=row.response_ref,
            error_json=row.error_json,
            latency_ms=row.latency_ms,
        )


class RoutingEventRepository:
    """Repository for RoutingEvent records."""

    def __init__(self, session: Any) -> None:
        self.session = session

    def load(self, row: Any) -> RoutingEvent:
        """Load RoutingEvent from database row."""
        return RoutingEvent(
            event_id=row.event_id,
            state_id=row.state_id,
            edge_id=row.edge_id,
            routing_group_id=row.routing_group_id,
            ordinal=row.ordinal,
            mode=RoutingMode(row.mode),  # Convert HERE
            created_at=row.created_at,
            reason_hash=row.reason_hash,
            reason_ref=row.reason_ref,
        )


class BatchRepository:
    """Repository for Batch records."""

    def __init__(self, session: Any) -> None:
        self.session = session

    def load(self, row: Any) -> Batch:
        """Load Batch from database row."""
        return Batch(
            batch_id=row.batch_id,
            run_id=row.run_id,
            aggregation_node_id=row.aggregation_node_id,
            attempt=row.attempt,
            status=BatchStatus(row.status),  # Convert HERE
            created_at=row.created_at,
            aggregation_state_id=row.aggregation_state_id,
            trigger_reason=row.trigger_reason,
            completed_at=row.completed_at,
        )
```

### Step 3: Update exports and tests

Update `__init__.py` and add tests for the new types.

### Step 4: Run tests and commit

```bash
git add -u
git commit -m "feat(contracts): add Call, Artifact, RoutingEvent, Batch to audit.py"
```

---

## Task 6d: Add Lineage Types to audit.py

**Context:** Checkpoint for crash recovery. RowLineage fixes a gap where `LineageResult.source_row` was `Row` (DB record only) but explain output needs the resolved payload. RowLineage = Row fields + resolved payload.

**Integration Fix:** After this task, `LineageResult.source_row` should be `RowLineage` (not `Row`), and `explain()` should use `explain_row()` internally to resolve payloads.

**Files:**
- Modify: `src/elspeth/contracts/audit.py`
- Modify: `src/elspeth/contracts/__init__.py`
- Modify: `src/elspeth/core/landscape/lineage.py` (update LineageResult.source_row type)
- Modify: `src/elspeth/core/landscape/recorder.py` (update explain_row to match new RowLineage)

### Step 1: Add to audit.py

```python
@dataclass
class Checkpoint:
    """Checkpoint for crash recovery.

    Captures run progress at row/transform boundaries.
    """

    checkpoint_id: str
    run_id: str
    token_id: str
    node_id: str
    sequence_number: int
    created_at: datetime | None
    aggregation_state_json: str | None = None


@dataclass
class RowLineage:
    """Source row with resolved payload for explain output.

    Combines Row DB record fields with resolved payload data.
    Used by LineageResult.source_row for complete explain output.

    Supports graceful payload degradation - hash always preserved,
    actual data may be unavailable after retention purge.
    """

    # From Row (DB record fields)
    row_id: str
    run_id: str
    source_node_id: str
    row_index: int
    source_data_hash: str  # Consistent naming with Row
    created_at: datetime

    # Resolved payload (from PayloadStore)
    source_data: dict[str, object] | None  # None if purged
    payload_available: bool
```

### Step 2: Update LineageResult in lineage.py

Change `source_row: Row` to `source_row: RowLineage`:

```python
from elspeth.contracts import RowLineage

@dataclass
class LineageResult:
    token: Token
    source_row: RowLineage  # Changed from Row
    node_states: list[NodeState]
    # ... rest unchanged
```

### Step 3: Update explain() to use explain_row()

In `lineage.py`, replace `get_row()` with payload resolution:

```python
# Old: row = recorder.get_row(token.row_id)
# New: resolve payload via explain_row
source_row = recorder.explain_row(run_id, token.row_id)
if source_row is None:
    return None
```

### Step 4: Update explain_row() in recorder.py

Update to construct the new RowLineage with all fields:

```python
return RowLineage(
    row_id=row.row_id,
    run_id=row.run_id,
    source_node_id=row.source_node_id,
    row_index=row.row_index,
    source_data_hash=row.source_data_hash,
    created_at=row.created_at,
    source_data=source_data,
    payload_available=payload_available,
)
```

### Step 5: Update exports, add tests, commit

```bash
git commit -m "feat(contracts): add Checkpoint and RowLineage to audit.py

RowLineage fixes integration gap: combines Row DB fields with resolved
payload for complete explain output. LineageResult.source_row now uses
RowLineage instead of Row."
```

---

## Task 7: Create contracts/data.py

**Context:** PluginSchema base class for plugin input/output schemas.

**Files:**
- Create: `src/elspeth/contracts/data.py`
- Modify: `src/elspeth/contracts/__init__.py`
- Create: `tests/contracts/test_data.py`

### Step 1: Create data.py

Create `src/elspeth/contracts/data.py`:

```python
"""Plugin data schema contracts.

PluginSchema is the base class for plugin input/output schemas.
Plugins declare their expected data shape by subclassing this.
"""

from pydantic import BaseModel


class PluginSchema(BaseModel):
    """Base class for plugin input/output schemas.

    Subclass to define the expected shape of data for a plugin:

        class MyInputSchema(PluginSchema):
            name: str
            value: int

    Uses Pydantic for validation - this is a trust boundary
    (user data entering the system).
    """

    model_config = {"frozen": True, "extra": "forbid"}
```

### Step 2: Update exports

Add to `src/elspeth/contracts/__init__.py`:

```python
from elspeth.contracts.data import PluginSchema
```

### Step 3: Write tests

Create `tests/contracts/test_data.py`:

```python
"""Tests for data contracts."""

import pytest
from pydantic import ValidationError


class TestPluginSchema:
    """Tests for PluginSchema base class."""

    def test_subclass_validates_input(self) -> None:
        """PluginSchema subclasses validate input."""
        from elspeth.contracts import PluginSchema

        class MySchema(PluginSchema):
            name: str
            value: int

        # Valid input
        schema = MySchema(name="test", value=42)
        assert schema.name == "test"

        # Invalid input raises
        with pytest.raises(ValidationError):
            MySchema(name="test", value="not_an_int")

    def test_schema_is_frozen(self) -> None:
        """PluginSchema instances are immutable."""
        from elspeth.contracts import PluginSchema

        class MySchema(PluginSchema):
            name: str

        schema = MySchema(name="test")
        with pytest.raises(ValidationError):
            schema.name = "changed"

    def test_schema_forbids_extra(self) -> None:
        """PluginSchema rejects unknown fields."""
        from elspeth.contracts import PluginSchema

        class MySchema(PluginSchema):
            name: str

        with pytest.raises(ValidationError):
            MySchema(name="test", unknown_field="value")
```

### Step 4: Run tests and commit

```bash
git add src/elspeth/contracts/data.py tests/contracts/test_data.py
git commit -m "feat(contracts): add PluginSchema base class"
```

---

## Task 8: Create AST Enforcement Script

**Context:** AST-based checker that verifies cross-boundary types are defined in contracts/. Uses whitelist for intentional exceptions.

**Files:**
- Create: `scripts/check_contracts.py`
- Create: `.contracts-whitelist.yaml`
- Modify: `pyproject.toml` (add script entry)

### Step 1: Create the enforcement script

Create `scripts/check_contracts.py`:

```python
#!/usr/bin/env python3
"""AST-based enforcement for contracts package.

Scans the codebase for dataclasses, TypedDicts, NamedTuples, and Enums
that are used across module boundaries. Reports violations where such
types are defined outside contracts/ without whitelist exemption.

Usage:
    python scripts/check_contracts.py
    python scripts/check_contracts.py --fix  # Show where to move types

Exit codes:
    0: All contracts properly centralized
    1: Violations found
"""

import ast
import sys
from pathlib import Path

import yaml


def load_whitelist(path: Path) -> set[str]:
    """Load whitelisted type definitions."""
    if not path.exists():
        return set()
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return set(data.get("allowed_external_types", []))


def find_type_definitions(file_path: Path) -> list[tuple[str, int, str]]:
    """Find dataclass, TypedDict, NamedTuple, Enum definitions in a file.

    Returns: List of (type_name, line_number, kind)
    """
    try:
        source = file_path.read_text()
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError):
        return []

    definitions = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            # Check for @dataclass decorator
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Name) and decorator.id == "dataclass":
                    definitions.append((node.name, node.lineno, "dataclass"))
                elif isinstance(decorator, ast.Call):
                    if isinstance(decorator.func, ast.Name) and decorator.func.id == "dataclass":
                        definitions.append((node.name, node.lineno, "dataclass"))

            # Check for TypedDict, NamedTuple, Enum base classes
            for base in node.bases:
                if isinstance(base, ast.Name):
                    if base.id == "TypedDict":
                        definitions.append((node.name, node.lineno, "TypedDict"))
                    elif base.id == "NamedTuple":
                        definitions.append((node.name, node.lineno, "NamedTuple"))
                    elif base.id == "Enum":
                        definitions.append((node.name, node.lineno, "Enum"))
                    elif base.id in ("BaseModel", "PluginSchema"):
                        # Pydantic models in config are OK (trust boundary)
                        pass

    return definitions


def find_cross_module_usage(src_dir: Path, type_name: str, defining_file: Path) -> list[Path]:
    """Find files that import a type from outside contracts/."""
    usages = []
    defining_module = defining_file.relative_to(src_dir).with_suffix("").as_posix().replace("/", ".")

    for py_file in src_dir.rglob("*.py"):
        if py_file == defining_file:
            continue

        try:
            source = py_file.read_text()
            tree = ast.parse(source)
        except (SyntaxError, UnicodeDecodeError):
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and defining_module in node.module:
                    for alias in node.names:
                        if alias.name == type_name:
                            usages.append(py_file)

    return usages


def main() -> int:
    """Run the contracts enforcement check."""
    src_dir = Path("src/elspeth")
    contracts_dir = src_dir / "contracts"
    whitelist_path = Path(".contracts-whitelist.yaml")

    whitelist = load_whitelist(whitelist_path)
    violations = []

    # Scan all Python files outside contracts/
    for py_file in src_dir.rglob("*.py"):
        if contracts_dir in py_file.parents or py_file.parent == contracts_dir:
            continue  # Skip contracts/ itself

        definitions = find_type_definitions(py_file)
        for type_name, line_no, kind in definitions:
            qualified_name = f"{py_file.relative_to(src_dir).with_suffix('')}:{type_name}"

            if qualified_name in whitelist:
                continue

            # Check if used across module boundaries
            usages = find_cross_module_usage(src_dir, type_name, py_file)
            if usages:
                violations.append({
                    "file": str(py_file),
                    "line": line_no,
                    "type": type_name,
                    "kind": kind,
                    "used_in": [str(u) for u in usages[:3]],  # First 3
                })

    if violations:
        print("❌ Contract violations found:\n")
        for v in violations:
            print(f"  {v['file']}:{v['line']}: {v['kind']} '{v['type']}'")
            print(f"    Used in: {', '.join(v['used_in'])}")
            print(f"    Fix: Move to src/elspeth/contracts/ or add to .contracts-whitelist.yaml\n")
        return 1

    print("✅ All cross-boundary types are properly centralized in contracts/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

### Step 2: Create whitelist

Create `.contracts-whitelist.yaml`:

```yaml
# Types intentionally defined outside contracts/
# These are internal to their module and not used across boundaries
allowed_external_types:
  # TUI display types - presentation-specific, not cross-boundary contracts
  - "tui/types:TokenDisplayInfo"
```

### Step 3: Add to pyproject.toml

Add script entry:

```toml
[project.scripts]
check-contracts = "scripts.check_contracts:main"
```

### Step 4: Write tests

Create `tests/scripts/test_check_contracts.py`:

```python
"""Tests for contracts enforcement script."""

from pathlib import Path
from scripts.check_contracts import find_type_definitions, load_whitelist


def test_finds_dataclass_definitions(tmp_path: Path) -> None:
    """Finds @dataclass decorated classes."""
    test_file = tmp_path / "test.py"
    test_file.write_text('''
from dataclasses import dataclass

@dataclass
class MyType:
    name: str
''')

    definitions = find_type_definitions(test_file)
    assert len(definitions) == 1
    assert definitions[0][0] == "MyType"
    assert definitions[0][2] == "dataclass"


def test_finds_enum_definitions(tmp_path: Path) -> None:
    """Finds Enum subclasses."""
    test_file = tmp_path / "test.py"
    test_file.write_text('''
from enum import Enum

class MyEnum(Enum):
    A = "a"
''')

    definitions = find_type_definitions(test_file)
    assert len(definitions) == 1
    assert definitions[0][0] == "MyEnum"
    assert definitions[0][2] == "Enum"


def test_whitelist_loading(tmp_path: Path) -> None:
    """Loads whitelist from YAML."""
    whitelist_file = tmp_path / ".contracts-whitelist.yaml"
    whitelist_file.write_text('''
allowed_external_types:
  - "foo/bar:MyType"
''')

    whitelist = load_whitelist(whitelist_file)
    assert "foo/bar:MyType" in whitelist
```

### Step 5: Run and commit

Run: `python scripts/check_contracts.py`

```bash
git add scripts/check_contracts.py .contracts-whitelist.yaml
git commit -m "feat(contracts): add AST enforcement script

Scans codebase for cross-boundary types defined outside contracts/.
Uses whitelist for intentional exceptions."
```

---

## Task 9: Update Existing Code to Use Contracts

**Context:** Update all existing imports to use contracts package. Remove old definitions.

**Files:**
- Modify: `src/elspeth/core/landscape/models.py` (remove migrated types)
- Modify: `src/elspeth/plugins/base.py` (update imports)
- Modify: `src/elspeth/engine/*.py` (update imports)
- Run: `scripts/check_contracts.py` to verify

### Step 1: Update imports across codebase

Use grep to find all imports of migrated types and update them:

```bash
# Find all imports of old locations
grep -r "from elspeth.core.landscape.models import" src/
grep -r "from elspeth.plugins.enums import" src/

# Update to use contracts
# from elspeth.core.landscape.models import Run, Node
# becomes:
# from elspeth.contracts import Run, Node
```

### Step 2: Remove old definitions

After updating all imports, delete the migrated definitions from their original locations. Keep only internal types that aren't used across boundaries.

### Step 3: Run enforcement and tests

```bash
python scripts/check_contracts.py
pytest tests/ -v
```

### Step 4: Final commit

```bash
git add -u
git commit -m "refactor: migrate all imports to use contracts package

- All cross-boundary types now imported from elspeth.contracts
- Old definitions removed from original locations
- AST enforcement passes"
```

---

## Summary

| Task | Description | Key Points |
|------|-------------|------------|
| 5 | config.py | Pydantic re-exports (trust boundary validation) |
| 6a | audit.py core | Run, Node, Edge, Row, Token + repositories |
| 6b | audit.py NodeState | Discriminated union variants |
| 6c | audit.py events | Call, Artifact, RoutingEvent, Batch |
| 6d | audit.py lineage | Checkpoint, RowLineage |
| 7 | data.py | PluginSchema base class |
| 8 | enforcement | AST checker + whitelist |
| 9 | migration | Update imports, remove old definitions |

**Key principles:**
- Dataclasses are strict contracts (enum types only)
- Repository layer handles DB string→enum conversion
- Pydantic for trust boundaries (config, plugin schemas)
- AST enforcement prevents future drift
