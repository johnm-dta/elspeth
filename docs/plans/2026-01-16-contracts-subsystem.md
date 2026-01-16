# Contracts Subsystem Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a centralized `contracts/` package containing all cross-boundary data types, with AST-based enforcement to prevent drift.

**Architecture:** All dataclasses, enums, TypedDicts, and NamedTuples that cross subsystem boundaries live in `src/elspeth/contracts/`. Internal types are explicitly whitelisted. An AST checker script enforces this at CI time. Migration is bottom-up by dependency order with clean breaks (no legacy re-exports).

**Tech Stack:** Python 3.11+, ast module, dataclasses, Pydantic, PyYAML (for whitelist)

**Dependencies:**
- None (this is foundational infrastructure)

---

## Package Structure

```
src/elspeth/contracts/
├── __init__.py      # Re-exports everything
├── enums.py         # Status codes, modes, kinds
├── identity.py      # TokenInfo, entity identifiers
├── results.py       # TransformResult, GateResult, RowResult, ArtifactDescriptor
├── routing.py       # RoutingAction, RoutingSpec, EdgeInfo, RouteLabel
├── config.py        # ResolvedConfig, PipelineSettings
├── audit.py         # Run, Node, Edge, Row, Token, NodeState, LineageResult, etc.
└── data.py          # PluginSchema base
```

## Migration Order

Bottom-up by dependency:
1. `enums.py` — no dependencies
2. `identity.py` — depends on enums
3. `routing.py` — depends on enums
4. `results.py` — depends on enums, routing
5. `config.py` — depends on enums
6. `audit.py` — depends on enums, identity, routing
7. `data.py` — depends on nothing

---

## Task 1: Create contracts package with enums.py

**Context:** Consolidate all enums that cross boundaries into one file. This is the foundation everything else depends on.

**Files:**
- Create: `src/elspeth/contracts/__init__.py`
- Create: `src/elspeth/contracts/enums.py`
- Modify: `src/elspeth/plugins/enums.py` (remove migrated enums)
- Modify: `src/elspeth/core/landscape/models.py` (remove migrated enums)
- Modify: `src/elspeth/engine/processor.py` (update imports)
- Create: `tests/contracts/__init__.py`
- Create: `tests/contracts/test_enums.py`

### Step 1: Create the contracts package

Create `src/elspeth/contracts/__init__.py`:

```python
"""Shared contracts for cross-boundary data types.

All dataclasses, enums, TypedDicts, and NamedTuples that cross subsystem
boundaries MUST be defined here. Internal types are whitelisted in
.contracts-whitelist.yaml.

Import pattern:
    from elspeth.contracts import NodeType, TransformResult, Run
"""

from elspeth.contracts.enums import (
    BatchStatus,
    CallStatus,
    CallType,
    Determinism,
    ExportStatus,
    NodeStateStatus,
    NodeType,
    RoutingKind,
    RoutingMode,
    RowOutcome,
    RunStatus,
    TransformStatus,
)

__all__ = [
    # enums
    "BatchStatus",
    "CallStatus",
    "CallType",
    "Determinism",
    "ExportStatus",
    "NodeStateStatus",
    "NodeType",
    "RoutingKind",
    "RoutingMode",
    "RowOutcome",
    "RunStatus",
    "TransformStatus",
]
```

### Step 2: Create enums.py with all enums

Create `src/elspeth/contracts/enums.py`:

```python
"""All status codes, modes, and kinds used across subsystem boundaries.

These enums replace stringly-typed APIs throughout the codebase.
"""

from enum import Enum


class RunStatus(str, Enum):
    """Status of a pipeline run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class NodeStateStatus(str, Enum):
    """Status of a node processing a token."""

    OPEN = "open"
    COMPLETED = "completed"
    FAILED = "failed"


class ExportStatus(str, Enum):
    """Status of run export operation."""

    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


class BatchStatus(str, Enum):
    """Status of an aggregation batch."""

    DRAFT = "draft"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"


class NodeType(str, Enum):
    """Type of node in the execution graph."""

    SOURCE = "source"
    TRANSFORM = "transform"
    GATE = "gate"
    AGGREGATION = "aggregation"
    COALESCE = "coalesce"
    SINK = "sink"


class Determinism(str, Enum):
    """Determinism classification for plugins."""

    DETERMINISTIC = "deterministic"
    EXTERNAL_CALL = "external_call"
    IO_DEPENDENT = "io_dependent"
    NON_DETERMINISTIC = "non_deterministic"


class RoutingKind(str, Enum):
    """Kind of routing action from a gate."""

    CONTINUE = "continue"
    ROUTE = "route"
    FORK_TO_PATHS = "fork_to_paths"


class RoutingMode(str, Enum):
    """Mode for routing edges."""

    MOVE = "move"
    COPY = "copy"


class TransformStatus(str, Enum):
    """Status of a transform result."""

    SUCCESS = "success"
    ERROR = "error"


class RowOutcome(str, Enum):
    """Terminal outcome for a row in the pipeline."""

    COMPLETED = "completed"
    ROUTED = "routed"
    FORKED = "forked"
    FAILED = "failed"
    QUARANTINED = "quarantined"
    CONSUMED_IN_BATCH = "consumed_in_batch"
    COALESCED = "coalesced"


class CallType(str, Enum):
    """Type of external call (Phase 6)."""

    LLM = "llm"
    HTTP = "http"
    SQL = "sql"
    FILESYSTEM = "filesystem"


class CallStatus(str, Enum):
    """Status of an external call (Phase 6)."""

    SUCCESS = "success"
    ERROR = "error"
```

### Step 3: Write test for enums

Create `tests/contracts/__init__.py`:

```python
"""Tests for contracts package."""
```

Create `tests/contracts/test_enums.py`:

```python
"""Tests for contracts enums."""

import pytest


class TestEnumStringValues:
    """Verify enums serialize to expected string values."""

    def test_run_status_values(self) -> None:
        """RunStatus has expected string values."""
        from elspeth.contracts import RunStatus

        assert RunStatus.PENDING.value == "pending"
        assert RunStatus.RUNNING.value == "running"
        assert RunStatus.COMPLETED.value == "completed"
        assert RunStatus.FAILED.value == "failed"

    def test_node_type_values(self) -> None:
        """NodeType has expected string values."""
        from elspeth.contracts import NodeType

        assert NodeType.SOURCE.value == "source"
        assert NodeType.TRANSFORM.value == "transform"
        assert NodeType.GATE.value == "gate"
        assert NodeType.SINK.value == "sink"

    def test_routing_mode_values(self) -> None:
        """RoutingMode has expected string values."""
        from elspeth.contracts import RoutingMode

        assert RoutingMode.MOVE.value == "move"
        assert RoutingMode.COPY.value == "copy"

    def test_transform_status_values(self) -> None:
        """TransformStatus has expected string values."""
        from elspeth.contracts import TransformStatus

        assert TransformStatus.SUCCESS.value == "success"
        assert TransformStatus.ERROR.value == "error"


class TestEnumCoercion:
    """Verify enums can be created from string values."""

    def test_run_status_from_string(self) -> None:
        """Can create RunStatus from string."""
        from elspeth.contracts import RunStatus

        assert RunStatus("pending") == RunStatus.PENDING
        assert RunStatus("completed") == RunStatus.COMPLETED

    def test_invalid_value_raises(self) -> None:
        """Invalid string raises ValueError."""
        from elspeth.contracts import RunStatus

        with pytest.raises(ValueError):
            RunStatus("invalid")
```

### Step 4: Run tests

Run: `pytest tests/contracts/test_enums.py -v`
Expected: PASS

### Step 5: Update plugins/enums.py to re-export from contracts

Modify `src/elspeth/plugins/enums.py` to import from contracts (temporary bridge during migration):

```python
"""Plugin enums - re-exported from contracts.

DEPRECATED: Import directly from elspeth.contracts instead.
"""

from elspeth.contracts.enums import (
    Determinism,
    NodeType,
    RoutingKind,
)

__all__ = ["Determinism", "NodeType", "RoutingKind"]
```

**Note:** This re-export is temporary for this task only. Once all imports are updated in later tasks, we'll delete the re-exports.

### Step 6: Run full test suite

Run: `pytest tests/ -v --tb=short`
Expected: PASS (no regressions)

### Step 7: Commit

```bash
git add src/elspeth/contracts/ tests/contracts/
git add src/elspeth/plugins/enums.py
git commit -m "feat(contracts): create contracts package with enums

- Add contracts/ package as single source of truth for cross-boundary types
- Create enums.py with all status codes, modes, and kinds
- Add ExportStatus, BatchStatus, TransformStatus (previously strings)
- Add RoutingMode enum (replaces 'move'/'copy' strings)
- Temporary re-export from plugins/enums.py for compatibility"
```

---

## Task 2: Create identity.py with TokenInfo

**Context:** Move token identity structures to contracts.

**Files:**
- Create: `src/elspeth/contracts/identity.py`
- Modify: `src/elspeth/contracts/__init__.py` (add exports)
- Modify: `src/elspeth/engine/tokens.py` (remove TokenInfo, update imports)
- Create: `tests/contracts/test_identity.py`

### Step 1: Create identity.py

Create `src/elspeth/contracts/identity.py`:

```python
"""Entity identifiers and token structures.

These types answer: "How do we refer to things?"
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TokenInfo:
    """Identity and data for a token flowing through the DAG.

    Tokens track row instances through forks/joins:
    - row_id: Stable source row identity
    - token_id: Instance of row in a specific DAG path
    - branch_name: Which fork path this token is on (if forked)
    """

    row_id: str
    token_id: str
    row_data: dict[str, Any]
    branch_name: str | None = None
```

### Step 2: Update contracts __init__.py

Add to `src/elspeth/contracts/__init__.py`:

```python
from elspeth.contracts.identity import TokenInfo

__all__ = [
    # enums
    ...
    # identity
    "TokenInfo",
]
```

### Step 3: Write test

Create `tests/contracts/test_identity.py`:

```python
"""Tests for identity contracts."""


class TestTokenInfo:
    """Tests for TokenInfo."""

    def test_create_token_info(self) -> None:
        """Can create TokenInfo with required fields."""
        from elspeth.contracts import TokenInfo

        token = TokenInfo(
            row_id="row-123",
            token_id="tok-456",
            row_data={"field": "value"},
        )

        assert token.row_id == "row-123"
        assert token.token_id == "tok-456"
        assert token.row_data == {"field": "value"}
        assert token.branch_name is None

    def test_token_info_with_branch(self) -> None:
        """Can create TokenInfo with branch_name."""
        from elspeth.contracts import TokenInfo

        token = TokenInfo(
            row_id="row-123",
            token_id="tok-456",
            row_data={},
            branch_name="sentiment",
        )

        assert token.branch_name == "sentiment"

    def test_token_info_is_frozen(self) -> None:
        """TokenInfo is immutable."""
        from elspeth.contracts import TokenInfo
        import pytest

        token = TokenInfo(row_id="r", token_id="t", row_data={})

        with pytest.raises(AttributeError):
            token.row_id = "new"  # type: ignore
```

### Step 4: Run tests

Run: `pytest tests/contracts/test_identity.py -v`
Expected: PASS

### Step 5: Update engine/tokens.py to import from contracts

Modify `src/elspeth/engine/tokens.py` to import TokenInfo from contracts and remove local definition.

### Step 6: Run full test suite

Run: `pytest tests/ -v --tb=short`
Expected: PASS

### Step 7: Commit

```bash
git add src/elspeth/contracts/identity.py src/elspeth/contracts/__init__.py
git add tests/contracts/test_identity.py
git add src/elspeth/engine/tokens.py
git commit -m "feat(contracts): add identity.py with TokenInfo

- Move TokenInfo from engine/tokens.py to contracts/identity.py
- TokenInfo is frozen dataclass for immutability"
```

---

## Task 3: Create routing.py with routing contracts

**Context:** Consolidate routing action and edge types.

**Files:**
- Create: `src/elspeth/contracts/routing.py`
- Modify: `src/elspeth/contracts/__init__.py` (add exports)
- Modify: `src/elspeth/plugins/results.py` (remove RoutingAction)
- Modify: `src/elspeth/core/landscape/models.py` (remove RoutingSpec)
- Create: `tests/contracts/test_routing.py`

### Step 1: Create routing.py

Create `src/elspeth/contracts/routing.py`:

```python
"""Flow control and edge definitions.

These types answer: "Where does data go next?"
"""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Mapping

from elspeth.contracts.enums import RoutingKind, RoutingMode


def _freeze_dict(d: dict[str, Any] | None) -> Mapping[str, Any]:
    """Convert dict to immutable MappingProxyType."""
    if d is None:
        return MappingProxyType({})
    return MappingProxyType(d)


@dataclass(frozen=True)
class RoutingAction:
    """A routing decision from a gate.

    Gates return this to indicate where tokens should go next.
    Use the factory methods to create instances.
    """

    kind: RoutingKind
    destinations: tuple[str, ...] = ()
    reason: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    @classmethod
    def continue_(cls, *, reason: dict[str, Any] | None = None) -> "RoutingAction":
        """Continue to next node in pipeline."""
        return cls(
            kind=RoutingKind.CONTINUE,
            destinations=(),
            reason=_freeze_dict(reason),
        )

    @classmethod
    def route(cls, label: str, *, reason: dict[str, Any] | None = None) -> "RoutingAction":
        """Route to a specific labeled destination."""
        return cls(
            kind=RoutingKind.ROUTE,
            destinations=(label,),
            reason=_freeze_dict(reason),
        )

    @classmethod
    def fork_to_paths(
        cls, paths: list[str], *, reason: dict[str, Any] | None = None
    ) -> "RoutingAction":
        """Fork token to multiple parallel paths."""
        return cls(
            kind=RoutingKind.FORK_TO_PATHS,
            destinations=tuple(paths),
            reason=_freeze_dict(reason),
        )


@dataclass(frozen=True)
class RoutingSpec:
    """Specification for a routing edge in the recorded audit trail."""

    edge_id: str
    mode: RoutingMode

    def __post_init__(self) -> None:
        """Validate mode is valid RoutingMode."""
        if not isinstance(self.mode, RoutingMode):
            # Coerce string to enum if needed
            object.__setattr__(self, "mode", RoutingMode(self.mode))


@dataclass(frozen=True)
class EdgeInfo:
    """Information about an edge in the execution graph.

    Replaces tuple[str, str, dict[str, Any]] for type safety.
    """

    from_node: str
    to_node: str
    label: str
    mode: RoutingMode


@dataclass(frozen=True)
class RouteLabel:
    """Typed wrapper for route labels.

    Distinguishes the special 'continue' label from sink route names.
    """

    value: str
    is_continue: bool = False

    @classmethod
    def continue_(cls) -> "RouteLabel":
        """The special 'continue' label."""
        return cls(value="continue", is_continue=True)

    @classmethod
    def sink(cls, name: str) -> "RouteLabel":
        """A sink route label."""
        if name == "continue":
            raise ValueError("Use RouteLabel.continue_() for continue label")
        return cls(value=name, is_continue=False)
```

### Step 2: Update contracts __init__.py

Add exports for routing types.

### Step 3: Write tests

Create `tests/contracts/test_routing.py`:

```python
"""Tests for routing contracts."""

import pytest


class TestRoutingAction:
    """Tests for RoutingAction."""

    def test_continue_action(self) -> None:
        """Can create continue action."""
        from elspeth.contracts import RoutingAction, RoutingKind

        action = RoutingAction.continue_(reason={"rule": "default"})

        assert action.kind == RoutingKind.CONTINUE
        assert action.destinations == ()
        assert action.reason["rule"] == "default"

    def test_route_action(self) -> None:
        """Can create route action."""
        from elspeth.contracts import RoutingAction, RoutingKind

        action = RoutingAction.route("quarantine", reason={"score": 0.1})

        assert action.kind == RoutingKind.ROUTE
        assert action.destinations == ("quarantine",)

    def test_fork_action(self) -> None:
        """Can create fork action."""
        from elspeth.contracts import RoutingAction, RoutingKind

        action = RoutingAction.fork_to_paths(["sentiment", "classification"])

        assert action.kind == RoutingKind.FORK_TO_PATHS
        assert action.destinations == ("sentiment", "classification")

    def test_reason_is_immutable(self) -> None:
        """Reason dict is frozen."""
        from elspeth.contracts import RoutingAction

        action = RoutingAction.continue_(reason={"key": "value"})

        with pytest.raises(TypeError):
            action.reason["key"] = "new"  # type: ignore


class TestRoutingSpec:
    """Tests for RoutingSpec."""

    def test_create_with_enum(self) -> None:
        """Can create with RoutingMode enum."""
        from elspeth.contracts import RoutingSpec, RoutingMode

        spec = RoutingSpec(edge_id="edge-1", mode=RoutingMode.MOVE)

        assert spec.mode == RoutingMode.MOVE

    def test_coerces_string_to_enum(self) -> None:
        """String mode is coerced to enum."""
        from elspeth.contracts import RoutingSpec, RoutingMode

        spec = RoutingSpec(edge_id="edge-1", mode="copy")  # type: ignore

        assert spec.mode == RoutingMode.COPY


class TestEdgeInfo:
    """Tests for EdgeInfo."""

    def test_create_edge_info(self) -> None:
        """Can create EdgeInfo."""
        from elspeth.contracts import EdgeInfo, RoutingMode

        edge = EdgeInfo(
            from_node="node-1",
            to_node="node-2",
            label="continue",
            mode=RoutingMode.MOVE,
        )

        assert edge.from_node == "node-1"
        assert edge.to_node == "node-2"


class TestRouteLabel:
    """Tests for RouteLabel."""

    def test_continue_label(self) -> None:
        """Can create continue label."""
        from elspeth.contracts import RouteLabel

        label = RouteLabel.continue_()

        assert label.is_continue is True
        assert label.value == "continue"

    def test_sink_label(self) -> None:
        """Can create sink label."""
        from elspeth.contracts import RouteLabel

        label = RouteLabel.sink("quarantine")

        assert label.is_continue is False
        assert label.value == "quarantine"

    def test_sink_label_rejects_continue(self) -> None:
        """Cannot create sink label named 'continue'."""
        from elspeth.contracts import RouteLabel

        with pytest.raises(ValueError):
            RouteLabel.sink("continue")
```

### Step 4: Run tests

Run: `pytest tests/contracts/test_routing.py -v`
Expected: PASS

### Step 5: Update source files to import from contracts

Update imports in:
- `src/elspeth/plugins/results.py`
- `src/elspeth/core/landscape/models.py`
- `src/elspeth/engine/executors.py`

### Step 6: Run full test suite

Run: `pytest tests/ -v --tb=short`
Expected: PASS

### Step 7: Commit

```bash
git add src/elspeth/contracts/routing.py src/elspeth/contracts/__init__.py
git add tests/contracts/test_routing.py
git add src/elspeth/plugins/results.py src/elspeth/core/landscape/models.py
git commit -m "feat(contracts): add routing.py with flow control types

- Move RoutingAction from plugins/results.py
- Move RoutingSpec from landscape/models.py
- Add EdgeInfo to replace tuple[str, str, dict]
- Add RouteLabel to distinguish 'continue' from sink names"
```

---

## Task 4: Create results.py with operation outcomes

**Context:** Consolidate transform, gate, and row result types.

**Files:**
- Create: `src/elspeth/contracts/results.py`
- Modify: `src/elspeth/contracts/__init__.py`
- Modify: `src/elspeth/plugins/results.py` (remove migrated types)
- Modify: `src/elspeth/engine/processor.py` (remove RowResult)
- Create: `tests/contracts/test_results.py`

### Step 1: Create results.py

Create `src/elspeth/contracts/results.py`:

```python
"""Operation outcomes and results.

These types answer: "What did an operation produce?"
"""

from dataclasses import dataclass, field
from typing import Any, Literal

from elspeth.contracts.enums import RowOutcome, TransformStatus
from elspeth.contracts.routing import RoutingAction
from elspeth.contracts.identity import TokenInfo


@dataclass(frozen=True)
class TransformResult:
    """Result of a transform operation.

    Use the factory methods to create instances.
    """

    status: TransformStatus
    row: dict[str, Any] | None = None
    reason: dict[str, Any] | None = None
    retryable: bool = False

    @classmethod
    def success(cls, row: dict[str, Any]) -> "TransformResult":
        """Create successful result with output row."""
        return cls(status=TransformStatus.SUCCESS, row=row)

    @classmethod
    def error(
        cls,
        reason: dict[str, Any],
        *,
        retryable: bool = False,
    ) -> "TransformResult":
        """Create error result with reason."""
        return cls(
            status=TransformStatus.ERROR,
            reason=reason,
            retryable=retryable,
        )


@dataclass(frozen=True)
class GateResult:
    """Result of a gate evaluation.

    Contains the (possibly modified) row and routing action.
    """

    row: dict[str, Any]
    action: RoutingAction


@dataclass(frozen=True)
class AcceptResult:
    """Result of aggregation accept check.

    Indicates whether the row was accepted into a batch.
    """

    accepted: bool
    trigger: bool = False
    batch_id: str | None = None


# RowResult as discriminated union
@dataclass(frozen=True)
class RowResultCompleted:
    """Row completed processing and reached a sink."""

    outcome: Literal[RowOutcome.COMPLETED] = RowOutcome.COMPLETED
    token: TokenInfo = field(default_factory=lambda: None)  # type: ignore
    final_data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RowResultRouted:
    """Row was routed to a specific sink by a gate."""

    outcome: Literal[RowOutcome.ROUTED] = RowOutcome.ROUTED
    token: TokenInfo = field(default_factory=lambda: None)  # type: ignore
    final_data: dict[str, Any] = field(default_factory=dict)
    sink_name: str = ""  # Required for ROUTED


@dataclass(frozen=True)
class RowResultForked:
    """Row was forked to multiple paths."""

    outcome: Literal[RowOutcome.FORKED] = RowOutcome.FORKED
    token: TokenInfo = field(default_factory=lambda: None)  # type: ignore
    child_tokens: tuple[TokenInfo, ...] = ()


@dataclass(frozen=True)
class RowResultFailed:
    """Row processing failed."""

    outcome: Literal[RowOutcome.FAILED] = RowOutcome.FAILED
    token: TokenInfo = field(default_factory=lambda: None)  # type: ignore
    final_data: dict[str, Any] = field(default_factory=dict)
    error: dict[str, Any] = field(default_factory=dict)


RowResult = RowResultCompleted | RowResultRouted | RowResultForked | RowResultFailed


@dataclass(frozen=True)
class ArtifactDescriptor:
    """Descriptor for an artifact written by a sink.

    Replaces untyped dict with explicit fields.
    """

    kind: Literal["file", "database", "blob"]
    path_or_uri: str
    content_hash: str | None = None
    row_count: int | None = None
```

### Step 2: Update __init__.py

Add exports for result types.

### Step 3: Write tests

Create `tests/contracts/test_results.py` with tests for TransformResult, GateResult, RowResult variants, and ArtifactDescriptor.

### Step 4: Run tests

Run: `pytest tests/contracts/test_results.py -v`
Expected: PASS

### Step 5: Update source files

Update imports in plugins/results.py, engine/processor.py, engine/executors.py.

### Step 6: Run full test suite and commit

---

## Task 5: Create config.py with configuration contracts

**Context:** Create typed configuration structures to replace dict[str, Any].

**Files:**
- Create: `src/elspeth/contracts/config.py`
- Modify: `src/elspeth/contracts/__init__.py`
- Modify: `src/elspeth/core/config.py` (add resolve_config returning typed structure)
- Create: `tests/contracts/test_config.py`

### Step 1: Create config.py

Create `src/elspeth/contracts/config.py`:

```python
"""Configuration structures.

These types answer: "How is the system configured?"
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ResolvedConfig:
    """Fully resolved pipeline configuration.

    This replaces dict[str, Any] for config that flows to:
    - LandscapeRecorder.begin_run()
    - PipelineConfig.config

    All fields are validated before this is created.
    """

    datasource_plugin: str
    datasource_options: dict[str, Any]
    row_plugins: tuple[dict[str, Any], ...]
    sinks: dict[str, dict[str, Any]]
    landscape_url: str
    run_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for serialization/hashing."""
        return {
            "datasource": {
                "plugin": self.datasource_plugin,
                "options": self.datasource_options,
            },
            "row_plugins": list(self.row_plugins),
            "sinks": self.sinks,
            "landscape": {"url": self.landscape_url},
            "run_name": self.run_name,
        }


@dataclass(frozen=True)
class PipelineSettings:
    """Runtime settings for pipeline execution.

    Replaces PipelineConfig.config: dict[str, Any].
    """

    max_concurrent_rows: int = 1
    timeout_ms: int | None = None
    checkpoint_enabled: bool = False
    checkpoint_frequency: int | None = None
```

### Steps 2-7: Follow standard pattern

Write tests, update imports, run full suite, commit.

---

## Task 6: Create audit.py with Landscape models

**Context:** Move all Landscape data models to contracts. This is the largest migration.

**Files:**
- Create: `src/elspeth/contracts/audit.py`
- Modify: `src/elspeth/contracts/__init__.py`
- Modify: `src/elspeth/core/landscape/models.py` (remove all, import from contracts)
- Modify: `src/elspeth/core/landscape/lineage.py` (update imports)
- Modify: `src/elspeth/tui/types.py` (remove types, import from contracts)
- Create: `tests/contracts/test_audit.py`

### Step 1: Create audit.py

This file will contain:
- `Run`, `Node`, `Edge`, `Row`, `Token`
- `NodeStateOpen`, `NodeStateCompleted`, `NodeStateFailed`, `NodeState`
- `RoutingEvent`, `Call`, `Batch`, `BatchMember`, `Artifact`
- `LineageResult`
- `LineageData`, `SourceInfo`, `NodeInfo`, `TokenInfo` (from tui/types.py)
- `NodeStateDisplay` (NEW)

### Steps 2-7: Follow standard pattern

Due to size, this task may need to be split into subtasks:
- 6a: Core models (Run, Node, Edge, Row, Token)
- 6b: NodeState variants
- 6c: Events and calls (RoutingEvent, Call, Batch, Artifact)
- 6d: Lineage types (LineageResult, LineageData, NodeStateDisplay)

---

## Task 7: Create data.py with schema base

**Context:** Move PluginSchema base class.

**Files:**
- Create: `src/elspeth/contracts/data.py`
- Modify: `src/elspeth/contracts/__init__.py`
- Modify: `src/elspeth/plugins/schemas.py` (import from contracts)
- Create: `tests/contracts/test_data.py`

### Step 1: Create data.py

Create `src/elspeth/contracts/data.py`:

```python
"""Row data shape definitions.

These types define what shape row data takes as it flows through the pipeline.
"""

from pydantic import BaseModel, ConfigDict


class PluginSchema(BaseModel):
    """Base class for plugin input/output schemas.

    Plugins declare their expected row shape by subclassing this.
    Validation happens at plugin boundaries.
    """

    model_config = ConfigDict(
        extra="forbid",  # Reject unknown fields by default
        frozen=True,
    )
```

### Steps 2-7: Follow standard pattern

---

## Task 8: Create enforcement script

**Context:** AST-based checker to enforce contracts boundaries.

**Files:**
- Create: `scripts/check_contracts.py`
- Create: `.contracts-whitelist.yaml`
- Modify: `pyproject.toml` (add script entry point)

### Step 1: Create whitelist file

Create `.contracts-whitelist.yaml`:

```yaml
# Types that are explicitly allowed outside contracts/
# Each entry needs a reason

whitelist:
  # Service classes (have methods, not pure data)
  - path: "elspeth/core/landscape/database.py"
    types: ["LandscapeDB"]
    reason: "Service class with connection management"

  - path: "elspeth/core/landscape/recorder.py"
    types: ["LandscapeRecorder"]
    reason: "Service class with database operations"

  - path: "elspeth/core/payload_store.py"
    types: ["PayloadStore", "FilesystemPayloadStore"]
    reason: "Protocol and implementation class"

  - path: "elspeth/engine/orchestrator.py"
    types: ["Orchestrator", "PipelineConfig"]
    reason: "Service class and its config"

  - path: "elspeth/engine/processor.py"
    types: ["RowProcessor"]
    reason: "Service class"

  - path: "elspeth/engine/retry.py"
    types: ["RetryManager", "RetryState"]
    reason: "Internal retry orchestration"

  - path: "elspeth/engine/spans.py"
    types: ["SpanFactory", "NoOpSpan"]
    reason: "Tracing helpers wrapping external lib"

  # Plugin base classes (for inheritance, not data)
  - path: "elspeth/plugins/base.py"
    types: ["BaseTransform", "BaseGate", "BaseAggregation", "BaseSink", "BaseSource"]
    reason: "Implementation base classes"

  - path: "elspeth/plugins/manager.py"
    types: ["PluginManager"]
    reason: "Service class"

  - path: "elspeth/plugins/context.py"
    types: ["PluginContext"]
    reason: "Runtime context, not serialized data"

  - path: "elspeth/plugins/sentinels.py"
    types: ["_MissingSentinel", "MISSING"]
    reason: "Internal sentinel value"

  # DAG internals
  - path: "elspeth/core/dag.py"
    types: ["ExecutionGraph", "NodeInfo"]
    reason: "Service class with NetworkX internals"

  # TUI components (wrapping external Textual lib)
  - path: "elspeth/tui/**"
    types: ["*"]
    reason: "UI components wrapping Textual"

  # Config classes (Pydantic models, could move to contracts later)
  - path: "elspeth/core/config.py"
    types: ["*Settings", "ElspethSettings"]
    reason: "Pydantic config models - consider moving to contracts"
```

### Step 2: Create check script

Create `scripts/check_contracts.py`:

```python
#!/usr/bin/env python3
"""AST-based checker for contracts enforcement.

Scans the codebase for dataclasses, enums, TypedDicts, and NamedTuples
that are not in contracts/ and not in the whitelist.

Usage:
    python scripts/check_contracts.py
    python scripts/check_contracts.py --verbose
"""

import ast
import sys
from pathlib import Path

import yaml


def load_whitelist(path: Path) -> dict:
    """Load whitelist from YAML file."""
    if not path.exists():
        return {"whitelist": []}
    with open(path) as f:
        return yaml.safe_load(f)


def is_whitelisted(filepath: str, typename: str, whitelist: list) -> bool:
    """Check if a type is whitelisted."""
    for entry in whitelist:
        pattern = entry["path"]
        types = entry["types"]

        # Handle glob patterns
        if "**" in pattern:
            base = pattern.replace("/**", "").replace("**", "")
            if filepath.startswith(base):
                if "*" in types or typename in types:
                    return True
        elif filepath.endswith(pattern.replace("elspeth/", "")):
            if "*" in types or typename in types:
                return True
            # Handle pattern types like "*Settings"
            for t in types:
                if t.startswith("*") and typename.endswith(t[1:]):
                    return True

    return False


def find_type_definitions(filepath: Path) -> list[tuple[str, int, str]]:
    """Find dataclass, enum, TypedDict, NamedTuple definitions in a file.

    Returns list of (typename, lineno, kind).
    """
    try:
        with open(filepath) as f:
            tree = ast.parse(f.read(), filename=str(filepath))
    except SyntaxError:
        return []

    definitions = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            # Check for dataclass decorator
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Name) and decorator.id == "dataclass":
                    definitions.append((node.name, node.lineno, "dataclass"))
                    break
                if isinstance(decorator, ast.Call):
                    if isinstance(decorator.func, ast.Name) and decorator.func.id == "dataclass":
                        definitions.append((node.name, node.lineno, "dataclass"))
                        break

            # Check for Enum base class
            for base in node.bases:
                if isinstance(base, ast.Name) and base.id == "Enum":
                    definitions.append((node.name, node.lineno, "enum"))
                    break
                if isinstance(base, ast.Attribute) and base.attr == "Enum":
                    definitions.append((node.name, node.lineno, "enum"))
                    break

            # Check for TypedDict
            for base in node.bases:
                if isinstance(base, ast.Name) and base.id == "TypedDict":
                    definitions.append((node.name, node.lineno, "typeddict"))
                    break

    return definitions


def main() -> int:
    """Run the contracts checker."""
    verbose = "--verbose" in sys.argv or "-v" in sys.argv

    root = Path("src/elspeth")
    contracts_dir = root / "contracts"
    whitelist_path = Path(".contracts-whitelist.yaml")

    whitelist_data = load_whitelist(whitelist_path)
    whitelist = whitelist_data.get("whitelist", [])

    violations = []

    for filepath in root.rglob("*.py"):
        # Skip contracts directory itself
        if contracts_dir in filepath.parents or filepath.parent == contracts_dir:
            continue

        # Skip test files
        if "test" in filepath.name:
            continue

        relative = str(filepath.relative_to(root.parent))

        definitions = find_type_definitions(filepath)

        for typename, lineno, kind in definitions:
            if not is_whitelisted(relative, typename, whitelist):
                violations.append({
                    "file": relative,
                    "line": lineno,
                    "type": typename,
                    "kind": kind,
                })

    if violations:
        print(f"Found {len(violations)} contract violations:\n")
        for v in violations:
            print(f"  {v['file']}:{v['line']}: {v['kind']} `{v['type']}` not in contracts or whitelist")
        print(f"\nTo fix: Move to contracts/ or add to .contracts-whitelist.yaml with reason")
        return 1
    else:
        if verbose:
            print("No contract violations found.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
```

### Step 3: Test the script

Run: `python scripts/check_contracts.py --verbose`
Expected: Lists any violations (will have many before migration is complete)

### Step 4: Add to CI

Add to `.github/workflows/ci.yml` or equivalent:

```yaml
- name: Check contracts
  run: python scripts/check_contracts.py
```

### Step 5: Commit

```bash
git add scripts/check_contracts.py .contracts-whitelist.yaml
git commit -m "feat(contracts): add AST-based enforcement script

- check_contracts.py scans for types outside contracts/
- .contracts-whitelist.yaml for explicit exceptions with reasons
- Intended to run in CI to prevent drift"
```

---

## Summary

| Task | Description | Estimated Effort |
|------|-------------|------------------|
| 1 | Create contracts package with enums.py | 1-2 hours |
| 2 | Create identity.py with TokenInfo | 30 min |
| 3 | Create routing.py with flow control types | 1 hour |
| 4 | Create results.py with operation outcomes | 1-2 hours |
| 5 | Create config.py with configuration contracts | 1 hour |
| 6 | Create audit.py with Landscape models | 2-3 hours |
| 7 | Create data.py with schema base | 30 min |
| 8 | Create enforcement script | 1 hour |

**Total estimated:** 8-11 hours

**Key principles:**
- Bottom-up migration by dependency order
- Clean breaks (no legacy re-exports)
- AST enforcement from day one
- Whitelist requires explicit reason
