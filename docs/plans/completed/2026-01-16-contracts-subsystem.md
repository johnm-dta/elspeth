# Contracts Subsystem Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a centralized `contracts/` package containing all cross-boundary data types, with AST-based enforcement to prevent drift.

**Architecture:** All dataclasses, enums, TypedDicts, and NamedTuples that cross subsystem boundaries live in `src/elspeth/contracts/`. Internal types are explicitly whitelisted. An AST checker script enforces this at CI time. Migration is bottom-up by dependency order with clean breaks (no re-exports, no compatibility shims).

**Tech Stack:** Python 3.11+, ast module, dataclasses, Pydantic, PyYAML (for whitelist)

**Dependencies:**
- None (this is foundational infrastructure)

**Review Notes (2026-01-16):**
This plan was reviewed against the architecture documentation and CLAUDE.md principles. Key corrections applied:

1. **Determinism**: 6 values from architecture + SEEDED enhancement. NO "unknown" value - undeclared determinism crashes at registration per "I don't know what happened" prohibition.
2. **RoutingAction**: MUST include `mode` field for move/copy semantics (required for DAG execution).
3. **TransformResult/GateResult**: Keep `Literal["success", "error"]` (not enum). Keep audit fields (`input_hash`, `output_hash`, `duration_ms`).
4. **ArtifactDescriptor**: Match architecture schema exactly - `artifact_type` (not `kind`), `content_hash` REQUIRED, `size_bytes` REQUIRED.
5. **RowOutcome**: Plain `Enum` (not `str, Enum`) - these are derived at query time, never stored/serialized.
6. **TokenInfo collision**: Rename TUI `TokenInfo` to `TokenDisplayInfo` before migration.
7. **No legacy re-exports**: Per CLAUDE.md "No Legacy Code Policy" - update all import sites directly.
8. **Strict contracts, boundary conversion**: `Edge.default_mode` and `RoutingSpec.mode` use `RoutingMode` enum (not `str` or `Literal`). No `__post_init__` coercion - conversion happens in repository/deserializer layers.

---

## Package Structure

```
src/elspeth/contracts/
├── __init__.py      # Re-exports everything
├── enums.py         # Status codes, modes, kinds
├── identity.py      # TokenInfo, entity identifiers
├── results.py       # TransformResult, GateResult, RowResult, ArtifactDescriptor
├── routing.py       # RoutingAction, RoutingSpec, EdgeInfo
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

## Architectural Pattern: Strict Contracts, Boundary Conversion

**Principle:** Dataclasses are contracts with maximum type enforcement. No coercion, no `isinstance()`.

**The database IS the audit trail.** This is not a cache, not a work queue, not ephemeral state. This is the legally-defensible record of every decision. Data for orchestration comes exclusively through sinks - the database is purely for audit.

**Bad data in the audit database = catastrophic failure:**
- Writing bad data = audit integrity compromised = legal liability
- Reading bad data = either we corrupted it OR someone tampered = incident response

**Database is a seam, not a trust boundary.** We control what goes in and comes out. If bad data exists, that's a "crash right now" situation - not something to silently coerce and hope nobody notices.

**Wrong approach:** `__post_init__` coercion in dataclasses.
```python
# DON'T DO THIS - violates CLAUDE.md prohibition on isinstance()
def __post_init__(self) -> None:
    if not isinstance(self.mode, RoutingMode):  # Defensive programming!
        object.__setattr__(self, "mode", RoutingMode(self.mode))
```

**Right approach:** Conversion happens at explicit boundary layers.

```python
# Dataclass is a strict contract - no coercion
@dataclass(frozen=True)
class Edge:
    default_mode: RoutingMode  # Must be enum, not string

# Repository layer handles DB conversion
class EdgeRepository:
    def load(self, row: Row) -> Edge:
        return Edge(
            edge_id=row.edge_id,
            default_mode=RoutingMode(row.default_mode),  # Convert HERE
            ...
        )

# JSON deserializer handles export conversion
def edge_from_export(data: dict[str, Any]) -> Edge:
    return Edge(
        edge_id=data["edge_id"],
        default_mode=RoutingMode(data["default_mode"]),  # Convert HERE
        ...
    )

# Tests use proper types - no shortcuts
edge = Edge(default_mode=RoutingMode.MOVE)  # Not "move"
```

**Why this is better for auditability:**
1. **Traceable conversions:** You know exactly where string→enum happens
2. **No hidden magic:** Dataclass does what the type says, nothing more
3. **Fail-fast at source:** Bad data in DB crashes at load time, not later
4. **Test discipline:** Tests use real types, catching type bugs early

**Where conversion happens:**
| Source | Conversion Layer |
|--------|------------------|
| SQLAlchemy queries | Repository pattern (e.g., `EdgeRepository.load()`) |
| JSON audit exports | Explicit deserializer (e.g., `edge_from_export()`) |
| YAML test fixtures | Use proper types directly |
| Config files | Pydantic validators (legitimate trust boundary) |

**The only legitimate trust boundaries** (per CLAUDE.md):
- External API responses (LLM providers, HTTP endpoints)
- Plugin schema contracts (external code meets framework)
- Configuration validation (user-provided YAML)
- **Sources and Sinks** (see below)

**Sources and Sinks ARE trust boundaries - the opposite of the audit database:**

| Layer | Data Origin | Trust Level | Error Handling |
|-------|-------------|-------------|----------------|
| **Audit DB (Landscape)** | We wrote it | Full trust | Bad data = CRASH (integrity failure) |
| **Sources** | User-provided | Zero trust | Bad data = validate, quarantine, or reject |
| **Sinks** | We produce for user | N/A (outbound) | Ensure we write valid output |

Sources are where external garbage enters the system:
- CSV with malformed rows
- API returning unexpected JSON
- Database with NULL where you expected values

**We manage sources, we own the audit database.** The source plugin validates/normalizes user data at ingestion. By the time it reaches the audit trail, it's OUR data - clean, validated, hashed.

```
User Data (garbage) → Source Plugin (trust boundary, validate) → Audit DB (our data, strict)
                                                                      ↓
                                              Sink Plugin (our data) → User Output
```

The audit database is NOT a trust boundary - it's our code talking to our code.

---

## Task 1: Create contracts package with enums.py

**Context:** Consolidate all enums that cross boundaries into one file. This is the foundation everything else depends on.

**Files:**
- Create: `src/elspeth/contracts/__init__.py`
- Create: `src/elspeth/contracts/enums.py`
- Modify: `src/elspeth/plugins/enums.py` (delete migrated enums, update all import sites)
- Modify: `src/elspeth/core/landscape/models.py` (update imports)
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
]
```

### Step 2: Create enums.py with all enums

Create `src/elspeth/contracts/enums.py`:

```python
"""All status codes, modes, and kinds used across subsystem boundaries.

CRITICAL: Every plugin MUST declare a Determinism value at registration.
There is no "unknown" - undeclared determinism crashes at registration time.
This is per ELSPETH's principle: "I don't know what happened" is never acceptable.
"""

from enum import Enum


class RunStatus(str, Enum):
    """Status of a pipeline run.

    Uses (str, Enum) because this IS stored in the database (runs.status).
    """

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class NodeStateStatus(str, Enum):
    """Status of a node processing a token.

    Uses (str, Enum) for database serialization to node_states.status.
    """

    OPEN = "open"
    COMPLETED = "completed"
    FAILED = "failed"


class ExportStatus(str, Enum):
    """Status of run export operation.

    Uses (str, Enum) for database serialization.
    """

    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


class BatchStatus(str, Enum):
    """Status of an aggregation batch.

    Uses (str, Enum) for database serialization to batches.status.
    """

    DRAFT = "draft"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"


class NodeType(str, Enum):
    """Type of node in the execution graph.

    Uses (str, Enum) for database serialization to nodes.node_type.
    """

    SOURCE = "source"
    TRANSFORM = "transform"
    GATE = "gate"
    AGGREGATION = "aggregation"
    COALESCE = "coalesce"
    SINK = "sink"


class Determinism(str, Enum):
    """Plugin determinism classification for reproducibility.

    Every plugin MUST declare one of these at registration. No default.
    Undeclared determinism = crash at registration time.

    Each value tells you what to do for replay/verify:
    - DETERMINISTIC: Just re-run, expect identical output
    - SEEDED: Capture seed, replay with same seed
    - IO_READ: Capture what was read (time, files, env)
    - IO_WRITE: Be careful - has side effects on replay
    - EXTERNAL_CALL: Record request/response for replay
    - NON_DETERMINISTIC: Must record output, cannot reproduce

    Uses (str, Enum) for database serialization to nodes.determinism.
    """

    DETERMINISTIC = "deterministic"
    SEEDED = "seeded"
    IO_READ = "io_read"
    IO_WRITE = "io_write"
    EXTERNAL_CALL = "external_call"
    NON_DETERMINISTIC = "non_deterministic"


class RoutingKind(str, Enum):
    """Kind of routing action from a gate.

    Uses (str, Enum) for serialization in routing_events.
    """

    CONTINUE = "continue"
    ROUTE = "route"
    FORK_TO_PATHS = "fork_to_paths"


class RoutingMode(str, Enum):
    """Mode for routing edges.

    MOVE: Token exits current path, goes to destination only
    COPY: Token clones to destination AND continues on current path

    Uses (str, Enum) for database serialization.
    """

    MOVE = "move"
    COPY = "copy"


class RowOutcome(Enum):
    """Terminal outcome for a token in the pipeline.

    IMPORTANT: These are DERIVED at query time from node_states,
    routing_events, and batch_members - NOT stored in the database.
    Therefore this is plain Enum, not (str, Enum).

    If you need the string value, use .value explicitly.
    """

    COMPLETED = "completed"
    ROUTED = "routed"
    FORKED = "forked"
    FAILED = "failed"
    QUARANTINED = "quarantined"
    CONSUMED_IN_BATCH = "consumed_in_batch"
    COALESCED = "coalesced"


class CallType(str, Enum):
    """Type of external call (Phase 6).

    Uses (str, Enum) for database serialization to calls.call_type.
    """

    LLM = "llm"
    HTTP = "http"
    SQL = "sql"
    FILESYSTEM = "filesystem"


class CallStatus(str, Enum):
    """Status of an external call (Phase 6).

    Uses (str, Enum) for database serialization to calls.status.
    """

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


class TestDeterminism:
    """Tests for Determinism enum - critical for replay/verify."""

    def test_has_all_required_values(self) -> None:
        """Determinism has all 6 values from architecture."""
        from elspeth.contracts import Determinism

        assert hasattr(Determinism, "DETERMINISTIC")
        assert hasattr(Determinism, "SEEDED")
        assert hasattr(Determinism, "IO_READ")
        assert hasattr(Determinism, "IO_WRITE")
        assert hasattr(Determinism, "EXTERNAL_CALL")
        assert hasattr(Determinism, "NON_DETERMINISTIC")

    def test_no_unknown_value(self) -> None:
        """Determinism must NOT have 'unknown' - we crash instead."""
        from elspeth.contracts import Determinism

        values = [d.value for d in Determinism]
        assert "unknown" not in values

    def test_string_values_match_architecture(self) -> None:
        """String values match architecture specification."""
        from elspeth.contracts import Determinism

        assert Determinism.DETERMINISTIC.value == "deterministic"
        assert Determinism.SEEDED.value == "seeded"
        assert Determinism.IO_READ.value == "io_read"
        assert Determinism.IO_WRITE.value == "io_write"
        assert Determinism.EXTERNAL_CALL.value == "external_call"
        assert Determinism.NON_DETERMINISTIC.value == "non_deterministic"


class TestRowOutcome:
    """Tests for RowOutcome enum - derived, not stored."""

    def test_is_not_str_enum(self) -> None:
        """RowOutcome should NOT be a str subclass - it's derived, not stored."""
        from elspeth.contracts import RowOutcome

        # RowOutcome.COMPLETED should not be equal to string without .value
        assert RowOutcome.COMPLETED != "completed"
        assert RowOutcome.COMPLETED.value == "completed"

    def test_has_all_terminal_states(self) -> None:
        """RowOutcome has all terminal states from architecture."""
        from elspeth.contracts import RowOutcome

        assert hasattr(RowOutcome, "COMPLETED")
        assert hasattr(RowOutcome, "ROUTED")
        assert hasattr(RowOutcome, "FORKED")
        assert hasattr(RowOutcome, "FAILED")
        assert hasattr(RowOutcome, "QUARANTINED")
        assert hasattr(RowOutcome, "CONSUMED_IN_BATCH")
        assert hasattr(RowOutcome, "COALESCED")


class TestRoutingMode:
    """Tests for RoutingMode enum."""

    def test_routing_mode_values(self) -> None:
        """RoutingMode has move and copy."""
        from elspeth.contracts import RoutingMode

        assert RoutingMode.MOVE.value == "move"
        assert RoutingMode.COPY.value == "copy"


class TestEnumCoercion:
    """Verify enums that ARE stored can be created from string values."""

    def test_run_status_from_string(self) -> None:
        """Can create RunStatus from string (for DB reads)."""
        from elspeth.contracts import RunStatus

        assert RunStatus("running") == RunStatus.RUNNING
        assert RunStatus("completed") == RunStatus.COMPLETED

    def test_invalid_value_raises(self) -> None:
        """Invalid string raises ValueError - no silent fallback."""
        from elspeth.contracts import RunStatus

        with pytest.raises(ValueError):
            RunStatus("invalid")
```

### Step 4: Run tests

Run: `pytest tests/contracts/test_enums.py -v`
Expected: PASS

### Step 5: Update all import sites (NO RE-EXPORTS)

**IMPORTANT:** Per CLAUDE.md "No Legacy Code Policy", we do NOT create re-exports.
Delete types from source files and update ALL import sites directly.

Find all files importing from `plugins/enums.py`:
```bash
grep -r "from elspeth.plugins.enums" src/
```

Update each to import from `elspeth.contracts`:
- `src/elspeth/plugins/base.py`
- `src/elspeth/plugins/results.py`
- `src/elspeth/engine/executors.py`
- `src/elspeth/core/landscape/models.py`
- (any others found)

Then delete the migrated types from `src/elspeth/plugins/enums.py` (or delete the file entirely if empty).

### Step 6: Run full test suite

Run: `pytest tests/ -v --tb=short`
Expected: PASS (no regressions)

### Step 7: Commit

```bash
git add src/elspeth/contracts/ tests/contracts/
git add -u  # Stage all modified files
git commit -m "feat(contracts): create contracts package with enums

- Add contracts/ package as single source of truth for cross-boundary types
- Determinism has 6 values per architecture (DETERMINISTIC, SEEDED, IO_READ, IO_WRITE, EXTERNAL_CALL, NON_DETERMINISTIC) - no unknown
- RowOutcome is plain Enum (not str,Enum) - derived values never stored
- Update all import sites directly (no legacy re-exports per CLAUDE.md)"
```

---

## Task 2: Create identity.py with TokenInfo

**Context:** Move token identity structures to contracts. Resolve naming collision with TUI type.

**Files:**
- Create: `src/elspeth/contracts/identity.py`
- Modify: `src/elspeth/contracts/__init__.py` (add exports)
- Modify: `src/elspeth/engine/tokens.py` (remove TokenInfo, import from contracts)
- Modify: `src/elspeth/tui/types.py` (rename TokenInfo to TokenDisplayInfo)
- Create: `tests/contracts/test_identity.py`

### Step 1: Rename TUI TokenInfo FIRST (resolve collision)

Modify `src/elspeth/tui/types.py`:

```python
# BEFORE:
class TokenInfo(TypedDict):
    ...

# AFTER:
class TokenDisplayInfo(TypedDict):
    """Token information formatted for TUI display.

    Note: This is a DISPLAY type, not the canonical TokenInfo from contracts.
    It contains presentation-specific fields like 'path' for breadcrumb display.
    """
    token_id: str
    row_id: str
    path: list[str]
    ...
```

Update all references to `TokenInfo` in `src/elspeth/tui/` files to use `TokenDisplayInfo`.

### Step 2: Create identity.py

Create `src/elspeth/contracts/identity.py`:

```python
"""Entity identifiers and token structures.

These types answer: "How do we refer to things?"
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class TokenInfo:
    """Identity and data for a token flowing through the DAG.

    Tokens track row instances through forks/joins:
    - row_id: Stable source row identity
    - token_id: Instance of row in a specific DAG path
    - branch_name: Which fork path this token is on (if forked)

    Note: NOT frozen because row_data is mutable dict and executors
    update tokens as they flow through the pipeline.
    """

    row_id: str
    token_id: str
    row_data: dict[str, Any]
    branch_name: str | None = None
```

### Step 3: Update contracts __init__.py

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

### Step 4: Write test

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

    def test_token_info_row_data_mutable(self) -> None:
        """TokenInfo.row_data can be modified (not frozen)."""
        from elspeth.contracts import TokenInfo

        token = TokenInfo(row_id="r", token_id="t", row_data={"a": 1})
        token.row_data["b"] = 2

        assert token.row_data == {"a": 1, "b": 2}
```

### Step 5: Run tests

Run: `pytest tests/contracts/test_identity.py -v`
Expected: PASS

### Step 6: Update engine/tokens.py

Remove TokenInfo definition from `src/elspeth/engine/tokens.py`, import from contracts:

```python
from elspeth.contracts import TokenInfo

# TokenInfo is now imported, not defined here
```

### Step 7: Run full test suite

Run: `pytest tests/ -v --tb=short`
Expected: PASS

### Step 8: Commit

```bash
git add src/elspeth/contracts/identity.py src/elspeth/contracts/__init__.py
git add tests/contracts/test_identity.py
git add src/elspeth/engine/tokens.py src/elspeth/tui/types.py
git commit -m "feat(contracts): add identity.py with TokenInfo

- Move TokenInfo from engine/tokens.py to contracts/identity.py
- Rename TUI TokenInfo to TokenDisplayInfo (resolves naming collision)
- TokenInfo is NOT frozen (row_data mutates during processing)"
```

---

## Task 3: Create routing.py with routing contracts

**Context:** Consolidate routing action and edge types. CRITICAL: RoutingAction MUST include `mode` field.

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
from typing import Any, Mapping

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

    CRITICAL: The `mode` field determines move vs copy semantics:
    - MOVE: Token exits current path, goes to destination only
    - COPY: Token clones to destination AND continues on current path

    This field is REQUIRED per architecture. Without it, executors cannot
    correctly record routing events or determine token flow.
    """

    kind: RoutingKind
    destinations: tuple[str, ...]
    mode: RoutingMode
    reason: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    @classmethod
    def continue_(cls, *, reason: dict[str, Any] | None = None) -> "RoutingAction":
        """Continue to next node in pipeline."""
        return cls(
            kind=RoutingKind.CONTINUE,
            destinations=(),
            mode=RoutingMode.MOVE,  # Default for continue
            reason=_freeze_dict(reason),
        )

    @classmethod
    def route(
        cls,
        label: str,
        *,
        mode: RoutingMode = RoutingMode.MOVE,
        reason: dict[str, Any] | None = None,
    ) -> "RoutingAction":
        """Route to a specific labeled destination.

        Args:
            label: Semantic route label (resolved via plugin's routes config)
            mode: MOVE (exit pipeline) or COPY (also continue)
            reason: Classification metadata for audit trail
        """
        return cls(
            kind=RoutingKind.ROUTE,
            destinations=(label,),
            mode=mode,
            reason=_freeze_dict(reason),
        )

    @classmethod
    def fork_to_paths(
        cls,
        paths: list[str],
        *,
        reason: dict[str, Any] | None = None,
    ) -> "RoutingAction":
        """Fork token to multiple parallel paths (always copy mode)."""
        return cls(
            kind=RoutingKind.FORK_TO_PATHS,
            destinations=tuple(paths),
            mode=RoutingMode.COPY,  # Fork always copies
            reason=_freeze_dict(reason),
        )


@dataclass(frozen=True)
class RoutingSpec:
    """Specification for a routing edge in the recorded audit trail.

    Strict contract - mode MUST be RoutingMode enum, not string.
    Conversion from DB strings happens in repository layer.
    """

    edge_id: str
    mode: RoutingMode


@dataclass(frozen=True)
class EdgeInfo:
    """Information about an edge in the execution graph.

    Replaces tuple[str, str, dict[str, Any]] for type safety.
    Strict contract - mode MUST be RoutingMode enum.
    """

    from_node: str
    to_node: str
    label: str
    mode: RoutingMode
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

    def test_has_mode_field(self) -> None:
        """RoutingAction MUST have mode field - required for DAG execution."""
        from elspeth.contracts import RoutingAction, RoutingMode

        action = RoutingAction.continue_()

        assert hasattr(action, "mode")
        assert action.mode == RoutingMode.MOVE

    def test_continue_action(self) -> None:
        """Can create continue action."""
        from elspeth.contracts import RoutingAction, RoutingKind, RoutingMode

        action = RoutingAction.continue_(reason={"rule": "default"})

        assert action.kind == RoutingKind.CONTINUE
        assert action.destinations == ()
        assert action.mode == RoutingMode.MOVE
        assert action.reason["rule"] == "default"

    def test_route_action_default_move(self) -> None:
        """Route action defaults to MOVE mode."""
        from elspeth.contracts import RoutingAction, RoutingKind, RoutingMode

        action = RoutingAction.route("quarantine", reason={"score": 0.1})

        assert action.kind == RoutingKind.ROUTE
        assert action.destinations == ("quarantine",)
        assert action.mode == RoutingMode.MOVE

    def test_route_action_with_copy(self) -> None:
        """Route action can use COPY mode."""
        from elspeth.contracts import RoutingAction, RoutingMode

        action = RoutingAction.route("archive", mode=RoutingMode.COPY)

        assert action.mode == RoutingMode.COPY

    def test_fork_action_always_copy(self) -> None:
        """Fork action always uses COPY mode."""
        from elspeth.contracts import RoutingAction, RoutingKind, RoutingMode

        action = RoutingAction.fork_to_paths(["sentiment", "classification"])

        assert action.kind == RoutingKind.FORK_TO_PATHS
        assert action.destinations == ("sentiment", "classification")
        assert action.mode == RoutingMode.COPY

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

    def test_correct_usage_with_enum(self) -> None:
        """Document correct usage - repository layer converts before construction.

        Per Data Manifesto: dataclasses are strict contracts. Type enforcement
        happens at static analysis time (mypy), not runtime. The repository
        layer must convert: RoutingSpec(edge_id=row.edge_id, mode=RoutingMode(row.mode))

        Passing a string instead of RoutingMode is a type error caught by mypy,
        not a runtime exception. This test documents the correct pattern.
        """
        from elspeth.contracts import RoutingSpec, RoutingMode

        # Correct: repository converts DB string to enum before constructing
        spec = RoutingSpec(edge_id="edge-1", mode=RoutingMode.COPY)
        assert spec.mode == RoutingMode.COPY
        assert isinstance(spec.mode, RoutingMode)

        # WRONG (but not enforced at runtime - caught by mypy):
        # spec = RoutingSpec(edge_id="edge-1", mode="copy")  # type: ignore
        # ↑ This would pass at runtime but fail mypy


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
        assert edge.label == "continue"
        assert edge.mode == RoutingMode.MOVE
```

### Step 4: Run tests

Run: `pytest tests/contracts/test_routing.py -v`
Expected: PASS

### Step 5: Update source files to import from contracts

Update imports in:
- `src/elspeth/plugins/results.py` - remove RoutingAction, import from contracts
- `src/elspeth/core/landscape/models.py` - remove RoutingSpec, import from contracts
- `src/elspeth/engine/executors.py` - update imports

### Step 6: Run full test suite

Run: `pytest tests/ -v --tb=short`
Expected: PASS

### Step 7: Commit

```bash
git add src/elspeth/contracts/routing.py src/elspeth/contracts/__init__.py
git add tests/contracts/test_routing.py
git add src/elspeth/plugins/results.py src/elspeth/core/landscape/models.py
git commit -m "feat(contracts): add routing.py with flow control types

- Move RoutingAction from plugins/results.py (INCLUDES mode field)
- Move RoutingSpec from landscape/models.py
- Add EdgeInfo to replace tuple[str, str, dict]
- mode field is REQUIRED for move/copy routing semantics"
```

---

## Task 4: Create results.py with operation outcomes

**Context:** Consolidate transform, gate, and row result types.

**CRITICAL:**
- TransformResult.status uses `Literal["success", "error"]` (NOT enum)
- TransformResult and GateResult KEEP audit fields
- ArtifactDescriptor matches architecture schema exactly

**Files:**
- Create: `src/elspeth/contracts/results.py`
- Modify: `src/elspeth/contracts/__init__.py`
- Modify: `src/elspeth/plugins/results.py` (remove migrated types)
- Modify: `src/elspeth/engine/processor.py` (remove RowResult)
- Modify: `src/elspeth/engine/artifacts.py` (remove ArtifactDescriptor)
- Create: `tests/contracts/test_results.py`

### Step 1: Create results.py

Create `src/elspeth/contracts/results.py`:

```python
"""Operation outcomes and results.

These types answer: "What did an operation produce?"

IMPORTANT:
- TransformResult.status uses Literal["success", "error"], NOT an enum
- TransformResult and GateResult KEEP audit fields (input_hash, output_hash, duration_ms)
- ArtifactDescriptor matches architecture schema (artifact_type, content_hash REQUIRED, size_bytes REQUIRED)
"""

from dataclasses import dataclass, field
from typing import Any, Literal

from elspeth.contracts.enums import RowOutcome
from elspeth.contracts.routing import RoutingAction
from elspeth.contracts.identity import TokenInfo


@dataclass
class TransformResult:
    """Result of a transform operation.

    Use the factory methods to create instances.

    IMPORTANT: status uses Literal["success", "error"], NOT enum, per architecture.
    Audit fields (input_hash, output_hash, duration_ms) are populated by executors.
    """

    status: Literal["success", "error"]
    row: dict[str, Any] | None
    reason: dict[str, Any] | None
    retryable: bool = False

    # Audit fields - set by executor, not by plugin
    input_hash: str | None = field(default=None, repr=False)
    output_hash: str | None = field(default=None, repr=False)
    duration_ms: float | None = field(default=None, repr=False)

    @classmethod
    def success(cls, row: dict[str, Any]) -> "TransformResult":
        """Create successful result with output row."""
        return cls(status="success", row=row, reason=None)

    @classmethod
    def error(
        cls,
        reason: dict[str, Any],
        *,
        retryable: bool = False,
    ) -> "TransformResult":
        """Create error result with reason."""
        return cls(
            status="error",
            row=None,
            reason=reason,
            retryable=retryable,
        )


@dataclass
class GateResult:
    """Result of a gate evaluation.

    Contains the (possibly modified) row and routing action.
    Audit fields are populated by GateExecutor, not by plugin.
    """

    row: dict[str, Any]
    action: RoutingAction

    # Audit fields - set by executor, not by plugin
    input_hash: str | None = field(default=None, repr=False)
    output_hash: str | None = field(default=None, repr=False)
    duration_ms: float | None = field(default=None, repr=False)


@dataclass
class AcceptResult:
    """Result of aggregation accept check.

    Indicates whether the row was accepted into a batch.
    """

    accepted: bool
    trigger: bool
    batch_id: str | None = field(default=None, repr=False)


@dataclass
class RowResult:
    """Final result of processing a row through the pipeline.

    Uses RowOutcome enum. The outcome is derived at query time
    from node_states/routing_events/batch_members, but this type
    is used to communicate the result during processing.
    """

    token: TokenInfo
    final_data: dict[str, Any]
    outcome: RowOutcome
    sink_name: str | None = None


@dataclass(frozen=True)
class ArtifactDescriptor:
    """Descriptor for an artifact written by a sink.

    Matches architecture artifacts table schema:
    - artifact_type: NOT NULL (matches DB column name)
    - content_hash: NOT NULL (REQUIRED for audit integrity)
    - size_bytes: NOT NULL (REQUIRED for verification)

    Factory methods provide convenient construction for each artifact type.
    """

    artifact_type: Literal["file", "database", "webhook"]
    path_or_uri: str
    content_hash: str  # REQUIRED - audit integrity
    size_bytes: int    # REQUIRED - verification
    metadata: dict[str, object] | None = None

    @classmethod
    def for_file(
        cls,
        path: str,
        content_hash: str,
        size_bytes: int,
    ) -> "ArtifactDescriptor":
        """Create descriptor for file-based artifacts."""
        return cls(
            artifact_type="file",
            path_or_uri=f"file://{path}",
            content_hash=content_hash,
            size_bytes=size_bytes,
        )

    @classmethod
    def for_database(
        cls,
        url: str,
        table: str,
        content_hash: str,
        payload_size: int,
        row_count: int,
    ) -> "ArtifactDescriptor":
        """Create descriptor for database artifacts."""
        return cls(
            artifact_type="database",
            path_or_uri=f"db://{table}@{url}",
            content_hash=content_hash,
            size_bytes=payload_size,
            metadata={"table": table, "row_count": row_count},
        )

    @classmethod
    def for_webhook(
        cls,
        url: str,
        content_hash: str,
        request_size: int,
        response_code: int,
    ) -> "ArtifactDescriptor":
        """Create descriptor for webhook artifacts."""
        return cls(
            artifact_type="webhook",
            path_or_uri=f"webhook://{url}",
            content_hash=content_hash,
            size_bytes=request_size,
            metadata={"response_code": response_code},
        )
```

### Step 2: Update __init__.py

Add exports for result types.

### Step 3: Write tests

Create `tests/contracts/test_results.py`:

```python
"""Tests for results contracts."""

import pytest


class TestTransformResult:
    """Tests for TransformResult."""

    def test_success_factory(self) -> None:
        """Can create success result."""
        from elspeth.contracts import TransformResult

        result = TransformResult.success({"output": "data"})

        assert result.status == "success"
        assert result.row == {"output": "data"}
        assert result.reason is None

    def test_error_factory(self) -> None:
        """Can create error result."""
        from elspeth.contracts import TransformResult

        result = TransformResult.error(
            {"error": "something went wrong"},
            retryable=True,
        )

        assert result.status == "error"
        assert result.row is None
        assert result.reason == {"error": "something went wrong"}
        assert result.retryable is True

    def test_status_is_literal_not_enum(self) -> None:
        """Status is Literal, not enum - can compare directly to string."""
        from elspeth.contracts import TransformResult

        result = TransformResult.success({"x": 1})

        # This works because status is Literal["success", "error"]
        assert result.status == "success"

    def test_has_audit_fields(self) -> None:
        """TransformResult has audit fields for executor to populate."""
        from elspeth.contracts import TransformResult

        result = TransformResult.success({"x": 1})

        # Audit fields default to None, set by executor
        assert result.input_hash is None
        assert result.output_hash is None
        assert result.duration_ms is None

        # Executor can set them
        result.input_hash = "abc123"
        result.output_hash = "def456"
        result.duration_ms = 42.5

        assert result.input_hash == "abc123"


class TestGateResult:
    """Tests for GateResult."""

    def test_create_gate_result(self) -> None:
        """Can create GateResult with row and action."""
        from elspeth.contracts import GateResult, RoutingAction

        action = RoutingAction.continue_()
        result = GateResult(row={"data": "value"}, action=action)

        assert result.row == {"data": "value"}
        assert result.action == action

    def test_has_audit_fields(self) -> None:
        """GateResult has audit fields."""
        from elspeth.contracts import GateResult, RoutingAction

        result = GateResult(row={}, action=RoutingAction.continue_())

        assert hasattr(result, "input_hash")
        assert hasattr(result, "output_hash")
        assert hasattr(result, "duration_ms")


class TestArtifactDescriptor:
    """Tests for ArtifactDescriptor."""

    def test_content_hash_required(self) -> None:
        """content_hash is required, not optional."""
        from elspeth.contracts import ArtifactDescriptor

        # Must provide content_hash - this should work
        artifact = ArtifactDescriptor(
            artifact_type="file",
            path_or_uri="file:///tmp/out.csv",
            content_hash="abc123",
            size_bytes=1024,
        )

        assert artifact.content_hash == "abc123"

    def test_size_bytes_required(self) -> None:
        """size_bytes is required, not optional."""
        from elspeth.contracts import ArtifactDescriptor

        artifact = ArtifactDescriptor(
            artifact_type="file",
            path_or_uri="file:///tmp/out.csv",
            content_hash="abc123",
            size_bytes=1024,
        )

        assert artifact.size_bytes == 1024

    def test_uses_artifact_type_not_kind(self) -> None:
        """Field is artifact_type (matches DB schema), not 'kind'."""
        from elspeth.contracts import ArtifactDescriptor

        artifact = ArtifactDescriptor.for_file("/tmp/x", "hash", 100)

        assert hasattr(artifact, "artifact_type")
        assert not hasattr(artifact, "kind")

    def test_for_file_factory(self) -> None:
        """for_file creates correct descriptor."""
        from elspeth.contracts import ArtifactDescriptor

        artifact = ArtifactDescriptor.for_file(
            path="/tmp/output.csv",
            content_hash="abc123",
            size_bytes=2048,
        )

        assert artifact.artifact_type == "file"
        assert artifact.path_or_uri == "file:///tmp/output.csv"
        assert artifact.content_hash == "abc123"
        assert artifact.size_bytes == 2048

    def test_for_database_factory(self) -> None:
        """for_database creates correct descriptor."""
        from elspeth.contracts import ArtifactDescriptor

        artifact = ArtifactDescriptor.for_database(
            url="postgresql://localhost/db",
            table="results",
            content_hash="def456",
            payload_size=4096,
            row_count=100,
        )

        assert artifact.artifact_type == "database"
        assert artifact.metadata["table"] == "results"
        assert artifact.metadata["row_count"] == 100

    def test_for_webhook_factory(self) -> None:
        """for_webhook creates correct descriptor."""
        from elspeth.contracts import ArtifactDescriptor

        artifact = ArtifactDescriptor.for_webhook(
            url="https://api.example.com/webhook",
            content_hash="ghi789",
            request_size=512,
            response_code=200,
        )

        assert artifact.artifact_type == "webhook"
        assert artifact.metadata["response_code"] == 200
```

### Step 4: Run tests

Run: `pytest tests/contracts/test_results.py -v`
Expected: PASS

### Step 5: Update source files

Update imports in plugins/results.py, engine/processor.py, engine/executors.py, engine/artifacts.py.

### Step 6: Run full test suite and commit

---

## Task 5-8: Remaining Tasks

Tasks 5-8 follow the same pattern as Tasks 1-4. Key points:

**Task 5 (config.py):** Configuration contracts - ResolvedConfig, PipelineSettings.

**Task 6 (audit.py):** Landscape models - largest migration. Consider splitting into subtasks:
- 6a: Core models (Run, Node, Edge, Row, Token)
  - **Type upgrade:** `Edge.default_mode: str` → `RoutingMode` (strict, no coercion)
  - **Add repository layer:** `EdgeRepository.load()` converts DB strings to enums
- 6b: NodeState variants
- 6c: Events and calls (RoutingEvent, Call, Batch, Artifact)
- 6d: Lineage types

### Task 6a Detail: Edge with Repository Layer

**The strict contract (in `contracts/audit.py`):**

```python
from dataclasses import dataclass
from datetime import datetime
from elspeth.contracts.enums import RoutingMode


@dataclass
class Edge:
    """An edge in the execution graph.

    Strict contract - default_mode MUST be RoutingMode enum.
    Conversion from DB strings happens in EdgeRepository.

    Per Data Manifesto: The audit database is OUR data. If we read
    garbage from it, something catastrophic happened - crash immediately.
    """

    edge_id: str
    run_id: str
    from_node_id: str
    to_node_id: str
    label: str
    default_mode: RoutingMode  # Strict: enum only, not string
    created_at: datetime
```

**The repository layer (in `core/landscape/repositories.py`):**

```python
from sqlalchemy import Row
from elspeth.contracts import Edge, RoutingMode


class EdgeRepository:
    """Repository for Edge records in the audit database.

    Handles the seam between SQLAlchemy rows (strings) and
    domain objects (strict types). This is NOT a trust boundary -
    if the database has bad data, we crash.
    """

    def __init__(self, session):
        self.session = session

    def load(self, row: Row) -> Edge:
        """Load Edge from database row.

        Conversion happens HERE, not in the dataclass.
        If row.default_mode is not a valid RoutingMode, this crashes.
        That's intentional - bad data in audit DB is catastrophic.
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

    def load_many(self, rows: list[Row]) -> list[Edge]:
        """Load multiple Edges from database rows."""
        return [self.load(row) for row in rows]

    def save(self, edge: Edge) -> None:
        """Save Edge to database.

        The enum serializes to string automatically via .value
        """
        self.session.execute(
            edges_table.insert().values(
                edge_id=edge.edge_id,
                run_id=edge.run_id,
                from_node_id=edge.from_node_id,
                to_node_id=edge.to_node_id,
                label=edge.label,
                default_mode=edge.default_mode.value,  # Enum → string
                created_at=edge.created_at,
            )
        )
```

**Test for the repository (in `tests/core/landscape/test_repositories.py`):**

```python
import pytest
from datetime import datetime
from elspeth.contracts import Edge, RoutingMode
from elspeth.core.landscape.repositories import EdgeRepository


class TestEdgeRepository:
    """Tests for EdgeRepository."""

    def test_load_converts_string_to_enum(self, mock_row) -> None:
        """Repository converts DB string to RoutingMode enum."""
        mock_row.default_mode = "move"  # String from DB

        repo = EdgeRepository(session=None)
        edge = repo.load(mock_row)

        assert edge.default_mode == RoutingMode.MOVE
        assert isinstance(edge.default_mode, RoutingMode)

    def test_load_crashes_on_invalid_mode(self, mock_row) -> None:
        """Invalid mode in DB crashes immediately - audit integrity failure."""
        mock_row.default_mode = "garbage"

        repo = EdgeRepository(session=None)

        with pytest.raises(ValueError, match="'garbage' is not a valid RoutingMode"):
            repo.load(mock_row)

    def test_edge_requires_enum_not_string(self) -> None:
        """Document that Edge requires RoutingMode enum, not string.

        Type enforcement is at mypy time, not runtime. This test shows
        correct usage - the repository layer must convert before construction.
        """
        # Correct: use enum
        edge = Edge(
            edge_id="edge-1",
            run_id="run-1",
            from_node_id="a",
            to_node_id="b",
            label="continue",
            default_mode=RoutingMode.MOVE,  # Correct: enum
            created_at=datetime.now(),
        )
        assert edge.default_mode == RoutingMode.MOVE

        # WRONG (caught by mypy, not runtime):
        # Edge(..., default_mode="move")  # type: ignore
        # ↑ Passes at runtime but fails static analysis
```

**Pattern applies to all audit types with enum fields:**
- `Node.determinism` → `NodeRepository.load()` converts
- `Node.node_type` → `NodeRepository.load()` converts
- `RoutingEvent.mode` → `RoutingEventRepository.load()` converts

**Task 7 (data.py):** PluginSchema base class.

**Task 8 (enforcement script):** AST-based checker with whitelist.

---

## Summary

| Task | Description | Key Fixes Applied |
|------|-------------|-------------------|
| 1 | Create contracts package with enums.py | Determinism 6 values, RowOutcome plain Enum, no re-exports |
| 2 | Create identity.py with TokenInfo | Rename TUI TokenInfo first, TokenInfo not frozen |
| 3 | Create routing.py | RoutingAction INCLUDES mode field |
| 4 | Create results.py | Literal status, audit fields kept, ArtifactDescriptor matches schema |
| 5 | Create config.py | Configuration contracts |
| 6 | Create audit.py | Landscape models, Edge.default_mode → RoutingMode (strict), add repository layer |
| 7 | Create data.py | Schema base |
| 8 | Create enforcement script | AST checker with whitelist |

**Key principles:**
- Bottom-up migration by dependency order
- Clean breaks (no legacy re-exports per CLAUDE.md)
- AST enforcement from day one
- Whitelist requires explicit reason
- Architecture is source of truth
- Drift acceptable only if it improves/doesn't break the 4 pillars (Accuracy, Auditability, Availability, Security)
