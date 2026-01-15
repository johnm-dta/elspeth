# ELSPETH Integration Audit Report

**Date:** 2026-01-15
**Auditor:** Claude Code (automated)
**Scope:** Full codebase integration analysis
**Status:** Complete

---

## Executive Summary

This audit examined integration patterns across all ELSPETH subsystems to identify:
1. **Soft integration anti-patterns** - loose data structures instead of strict contracts
2. **Positive integration validation** - verifying all integration points connect correctly

### Key Statistics

| Severity | Count | Description |
|----------|-------|-------------|
| Critical | 2 | Must fix before Phase 3 integration |
| High | 5 | Should fix before production |
| Medium | 10 | Quality improvements |
| Low | 8 | Minor inconsistencies |

### Overall Assessment

**Architecture: Strong** - Individual subsystems are well-designed with proper type safety.
**Integration Points: Weak** - Contracts weaken at subsystem boundaries.
**Recommendation:** Address critical issues before Phase 3; high-severity before production.

---

## Critical Issues

### CRIT-001: Ambiguous None Return in get_row_data()

**Location:** `src/elspeth/core/landscape/recorder.py:1472-1497`

**Pattern:** Implicit contract - caller cannot distinguish failure modes

**Current State:**
```python
def get_row_data(self, row_id: str) -> dict[str, Any] | None:
    row = self.get_row(row_id)
    if row is None:
        return None

    if row.source_data_ref and self._payload_store:
        payload_bytes = self._payload_store.retrieve(row.source_data_ref)
        data: dict[str, Any] = cast(dict[str, Any], json.loads(...))
        return data

    return None  # Ambiguous!
```

**Problem:** Returns `None` for four distinct scenarios:
1. Row not found in database
2. Row exists but `source_data_ref` is None (never stored)
3. Row exists but `payload_store` not configured
4. Payload was purged (hash preserved, data deleted)

**Impact:**
- TUI cannot explain *why* data is unavailable
- Violates ELSPETH's promise: "I don't know what happened" becomes possible
- Audit trail viewer shows "N/A" without distinguishing recoverable vs permanent

**Suggested Fix:**
```python
from dataclasses import dataclass
from enum import Enum

class RowDataState(Enum):
    AVAILABLE = "available"
    PURGED = "purged"              # Hash preserved, payload deleted
    NEVER_STORED = "never_stored"  # source_data_ref is None
    STORE_NOT_CONFIGURED = "store_not_configured"
    ROW_NOT_FOUND = "row_not_found"

@dataclass
class RowDataResult:
    state: RowDataState
    data: dict[str, Any] | None

def get_row_data(self, row_id: str) -> RowDataResult:
    row = self.get_row(row_id)
    if row is None:
        return RowDataResult(state=RowDataState.ROW_NOT_FOUND, data=None)

    if row.source_data_ref is None:
        return RowDataResult(state=RowDataState.NEVER_STORED, data=None)

    if self._payload_store is None:
        return RowDataResult(state=RowDataState.STORE_NOT_CONFIGURED, data=None)

    try:
        payload_bytes = self._payload_store.retrieve(row.source_data_ref)
        data = json.loads(payload_bytes.decode("utf-8"))
        return RowDataResult(state=RowDataState.AVAILABLE, data=data)
    except KeyError:
        return RowDataResult(state=RowDataState.PURGED, data=None)
```

**Effort:** 2 hours
**Dependencies:** Update TUI callers to handle RowDataResult

---

### CRIT-002: Plugin Protocol Violation - Undocumented close() Methods

**Location:** All Transform and Gate implementations

**Pattern:** Interface drift - implementations define methods not in protocol

**Affected Files:**
| File | Line | Plugin |
|------|------|--------|
| `plugins/transforms/passthrough.py` | 52 | PassThrough |
| `plugins/transforms/field_mapper.py` | 107 | FieldMapper |
| `plugins/gates/threshold_gate.py` | 141 | ThresholdGate |
| `plugins/gates/field_match_gate.py` | 187 | FieldMatchGate |
| `plugins/gates/filter_gate.py` | 239 | FilterGate |

**Current State:**
```python
# Protocol (protocols.py) does NOT define close()
class TransformProtocol(Protocol):
    def process(self, row: dict, ctx: PluginContext) -> TransformResult: ...
    # No close() method defined

# But ALL implementations define it:
class PassThrough:
    def close(self) -> None:
        """Clean up resources."""
        pass
```

**Problem:**
- Engine **never calls** `close()` on transforms/gates
- Developers add lifecycle methods expecting them to be called
- Resource leaks possible if transforms hold connections/file handles
- `SinkProtocol` correctly defines `close()` - inconsistency

**Options:**
1. **Add to protocol** - Define `close()` in Transform/Gate protocols, have engine call it
2. **Remove from implementations** - Delete the methods with clear documentation that transforms are stateless

**Suggested Fix (Option 1):**
```python
# In protocols.py
class TransformProtocol(Protocol):
    def process(self, row: dict, ctx: PluginContext) -> TransformResult: ...
    def close(self) -> None: ...  # Add lifecycle method

class GateProtocol(Protocol):
    def evaluate(self, row: dict, ctx: PluginContext) -> GateResult: ...
    def close(self) -> None: ...  # Add lifecycle method

# In processor.py - add cleanup
def process_row(self, ...) -> RowResult:
    try:
        # ... existing processing logic ...
    finally:
        for transform in transforms:
            if hasattr(transform, 'close'):
                transform.close()
```

**Effort:** 4 hours
**Dependencies:** Decide on lifecycle model first

---

## High-Severity Issues

### HIGH-001: Stringly-Typed Status Fields in Engine Results

**Location:** `src/elspeth/engine/orchestrator.py:45`, `src/elspeth/engine/processor.py:32`

**Pattern:** String literals where enums exist and should be used

**Current State:**
```python
# orchestrator.py:45
@dataclass
class RunResult:
    status: str  # "completed", "failed"

# processor.py:32
@dataclass
class RowResult:
    outcome: str  # "completed", "routed", "forked", "consumed", "failed"
```

**Problem:**
- Callers must string-match: `if result.outcome == "completed":`
- Typos like `"compleetd"` are silent until runtime
- `RowOutcome` enum exists in `plugins/results.py` but isn't used

**Suggested Fix:**
```python
# orchestrator.py
from enum import Enum

class RunStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class RunResult:
    status: RunStatus

# processor.py
from elspeth.plugins.results import RowOutcome

@dataclass
class RowResult:
    outcome: RowOutcome
```

**Effort:** 2 hours

---

### HIGH-002: Plugin Type Detection via Duck Typing

**Location:** `src/elspeth/engine/processor.py:134, 162`

**Pattern:** Implicit contract - method presence determines type

**Current State:**
```python
if hasattr(transform, "evaluate"):
    # It's a Gate
    outcome = self._gate_executor.execute_gate(...)
elif hasattr(transform, "accept"):
    # It's an Aggregation
    accept_result = self._aggregation_executor.accept(...)
else:
    # Regular Transform
    result = self._transform_executor.execute_transform(...)
```

**Problem:**
- Order matters: `evaluate` checked before `accept`
- A plugin with both methods hits first branch only
- Base classes exist in `plugins/base.py` but aren't used
- No mypy type checking - `transform` is `Any`

**Suggested Fix:**
```python
from elspeth.plugins.base import BaseGate, BaseAggregation, BaseTransform

if isinstance(transform, BaseGate):
    outcome = self._gate_executor.execute_gate(...)
elif isinstance(transform, BaseAggregation):
    accept_result = self._aggregation_executor.accept(...)
elif isinstance(transform, BaseTransform):
    result = self._transform_executor.execute_transform(...)
else:
    raise TypeError(f"Unknown transform type: {type(transform)}")
```

**Effort:** 2 hours

---

### HIGH-003: TUI Silent Degradation on Malformed Data

**Location:** `src/elspeth/tui/widgets/lineage_tree.py:44-137`

**Pattern:** Dict-based interface with defensive `.get()` sprawl

**Current State:**
```python
def _build_tree(self) -> TreeNode:
    run_id = self._data.get("run_id", "unknown")  # Silent default
    source = self._data.get("source") or {}       # Silent default
    if isinstance(source, dict):                  # Type check instead of validation
        source_name = source.get("name", "unknown")  # Triple nesting
```

**Problem:**
- Landscape data corruption renders as "unknown" instead of error
- Audit trail viewer silently hides data quality issues
- Users see incomplete information without knowing it's incomplete

**Suggested Fix:**
```python
from typing import TypedDict

class SourceInfo(TypedDict):
    name: str
    node_id: str | None

class LineageData(TypedDict):
    run_id: str
    source: SourceInfo
    transforms: list[dict[str, str | None]]
    sinks: list[dict[str, str | None]]
    tokens: list[dict[str, Any]]

class LineageTree:
    def __init__(self, lineage_data: LineageData) -> None:
        # TypedDict enforces schema at construction
        self._data = lineage_data

    def _build_tree(self) -> TreeNode:
        # Direct access - fails loudly if field missing
        run_id = self._data["run_id"]
        source = self._data["source"]
        source_name = source["name"]
```

**Effort:** 4 hours

---

### HIGH-004: Manager's Defensive getattr() Pattern

**Location:** `src/elspeth/plugins/manager.py:83-90, 149, 159, 169, 179, 189, 199`

**Pattern:** Bug-hiding defensive code (violates CLAUDE.md)

**Current State:**
```python
return cls(
    name=getattr(plugin_cls, "name", plugin_cls.__name__),  # Silent fallback
    version=getattr(plugin_cls, "plugin_version", "0.0.0"), # Silent fallback
    determinism=getattr(plugin_cls, "determinism", Determinism.DETERMINISTIC),
)
```

**Problem:**
- Plugin forgetting to define `name` silently gets `__name__` fallback
- Audit trail records wrong plugin name without error
- Violates CLAUDE.md: "If code would fail without a defensive pattern, that failure is a bug to fix"

**Suggested Fix:**
```python
@classmethod
def from_plugin(cls, plugin_cls: type, node_type: NodeType) -> "PluginSchema":
    # Required attributes - fail loudly if missing
    try:
        name = plugin_cls.name
    except AttributeError:
        raise ValueError(
            f"Plugin {plugin_cls.__name__} must define 'name' attribute"
        )

    try:
        version = plugin_cls.plugin_version
    except AttributeError:
        raise ValueError(
            f"Plugin {plugin_cls.__name__} must define 'plugin_version' attribute"
        )

    # determinism has a legitimate default (most plugins are deterministic)
    determinism = getattr(plugin_cls, "determinism", Determinism.DETERMINISTIC)

    return cls(name=name, node_type=node_type, version=version, ...)
```

**Effort:** 2 hours

---

### HIGH-005: PipelineConfig Uses Any for Plugin Fields

**Location:** `src/elspeth/engine/orchestrator.py:30-37`

**Pattern:** Entry point lacks type safety

**Current State:**
```python
@dataclass
class PipelineConfig:
    source: Any           # Should be SourceProtocol
    transforms: list[Any] # Should be list[TransformProtocol | GateProtocol | ...]
    sinks: dict[str, Any] # Should be dict[str, SinkProtocol]
    config: dict[str, Any]
```

**Problem:**
- Primary entry point to engine has zero type checking
- Caller can pass wrong types (dict instead of sink)
- IDE provides no autocomplete or type hints
- Type errors surface only at runtime during execution

**Suggested Fix:**
```python
from elspeth.plugins.protocols import (
    SourceProtocol,
    TransformProtocol,
    GateProtocol,
    AggregationProtocol,
    SinkProtocol,
)
from elspeth.core.config import ElspethSettings

TransformLike = TransformProtocol | GateProtocol | AggregationProtocol

@dataclass
class PipelineConfig:
    source: SourceProtocol
    transforms: list[TransformLike]
    sinks: dict[str, SinkProtocol]
    config: ElspethSettings | None = None
```

**Effort:** 1 hour

---

## Medium-Severity Issues

### MED-001: Optional Field Sprawl in Models

**Location:** `src/elspeth/core/landscape/models.py`

**Problem:** `NodeState` has 8 optional fields with unclear lifecycle semantics.

**Fix:** Use discriminated unions - `NodeStateOpen`, `NodeStateCompleted`, `NodeStateFailed`.

**Effort:** 4 hours

---

### MED-002: Dict-Based Routes Parameter

**Location:** `src/elspeth/core/landscape/recorder.py:936-987`

**Problem:** `routes: list[dict[str, str]]` requires caller to know schema.

**Fix:** Create `RoutingSpec` dataclass.

**Effort:** 1 hour

---

### MED-003: Route Validation at Runtime

**Location:** `src/elspeth/engine/executors.py:327-334`

**Problem:** Config errors discovered during row processing, not pipeline construction.

**Fix:** Validate all route labels at `Orchestrator.run()` initialization.

**Effort:** 3 hours

---

### MED-004: Node Type String Filtering

**Location:** `src/elspeth/tui/screens/explain_screen.py:60-62`

**Problem:** String comparison `n.node_type == "source"` instead of enum.

**Fix:** Use `NodeType` enum with validation.

**Effort:** 1 hour

---

### MED-005: TUI Optional Sprawl

**Location:** `src/elspeth/tui/screens/explain_screen.py:26-44`

**Problem:** 5 optional fields create 32 possible state combinations.

**Fix:** Discriminated union - `InitializedScreen` vs `UninitializedScreen`.

**Effort:** 3 hours

---

### MED-006: Protocol Docstring Bad Example

**Location:** `src/elspeth/plugins/protocols.py:164-170`

**Problem:** Example shows `row.get("suspicious")` - defensive pattern in documentation.

**Fix:** Update to `row["suspicious"]` with explanation.

**Effort:** 30 minutes

---

### MED-007: Tuple Return in can_resume()

**Location:** `src/elspeth/core/checkpoint/recovery.py:66-96`

**Problem:** `tuple[bool, str | None]` return has position-encoded semantics.

**Fix:** Create `ResumeCheck` dataclass.

**Effort:** 1 hour

---

### MED-008: CSV Export Couples to Private Attribute

**Location:** `src/elspeth/engine/orchestrator.py:561-564`

**Problem:** Checks for `sink._path` private attribute.

**Fix:** Use `ArtifactDescriptor.path_or_uri` instead.

**Effort:** 1 hour

---

### MED-009: resolve_config Not Exported

**Location:** `src/elspeth/core/__init__.py`

**Problem:** `resolve_config` used by CLI but not exported from `core`.

**Fix:** Add to `core/__init__.py` exports.

**Effort:** 10 minutes

---

### MED-010: String Status Not Validated

**Location:** `src/elspeth/core/landscape/recorder.py:123`

**Problem:** `Run.status` stored as string without enum validation.

**Fix:** Add `RunStatus` enum, validate at write.

**Effort:** 2 hours

---

## Low-Severity Issues

| ID | Location | Issue | Effort |
|----|----------|-------|--------|
| LOW-001 | `node_detail.py:70-88` | Silent JSON decode fallback | 1h |
| LOW-002 | `explain_app.py:29-37` | Magic string widget IDs | 30m |
| LOW-003 | `cli.py:381-428` | Plugin registry tuple unpacking | 1h |
| LOW-004 | `orchestrator.py:257-271` | Defensive `node_to_plugin.get()` | 30m |
| LOW-005 | Multiple locations | `_MISSING` sentinel duplication | 1h |
| LOW-006 | `landscape/__init__.py` | `Checkpoint` not exported | 10m |
| LOW-007 | `core/__init__.py` | Config subclasses not exported | 30m |
| LOW-008 | `recovery.py:66` | Could use `ResumeCheck` dataclass | 1h |

---

## Cross-Subsystem Integration Map

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ CLI (cli.py)                                                                │
│  ├─ Entry point: Typer commands                                             │
│  ├─ Calls: load_settings() → ElspethSettings                               │
│  └─ Creates: PipelineConfig(source: Any, transforms: list[Any], ...)  [!]  │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Engine (orchestrator.py, processor.py)                                      │
│  ├─ Receives: PipelineConfig with Any-typed plugins  [!]                   │
│  ├─ Discriminates: hasattr(transform, "evaluate") for type  [!]            │
│  ├─ Returns: RunResult(status: str), RowResult(outcome: str)  [!]          │
│  └─ Records: All operations to Landscape                                    │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Core/Landscape (recorder.py, models.py)                                     │
│  ├─ Stores: Run, Node, Token, NodeState, Artifact models                    │
│  ├─ Issue: routes param is dict[str, str]  [!]                             │
│  ├─ Issue: get_row_data() returns ambiguous None  [!]                      │
│  └─ Provides: Lineage queries via LandscapeRecorder                         │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ TUI (explain_screen.py, lineage_tree.py, node_detail.py)                    │
│  ├─ Receives: Untyped dict[str, Any] from Landscape queries                 │
│  ├─ Issue: Defensive .get() sprawl masks data corruption  [!]              │
│  ├─ Issue: String node_type filtering  [!]                                 │
│  └─ Displays: Lineage tree, node details (with silent fallbacks)            │
└─────────────────────────────────────────────────────────────────────────────┘

Legend: [!] = Integration issue identified
```

---

## Positive Findings

### Architectural Strengths

| Area | Assessment | Details |
|------|------------|---------|
| Canonical JSON | Excellent | Two-phase normalization, strict NaN rejection, RFC 8785 compliance |
| Audit Trail | Excellent | Complete coverage, immutable states, no gaps identified |
| DAG Abstraction | Excellent | NetworkX wrapper with domain-specific API |
| Plugin Protocols | Good | Pydantic validation, frozen results, clear contracts |
| Import Hygiene | Excellent | Zero circular imports, proper TYPE_CHECKING usage |
| Token Flow | Good | Identity preserved through pipeline, proper immutability |

### No Issues Found

- Circular imports: None
- Dead imports: None
- Unused exports: None
- Shadow imports: None
- Audit trail gaps: None

---

## Remediation Plan

### Phase 1: Critical (Before Phase 3 Integration)

| ID | Issue | Effort | Owner | Target |
|----|-------|--------|-------|--------|
| CRIT-001 | Ambiguous get_row_data() | 2h | - | Phase 3 |
| CRIT-002 | close() protocol violation | 4h | - | Phase 3 |

### Phase 2: High (Before Production)

| ID | Issue | Effort | Owner | Target |
|----|-------|--------|-------|--------|
| HIGH-001 | Stringly-typed status/outcome | 2h | - | Phase 4 |
| HIGH-002 | Duck-typed plugin detection | 2h | - | Phase 4 |
| HIGH-003 | TUI defensive .get() | 4h | - | Phase 4 |
| HIGH-004 | Manager defensive getattr | 2h | - | Phase 4 |
| HIGH-005 | PipelineConfig Any types | 1h | - | Phase 4 |

### Phase 3: Medium (Quality Improvement)

| ID | Issue | Effort | Owner | Target |
|----|-------|--------|-------|--------|
| MED-001 | NodeState optional sprawl | 4h | - | Phase 5 |
| MED-002 | Dict routes parameter | 1h | - | Phase 5 |
| MED-003 | Runtime route validation | 3h | - | Phase 5 |
| MED-004-010 | Various | 8h | - | Phase 5 |

---

## Appendix: Files Analyzed

### Core Subsystem
- `src/elspeth/core/landscape/recorder.py`
- `src/elspeth/core/landscape/models.py`
- `src/elspeth/core/landscape/database.py`
- `src/elspeth/core/landscape/lineage.py`
- `src/elspeth/core/landscape/exporter.py`
- `src/elspeth/core/landscape/formatters.py`
- `src/elspeth/core/landscape/reproducibility.py`
- `src/elspeth/core/landscape/schema.py`
- `src/elspeth/core/canonical.py`
- `src/elspeth/core/config.py`
- `src/elspeth/core/payload_store.py`
- `src/elspeth/core/dag.py`
- `src/elspeth/core/checkpoint/manager.py`
- `src/elspeth/core/checkpoint/recovery.py`
- `src/elspeth/core/rate_limit/registry.py`
- `src/elspeth/core/rate_limit/limiter.py`
- `src/elspeth/core/retention/purge.py`

### Engine Subsystem
- `src/elspeth/engine/orchestrator.py`
- `src/elspeth/engine/processor.py`
- `src/elspeth/engine/tokens.py`
- `src/elspeth/engine/artifacts.py`
- `src/elspeth/engine/spans.py`
- `src/elspeth/engine/retry.py`
- `src/elspeth/engine/executors.py`
- `src/elspeth/engine/adapters.py`

### Plugins Subsystem
- `src/elspeth/plugins/protocols.py`
- `src/elspeth/plugins/hookspecs.py`
- `src/elspeth/plugins/manager.py`
- `src/elspeth/plugins/base.py`
- `src/elspeth/plugins/config_base.py`
- `src/elspeth/plugins/schemas.py`
- `src/elspeth/plugins/results.py`
- `src/elspeth/plugins/enums.py`
- `src/elspeth/plugins/context.py`
- `src/elspeth/plugins/sources/csv_source.py`
- `src/elspeth/plugins/sources/json_source.py`
- `src/elspeth/plugins/transforms/passthrough.py`
- `src/elspeth/plugins/transforms/field_mapper.py`
- `src/elspeth/plugins/gates/threshold_gate.py`
- `src/elspeth/plugins/gates/field_match_gate.py`
- `src/elspeth/plugins/gates/filter_gate.py`
- `src/elspeth/plugins/sinks/csv_sink.py`
- `src/elspeth/plugins/sinks/json_sink.py`
- `src/elspeth/plugins/sinks/database_sink.py`

### TUI/CLI Subsystem
- `src/elspeth/cli.py`
- `src/elspeth/tui/explain_app.py`
- `src/elspeth/tui/screens/explain_screen.py`
- `src/elspeth/tui/widgets/lineage_tree.py`
- `src/elspeth/tui/widgets/node_detail.py`

---

*Generated by automated integration audit - 2026-01-15*
