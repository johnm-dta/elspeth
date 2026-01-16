# Integration Audit Fixes v2

**Date:** 2026-01-17
**Status:** Planning
**Context:** Follow-up audit after contracts centralization revealed additional integration gaps

## Background

A comprehensive integration audit was performed using 6 parallel explore agents analyzing:
- Core/Landscape subsystem
- Contracts subsystem
- Engine subsystem
- Plugins subsystem
- CLI/TUI subsystem
- Cross-subsystem boundaries

This audit found 37 distinct changes needed, categorized into three implementation chunks.

---

## CHUNK 1: Quick Wins (Trivial Changes)
**Estimated Time:** ~1 hour
**Risk:** Near-zero - additive changes, applying existing patterns

### 1.1 BaseSource Missing Metadata Attributes
**File:** `src/elspeth/plugins/base.py`
**Lines:** ~303-328 (BaseSource class)

Add after `node_id` class attribute:
```python
determinism: Determinism = Determinism.IO_READ  # Sources read from external world
plugin_version: str = "0.0.0"
```

**Why:** BaseTransform, BaseGate, BaseSink all have these defaults. BaseSource was missed during centralization.

---

### 1.2 Create Aggregations Hookimpl
**File:** `src/elspeth/plugins/aggregations/__init__.py` (CREATE)
**File:** `src/elspeth/plugins/aggregations/hookimpl.py` (CREATE)

```python
# __init__.py
"""Built-in aggregation plugins."""

# hookimpl.py
"""Hook implementation for built-in aggregation plugins."""

from elspeth.plugins.hookspecs import hookimpl


class ElspethBuiltinAggregations:
    """Hook implementer for built-in aggregation plugins."""

    @hookimpl
    def elspeth_get_aggregations(self) -> list:
        """Return built-in aggregation plugin classes."""
        return []


builtin_aggregations = ElspethBuiltinAggregations()
```

---

### 1.3 Create Coalesces Hookimpl
**File:** `src/elspeth/plugins/coalesces/__init__.py` (CREATE)
**File:** `src/elspeth/plugins/coalesces/hookimpl.py` (CREATE)

Same pattern as 1.2.

---

### 1.4 Register Aggregation/Coalesce Hookimpls in Manager
**File:** `src/elspeth/plugins/manager.py`
**Method:** `register_builtin_plugins()`

Add imports and registration:
```python
from elspeth.plugins.aggregations.hookimpl import builtin_aggregations
from elspeth.plugins.coalesces.hookimpl import builtin_coalesces

# In register_builtin_plugins():
self.register(builtin_aggregations)
self.register(builtin_coalesces)
```

---

### 1.5 Fix Stringly-Typed export_status
**File:** `src/elspeth/core/landscape/recorder.py`
**Method:** `set_export_status()` (line 379)

Add `_coerce_enum()` call at method entry:
```python
def set_export_status(
    self,
    run_id: str,
    status: ExportStatus | str,
    ...
) -> None:
    status_enum = _coerce_enum(status, ExportStatus)
    updates: dict[str, Any] = {"export_status": status_enum.value}

    if status_enum == ExportStatus.COMPLETED:
        updates["exported_at"] = _now()
```

---

### 1.6 Fix Stringly-Typed batch_status
**File:** `src/elspeth/core/landscape/recorder.py`
**Method:** `update_batch_status()` (line 1197)

Same pattern as 1.5 - add `_coerce_enum()` call.

---

### 1.7 Fix String/Enum Mixing in Reproducibility
**File:** `src/elspeth/core/landscape/reproducibility.py`
**Lines:** ~96-132

Change:
```python
# From:
if current_grade == ReproducibilityGrade.REPLAY_REPRODUCIBLE.value:

# To:
current_grade_enum = ReproducibilityGrade(current_grade)
if current_grade_enum == ReproducibilityGrade.REPLAY_REPRODUCIBLE:
```

---

### 1.8 Add Checkpoint Skip Logging
**File:** `src/elspeth/engine/orchestrator.py`
**Method:** `_maybe_checkpoint()`

Add warning when checkpointing silently skipped:
```python
if self._checkpoint_manager is None:
    logger.warning(
        "Checkpoint settings enabled but no checkpoint manager configured. "
        "Checkpointing will not occur for this run."
    )
    return
```

---

### 1.9 Export resolve_config from core
**File:** `src/elspeth/core/__init__.py`

Add to imports and `__all__`:
```python
from elspeth.core.config import (
    ...existing...,
    resolve_config,
)

__all__ = [
    ...existing...,
    "resolve_config",
]
```

---

### 1.10 Add Tests for New Hookimpls
**File:** `tests/plugins/test_hookimpl_registration.py`

Add tests for aggregation and coalesce discovery (returns empty list but hook is registered).

---

## CHUNK 2: Moderate Work (Type Safety & Contracts)
**Estimated Time:** ~5-6 hours
**Risk:** Low-Medium - type signature changes, import updates

### 2.1 Consolidate PluginSchema to Contracts
**Files:**
- `src/elspeth/contracts/data.py` - Move full implementation here
- `src/elspeth/plugins/schemas.py` - Become re-export shim
- 13 files need import updates (mechanical)

Move from `plugins/schemas.py` to `contracts/data.py`:
- PluginSchema class (with `extra="ignore"`, `strict=False`)
- SchemaValidationError class
- CompatibilityResult dataclass
- `validate_row()` function
- `check_compatibility()` function

Update `plugins/schemas.py` to re-export:
```python
from elspeth.contracts.data import (
    CompatibilityResult,
    PluginSchema,
    SchemaValidationError,
    check_compatibility,
    validate_row,
)
```

Files to update imports (change to `from elspeth.contracts`):
1. `plugins/transforms/field_mapper.py`
2. `plugins/protocols.py`
3. `plugins/transforms/passthrough.py`
4. `plugins/sources/csv_source.py`
5. `engine/schema_validator.py`
6. `plugins/sources/json_source.py`
7. `plugins/base.py`
8. `plugins/gates/field_match_gate.py`
9. `plugins/gates/threshold_gate.py`
10. `plugins/sinks/csv_sink.py`
11. `plugins/sinks/json_sink.py`
12. `plugins/gates/filter_gate.py`
13. `plugins/sinks/database_sink.py`

---

### 2.2 Convert Models to Use Enums
**File:** `src/elspeth/core/landscape/models.py`

Changes:
- Line 36: `export_status: str | None` → `ExportStatus | None`
- Line 50: `node_type: str` → `NodeType`
- Line 52: `determinism: str` → `Determinism`
- Line 69: `default_mode: str` → `RoutingMode`

**Also update:** `repositories.py` to coerce strings to enums when loading from DB.

---

### 2.3 Create TypedDict for Update Schemas
**File:** `src/elspeth/core/landscape/recorder.py` (or new `update_schemas.py`)

Add:
```python
class ExportStatusUpdate(TypedDict, total=False):
    export_status: str
    exported_at: datetime
    export_error: str
    export_format: str
    export_sink: str

class BatchStatusUpdate(TypedDict, total=False):
    status: str
    completed_at: datetime
    trigger_reason: str
    state_id: str
```

---

### 2.4 Call Schema Validator from Orchestrator
**File:** `src/elspeth/engine/orchestrator.py`

Add after DAG validation:
```python
from elspeth.engine.schema_validator import validate_pipeline_schemas

schema_errors = validate_pipeline_schemas(
    source_output=config.source.output_schema,
    transform_inputs=[t.input_schema for t in config.transforms],
    transform_outputs=[t.output_schema for t in config.transforms],
    sink_inputs=[s.input_schema for s in config.sinks.values() if hasattr(s, 'input_schema')],
)
if schema_errors:
    raise SchemaValidationError(f"Pipeline schema incompatibility: {schema_errors}")
```

---

### 2.5 Fix Stringly-Typed Routing Decisions
**Files:**
- `src/elspeth/engine/executors.py` (lines 182, 334, 338, 360)
- `src/elspeth/engine/processor.py` (lines 136, 177)

Change all string comparisons to enum:
```python
# From:
if action.kind == "continue":

# To:
if action.kind == RoutingKind.CONTINUE:
```

---

### 2.6 Make node_id Assignment Explicit
**File:** `src/elspeth/engine/orchestrator.py`

Extract method with validation:
```python
def _assign_plugin_node_ids(
    self,
    config: PipelineConfig,
    source_id: str,
    transform_id_map: dict[int, str],
    sink_id_map: dict[str, str],
) -> None:
    """Explicitly assign node_id to all plugins with validation."""
    if not hasattr(config.source, 'node_id'):
        raise PluginContractError(f"Source {config.source.name} missing node_id attribute")
    config.source.node_id = source_id
    # ... etc for transforms and sinks
```

---

### 2.7 Create RetryPolicy TypedDict
**File:** `src/elspeth/contracts/engine.py` (CREATE)

```python
class RetryPolicy(TypedDict, total=False):
    """Schema for retry configuration from plugin policies."""
    max_attempts: int
    base_delay: float
    max_delay: float
    jitter: float
    retry_on: list[str]
```

Update `RetryConfig.from_policy()` signature.

---

### 2.8 Create ExecutionResult TypedDict
**File:** `src/elspeth/contracts/cli.py` (CREATE)

```python
class ExecutionResult(TypedDict):
    """Result from pipeline execution."""
    run_id: str
    status: str
    rows_processed: int
    rows_succeeded: int
    rows_failed: int
    duration_seconds: float
```

Update `cli.py:_execute_pipeline()` return type.

---

## CHUNK 3: Substantial Work (Refactors & TUI Integration)
**Estimated Time:** ~10-12 hours
**Risk:** Medium-High - larger refactors, new functionality

### 3.1 Use Repository Pattern Consistently
**File:** `src/elspeth/core/landscape/recorder.py`

Refactor `get_run()`, `get_node()`, `get_row()`, `get_token()`, etc. to use repository classes instead of duplicating conversion logic inline.

**Pattern:**
```python
# From (inline):
def get_run(self, run_id: str) -> Run | None:
    ...
    return Run(status=RunStatus(row.status), ...)

# To (use repo):
def get_run(self, run_id: str) -> Run | None:
    ...
    return RunRepository.load(row)
```

---

### 3.2 Create RoutingMap Dataclass
**File:** `src/elspeth/engine/routing_map.py` (CREATE)

```python
@dataclass
class RoutingMap:
    """Type-safe routing resolution."""
    edges: dict[tuple[str, str], str]
    destinations: dict[tuple[str, str], str]

    def get_edge_id(self, node_id: str, label: str) -> str:
        """Get edge ID with clear error context."""
        key = (node_id, label)
        if key not in self.edges:
            raise MissingEdgeError(...)
        return self.edges[key]
```

Update all uses of raw `edge_map` and `route_resolution_map` dicts in orchestrator and executors.

---

### 3.3 Create ExecutionGraphProtocol
**File:** `src/elspeth/contracts/engine.py`

```python
class ExecutionGraphProtocol(Protocol):
    """Contract for execution graph implementations."""
    def add_node(self, node_id: str, *, node_type: str, plugin_name: str) -> None: ...
    def add_edge(self, from_node: str, to_node: str, *, label: str, mode: RoutingMode) -> None: ...
    def get_edges(self) -> list[EdgeInfo]: ...
    def topological_order(self) -> list[str]: ...
    # ... etc
```

---

### 3.4 Create SinkLike Protocol
**File:** `src/elspeth/plugins/protocols.py` or `contracts/engine.py`

```python
class SinkLike(Protocol):
    """Protocol for bulk sink operations (Phase 3B adapter interface)."""
    name: str
    node_id: str | None

    def write(self, rows: list[dict[str, Any]], ctx: PluginContext) -> ArtifactDescriptor: ...
```

Update `PipelineConfig.sinks` type hint.

---

### 3.5 Add Database Parameter to ExplainApp
**File:** `src/elspeth/tui/explain_app.py`

```python
def __init__(
    self,
    db: "LandscapeDB | None" = None,
    run_id: str | None = None,
    ...
) -> None:
    self._db = db
```

---

### 3.6 Wire ExplainScreen into ExplainApp
**File:** `src/elspeth/tui/explain_app.py`

Update `compose()` to create actual widgets instead of placeholders.

---

### 3.7 Create NodeStateDisplay TypedDict
**File:** `src/elspeth/tui/types.py`

```python
class NodeStateDisplay(TypedDict):
    """Contract for node state data displayed in TUI."""
    state_id: str
    node_id: str
    token_id: str
    plugin_name: str
    node_type: str
    status: str
    # ... all fields
```

---

### 3.8 Remove Defensive .get() Calls from NodeDetailPanel
**File:** `src/elspeth/tui/widgets/node_detail.py`

**⚠️ HIGH RISK** - Changes failure mode from "degraded display" to "crash on malformed data"

Change all `.get()` calls to direct access after NodeStateDisplay contract is in place.

---

### 3.9 Implement Token Lineage Path Tracing
**File:** `src/elspeth/core/landscape/recorder.py`

Add method:
```python
def trace_token_path(self, token_id: str) -> list[str]:
    """Get ordered list of node names for a token's DAG path."""
```

---

### 3.10 Fix Tokens Field in ExplainScreen
**File:** `src/elspeth/tui/screens/explain_screen.py`

Populate `tokens` field instead of hardcoding `[]`.

---

### 3.11 CLI Passes Database to TUI
**File:** `src/elspeth/cli.py`

Update `explain` command to load config, create db connection, resolve "latest", and pass to ExplainApp.

---

## Summary

| Chunk | Changes | Time | Risk |
|-------|---------|------|------|
| 1 (Quick Wins) | 10 | ~1 hr | Near-zero |
| 2 (Type Safety) | 8 | ~5-6 hrs | Low-Medium |
| 3 (Refactors) | 11 | ~10-12 hrs | Medium-High |
| **Total** | **29** | **~18 hrs** | |

## Implementation Notes

- Each chunk should be a separate PR
- Run full test suite after each chunk
- Chunk 3 items 3.5-3.11 are coupled (TUI wiring) - do together
- Item 3.8 (remove .get()) should be done LAST after NodeStateDisplay contract proven
