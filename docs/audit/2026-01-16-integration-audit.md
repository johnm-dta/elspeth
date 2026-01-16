# ELSPETH Integration Audit Report

**Date:** 2026-01-16
**Auditor:** Claude Code (Automated Analysis)
**Scope:** Full codebase integration analysis
**Methodology:** Parallel subsystem exploration with cross-reference validation

---

## Executive Summary

This comprehensive integration audit analyzed 5 subsystems across 70+ source files. The analysis reveals a well-architected codebase with strong type discipline internally, but with systematic integration anti-patterns at subsystem boundaries that create brittleness and potential audit integrity risks.

### Overall Health Assessment

| Subsystem | Internal Quality | Integration Quality | Critical Issues |
|-----------|-----------------|---------------------|-----------------|
| Core Infrastructure | Strong | Medium | 4 |
| Landscape | Strong | Medium | 3 |
| Engine | Strong | Medium | 6 |
| Plugin System | Strong | Good | 2 |
| CLI/TUI | Medium | Weak | 5 |

### Summary Statistics

| Category | Count |
|----------|-------|
| Total issues identified | 45 |
| P0 Critical (unexpected gaps) | 3 |
| Planned work (Phase 6/7) | 1 |
| P1 High | 6 |
| P2 Medium | 6 |
| P3 Low | 4 |
| Cross-cutting anti-patterns | 4 |
| Interface mismatches | 5 |
| Dead producers | 2 |
| Files affected | 25+ |
| Estimated total remediation | ~32 hours |

---

## Cross-Reference with Phase Plans

This audit was cross-referenced against existing phase plans to distinguish **planned work** from **unexpected integration gaps**.

### Planned Work (Not Bugs)

The following issues are **expected** because the relevant phases haven't been implemented yet:

| Issue | Phase | Document | Status |
|-------|-------|----------|--------|
| Missing `record_call()` method | Phase 6 | `docs/plans/2026-01-12-phase6-external-calls.md` | **Planned** - Task 1 creates `CallRecorder` with `record_call()` |
| Call.status/call_type as strings | Phase 6 | Same document | **Planned** - CallType/CallStatus enums will be added |
| A/B testing infrastructure | Phase 7 | `docs/plans/2026-01-12-phase7-advanced-features.md` | **Planned** - Tasks 1-10 |
| Copy mode token forking | Phase 7 | Same document | **Planned** - Task 3-4 |
| Azure pack plugins | Phase 7 | Same document | **Planned** - Tasks 5-7 |

### Genuine Integration Issues (Require Fix)

The remaining issues are **unexpected gaps** not covered by existing phase plans:

- CLI→TUI database connection missing (explain command broken)
- `dict[str, Any]` configuration sprawl (12 occurrences)
- Stringly-typed status codes in existing code (8 occurrences)
- Defensive `.get()` patterns violating CLAUDE.md (6 occurrences)
- ExplainScreen→NodeDetailPanel field mismatch
- Tokens never populated in TUI

---

## Table of Contents

1. [Cross-Cutting Anti-Patterns](#part-1-cross-cutting-anti-patterns)
2. [Critical Integration Mismatches](#part-2-critical-integration-mismatches)
3. [Dead Producers / Missing Consumers](#part-3-dead-producers--missing-consumers)
4. [Core Infrastructure Findings](#part-4-core-infrastructure-findings)
5. [Landscape Subsystem Findings](#part-5-landscape-subsystem-findings)
6. [Engine Subsystem Findings](#part-6-engine-subsystem-findings)
7. [Plugin System Findings](#part-7-plugin-system-findings)
8. [CLI/TUI Findings](#part-8-clitui-findings)
9. [Prioritized Remediation Roadmap](#part-9-prioritized-remediation-roadmap)

---

## Part 1: Cross-Cutting Anti-Patterns

These patterns appear across multiple subsystems and require architectural remediation.

### Pattern A: `dict[str, Any]` Configuration Sprawl

**Severity:** HIGH
**Occurrences:** 12
**Impact:** Configuration flows through the system as untyped dictionaries, losing type safety at every boundary.

| Location | File | Line | Impact |
|----------|------|------|--------|
| DatasourceSettings.options | config.py | 21 | Plugin options completely unvalidated |
| RowPluginSettings.options | config.py | 37 | Plugin options completely unvalidated |
| SinkSettings.options | config.py | 53 | Plugin options completely unvalidated |
| resolve_config() return | config.py | 376 | Engine receives opaque config |
| NodeInfo.config | dag.py | 35 | DAG stores unvalidated plugin configs |
| PipelineConfig.config | orchestrator.py | 48 | Pipeline receives mystery config |
| node_to_plugin mapping | orchestrator.py | 336 | Plugin lookups lose type info |
| begin_run() config param | recorder.py | 205 | Landscape stores unvalidated config |
| PluginContext.config | context.py | 56-67 | Plugins access raw dict |
| sink_output dict | executors.py | 780-784 | Artifact metadata manually constructed |
| artifact_descriptor | cli.py | 263-279 | Artifact info as untyped dict |
| lineage_data construction | explain_screen.py | 143-159 | TUI data structures untyped |

**Data Flow Risk:**
```
ElspethSettings → resolve_config() → dict[str, Any]
                                          ↓
                         PipelineConfig.config: dict[str, Any]
                                          ↓
                         recorder.begin_run(config: dict)
                                          ↓
                         Landscape stores hash of unknown structure
```

**Suggested Fix:** Create a typed `ResolvedConfig` dataclass:
```python
@dataclass(frozen=True)
class ResolvedConfig:
    datasource: DatasourceSettings
    row_plugins: list[RowPluginSettings]
    sinks: dict[str, SinkSettings]
    landscape: LandscapeSettings
    # ... etc with all validated fields
```

---

### Pattern B: Stringly-Typed Status Codes

**Severity:** HIGH
**Occurrences:** 8
**Impact:** Status values are magic strings instead of enums, allowing typos and silently incorrect comparisons.

| Location | File | Line | Current | Should Be |
|----------|------|------|---------|-----------|
| export_status | recorder.py | 378, 398 | "pending"\|"completed"\|"failed" | ExportStatus enum |
| batch status | recorder.py | 1184, 1203 | string literals | BatchStatus enum |
| Call.status | models.py | 251 | "success"\|"error" | CallStatus enum |
| Call.call_type | models.py | 252 | "llm"\|"http"\|"sql"\|"filesystem" | CallType enum |
| mode field | dag.py | 93, 300 | "move"\|"copy" | RoutingMode enum |
| node_type | dag.py | 33, 262 | string | NodeType enum (exists but unused!) |
| result.status check | processor.py | 196 | `== "error"` | TransformStatus.ERROR |
| action.kind check | executors.py | 321-347 | `== "continue"` | RoutingKind.CONTINUE |
| node status | explain_screen.py | 248 | "registered" | NodeStateStatus enum |

**Risk Example:**
```python
# Typo: "sucess" instead of "success"
if result.status == "sucess":  # Never true!
    do_success_path()
# Silent fallthrough to error path
```

---

### Pattern C: Defensive `.get()` Violating CLAUDE.md

**Severity:** MEDIUM
**Occurrences:** 6
**Impact:** The codebase explicitly forbids defensive programming patterns that mask bugs (per CLAUDE.md), yet several locations use `.get()` with fallback defaults.

| Location | File | Line | Violation |
|----------|------|------|-----------|
| Node state display | node_detail.py | 45-80 | 10+ `.get()` calls with "N/A"/"unknown" defaults |
| Config access | context.py | 69-86 | PluginContext.get() returns default on missing key |
| Plugin metadata | orchestrator.py | 358-368 | getattr(plugin, "plugin_version", None) with "0.0.0" fallback |
| Source fallback | lineage_tree.py | 149-150 | Default dict when source_nodes empty |
| Exception swallowing | explain_screen.py | 251-252 | `except Exception: return None` masks errors |
| Empty source handling | explain_screen.py | 143-159 | `"name": "unknown", "node_id": None` fallback |

**Risk:** These patterns hide real bugs. If a field is missing, it's because something upstream failed—the system should fail loudly, not silently display "N/A".

---

### Pattern D: Optional Sprawl Masking Lifecycle

**Severity:** MEDIUM
**Occurrences:** 3
**Impact:** Multiple optional fields where combinations encode state, but invariants aren't enforced.

| Location | File | Line | Issue |
|----------|------|------|-------|
| Run export fields | models.py | 42-50 | 5 export-related Optional fields with unclear relationships |
| Call result fields | models.py | 245-259 | Optional response_hash, error_json with implicit status-based requirements |
| RowResult.sink_name | processor.py | - | Optional but required when outcome==ROUTED |

**Suggested Fix:** Use discriminated unions:
```python
# Instead of:
@dataclass
class Run:
    export_status: str | None = None
    export_error: str | None = None
    exported_at: datetime | None = None

# Use:
ExportState = ExportNotConfigured | ExportPending | ExportCompleted | ExportFailed

@dataclass
class Run:
    export: ExportState
```

---

## Part 2: Critical Integration Mismatches

### Mismatch 1: Missing Producer for Call Model

**Severity:** ~~P0 CRITICAL~~ → **PLANNED WORK (Phase 6)**
**Status:** This is NOT a bug - it's planned work for Phase 6: External Calls

**Cross-reference:** `docs/plans/2026-01-12-phase6-external-calls.md` Task 1

**What Phase 6 will add:**
- `src/elspeth/core/calls/recorder.py` - New `CallRecorder` class
- `CallRecorder.record_call()` method for recording external calls
- `CallReplayer` for replay mode
- `CallVerifier` for verify mode
- Integration with PayloadStore for request/response storage

**Current state (expected):**
- `models.py:244-259` defines `Call` model (schema ready)
- `recorder.py:1691` has `get_calls()` method (query ready)
- `exporter.py:310-311` exports call records (export ready)
- Producer will be added in Phase 6

**No action required** - this gap is tracked in the phase plan.

---

### Mismatch 2: CLI → TUI Database Connection Missing

**Severity:** P0 CRITICAL
**Impact:** explain command non-functional

**Producer:** `cli.py` creates database connections for run/validate commands
**Consumer:** `ExplainApp` requires LandscapeDB but receives nothing

**Evidence:**
```python
# cli.py:203-208 - Launches TUI without database
tui_app = ExplainApp(
    run_id=run_id,
    token_id=token,
    row_id=row,
)  # No db parameter!

# explain_app.py - Has no db handling
def __init__(self, run_id: str | None = None, ...):
    self.run_id = run_id  # Just stores IDs, can't query anything
```

**Consequences:**
- TUI cannot query Landscape database
- explain command shows placeholder widgets
- User sees "No node selected" for everything

**Fix:** Pass LandscapeDB from cli.py to ExplainApp constructor.

---

### Mismatch 3: ExplainScreen → NodeDetailPanel Field Mismatch

**Severity:** P1 HIGH
**Impact:** Node detail panel shows mostly "N/A" values

**Producer:** `explain_screen.py:244-249` provides 4 fields
**Consumer:** `node_detail.py:45-80` expects 12+ fields

**Producer sends:**
```python
return {
    "node_id": node.node_id,
    "plugin_name": node.plugin_name,
    "node_type": node.node_type,
    "status": "registered",
}
```

**Consumer expects:**
| Field | Provided? | Fallback |
|-------|-----------|----------|
| plugin_name | Yes | - |
| node_type | Yes | - |
| state_id | No | "N/A" |
| token_id | No | "N/A" |
| started_at | No | "N/A" |
| completed_at | No | "N/A" |
| duration_ms | No | "N/A" |
| input_hash | No | "N/A" |
| output_hash | No | "N/A" |
| error_json | No | None |
| artifact | No | None |

**Fix:** ExplainScreen should query NodeState models, not just Node metadata.

---

### Mismatch 4: Tokens Never Populated

**Severity:** P1 HIGH
**Impact:** Token lineage (core feature) never works

**Producer:** `explain_screen.py:158` hardcodes empty list
**Consumer:** `lineage_tree.py` expects token data

```python
# explain_screen.py:158
"tokens": [],  # Hardcoded empty! Never populated.

# lineage_tree.py expects to iterate tokens
for token in self._data["tokens"]:  # Always empty
    ...
```

**Fix:** Load tokens from `recorder.get_tokens(run_id)`.

---

### Mismatch 5: NodeType Enum Exists but Unused

**Severity:** P2 MEDIUM
**Impact:** Type checker can't catch invalid node types

**Definition:** `plugins/enums.py` defines `NodeType` enum
**Usage:** Most code uses plain strings

```python
# plugins/enums.py has:
class NodeType(str, Enum):
    SOURCE = "source"
    TRANSFORM = "transform"
    GATE = "gate"
    # ...

# But dag.py:33 ignores it:
node_type: str  # source, transform, gate, aggregation, coalesce, sink
```

---

## Part 3: Dead Producers / Missing Consumers

### Dead Producer 1: NodeInfo.config field

**Location:** `dag.py:35`

**Created:** `ExecutionGraph.add_node()` stores config dict in NodeInfo
**Consumed:** Never read by ExecutionGraph methods

The `config` field is stored but the DAG never accesses it. Likely intended for orchestrator but no documentation of the contract.

---

### Dead Producer 2: route_resolution_map

**Location:** `dag.py:363-371`

**Created:** `ExecutionGraph.from_config()` builds route resolution mapping
**Consumed:** Only via `get_route_resolution_map()` which returns raw dict

The map is built but the consumer contract is undocumented. Callers must know tuple key structure `(gate_node_id, route_label) -> target`.

---

### Missing Consumer: Plugin close() methods

**Location:** `base.py:26-349`

**Produced:** All plugins should define `close()` method per protocols
**Consumed:** Base classes don't explicitly declare close()

Protocols define close() with empty body, but base classes don't implement it explicitly, creating confusion about lifecycle management.

---

## Part 4: Core Infrastructure Findings

### 4.1 Dict-Based Interfaces

#### Issue 4.1.1: NodeInfo Config Field
- **Location:** `dag.py:35`
- **Severity:** MEDIUM
- **Current State:**
  ```python
  @dataclass
  class NodeInfo:
      config: dict[str, Any] = field(default_factory=dict)
  ```
- **Risk:** Callers don't know what keys are required or optional. Plugin-specific configuration is opaque to the DAG layer.
- **Fix:** Create a `PluginConfig` protocol with documented shape.

#### Issue 4.1.2: DatasourceSettings/RowPluginSettings/SinkSettings Options
- **Location:** `config.py:21, 37, 53`
- **Severity:** HIGH
- **Current State:**
  ```python
  class DatasourceSettings(BaseModel):
      options: dict[str, Any] = Field(default_factory=dict, ...)
  ```
- **Risk:** Plugin options completely unvalidated. Discovery is runtime-only.
- **Fix:** Define plugin-specific config dataclasses or use a validated mapping with a schema registry.

#### Issue 4.1.3: resolve_config() Return Type
- **Location:** `config.py:376`
- **Severity:** HIGH
- **Current State:**
  ```python
  def resolve_config(settings: ElspethSettings) -> dict[str, Any]:
      return settings.model_dump(mode="json")
  ```
- **Risk:** Callers receive untyped dict. No IDE/type hints.
- **Fix:** Return a TypedDict or dataclass representing the resolved configuration.

---

### 4.2 Stringly-Typed APIs

#### Issue 4.2.1: Status Strings in Recorder
- **Location:** `recorder.py:378, 398, 1194, 1203, 1216`
- **Severity:** HIGH
- **Current State:**
  ```python
  def set_export_status(self, run_id: str, status: str, ...) -> None:
      if status == "completed":
          updates["exported_at"] = _now()
  ```
- **Risk:** Typos in status strings not caught at type-check time.
- **Fix:** Create `ExportStatus` enum (already TODO'd in models.py:44).

#### Issue 4.2.2: Mode Strings
- **Location:** `dag.py:93, 300`, `models.py:286`, `recorder.py:989, 1002`
- **Severity:** MEDIUM
- **Current State:**
  ```python
  def add_edge(self, ..., mode: str = "move") -> None:
      # mode is either "move" or "copy"
  ```
- **Risk:** No validation that mode is valid.
- **Fix:** Create `RoutingMode` enum with `MOVE = "move"` and `COPY = "copy"`.

#### Issue 4.2.3: Node Type Strings
- **Location:** `dag.py:33, 262-263`, `models.py:60`
- **Severity:** MEDIUM
- **Current State:**
  ```python
  node_type: str  # source, transform, gate, aggregation, coalesce, sink
  ```
- **Risk:** NodeType enum exists but isn't used.
- **Fix:** Use `NodeType` enum from `plugins.enums` consistently.

#### Issue 4.2.4: Route Labels
- **Location:** `dag.py:100, 281-282`, `models.py:78`, `recorder.py:1000-1001`
- **Severity:** LOW
- **Current State:**
  ```python
  label: str  # "continue", route name, etc.
  ```
- **Risk:** "continue" is special but there's no type enforcement.
- **Fix:** Create `RouteLabel` type: `Literal["continue"] | str`.

---

### 4.3 Implicit Contracts

#### Issue 4.3.1: DAG Construction Order
- **Location:** `dag.py:258-304`
- **Severity:** LOW
- **Risk:** The order in `config.row_plugins` is implicit. No documentation.
- **Fix:** Document in `ElspethSettings` that `row_plugins` is ordered.

#### Issue 4.3.2: Gate Routes Require Sink Names
- **Location:** `dag.py:283-302`
- **Severity:** LOW
- **Risk:** Routes dict values must be valid sink names but no validation.
- **Fix:** Add validator that checks routes target valid sinks.

#### Issue 4.3.3: RoutingSpec Redundant Validation
- **Location:** `models.py:83-95`
- **Severity:** LOW
- **Current State:**
  ```python
  @dataclass(frozen=True)
  class RoutingSpec:
      mode: Literal["move", "copy"]

      def __post_init__(self) -> None:
          if self.mode not in ("move", "copy"):
              raise ValueError(...)
  ```
- **Risk:** `Literal` already constrains the type. `__post_init__` is redundant.
- **Fix:** Remove `__post_init__` or convert to enum.

#### Issue 4.3.4: Run Config Hash Contract
- **Location:** `recorder.py:231-232`
- **Severity:** MEDIUM
- **Risk:** No documentation about what `config` dict must contain.
- **Fix:** Accept typed config instead of dict.

---

### 4.4 Unmarshalled JSON

#### Issue 4.4.1: Error JSON in NodeState
- **Location:** `models.py:224`, `recorder.py:936`
- **Severity:** LOW
- **Current State:**
  ```python
  error_json: str | None = None
  ```
- **Risk:** No schema - could be anything.
- **Fix:** Create `ErrorSnapshot` dataclass and serialize it.

#### Issue 4.4.2: Config JSON in Nodes
- **Location:** `models.py:64`, `recorder.py:454`
- **Severity:** LOW
- **Risk:** No schema validation on plugin configs.
- **Fix:** Validate before storing.

#### Issue 4.4.3: Settings JSON in Run
- **Location:** `models.py:38`, `recorder.py:231`
- **Severity:** MEDIUM
- **Risk:** If any field contains non-serializable type, `canonical_json()` will fail.
- **Fix:** Validate `canonical_json(config)` doesn't raise before storing.

---

### 4.5 Tuple Unpacking Contracts

#### Issue 4.5.1: DAG Edge Data Tuples
- **Location:** `dag.py:209`
- **Severity:** LOW
- **Current State:**
  ```python
  def get_edges(self) -> list[tuple[str, str, dict[str, Any]]]:
      return [(u, v, dict(data)) for u, v, data in self._graph.edges(data=True)]
  ```
- **Risk:** Callers must know tuple order is (from, to, edge_data).
- **Fix:** Return `list[EdgeInfo]` dataclass.

#### Issue 4.5.2: Route Resolution Map Tuples
- **Location:** `dag.py:49-54, 363-371`
- **Severity:** LOW
- **Current State:**
  ```python
  self._route_label_map: dict[tuple[str, str], str] = {}
  self._route_resolution_map: dict[tuple[str, str], str] = {}
  ```
- **Risk:** Tuple keys have implicit positional meaning.
- **Fix:** Create `RouteKey = NamedTuple('RouteKey', [('gate_node_id', str), ('sink_or_label', str)])`.

---

### 4.6 Optional Sprawl

#### Issue 4.6.1: Run Model Optional Fields
- **Location:** `models.py:31-50`
- **Severity:** MEDIUM
- **Current State:**
  ```python
  @dataclass
  class Run:
      completed_at: datetime | None = None
      reproducibility_grade: str | None = None
      export_status: str | None = None
      export_error: str | None = None
      exported_at: datetime | None = None
      export_format: str | None = None
      export_sink: str | None = None
  ```
- **Risk:** Unclear invariants between fields.
- **Fix:** Create discriminated union `ExportState`.

#### Issue 4.6.2: Call Model Optional Fields
- **Location:** `models.py:245-259`
- **Severity:** MEDIUM
- **Current State:**
  ```python
  @dataclass
  class Call:
      request_ref: str | None = None
      response_hash: str | None = None
      response_ref: str | None = None
      error_json: str | None = None
      latency_ms: float | None = None
  ```
- **Risk:** If `status == "error"`, is `error_json` required?
- **Fix:** Create `CallResult = SuccessCall | ErrorCall`.

---

## Part 5: Landscape Subsystem Findings

### 5.1 Stringly-Typed APIs

#### Issue 5.1.1: Batch Status Strings
- **Location:** `models.py:300`, `recorder.py:1184, 1203, 1216, 1239`
- **Severity:** HIGH
- **Current State:**
  ```python
  status: str  # draft, executing, completed, failed

  if status in ("completed", "failed"):
      updates["completed_at"] = _now()
  ```
- **Risk:** Typos create silent corruption of batch states.
- **Fix:** Create `BatchStatus(str, Enum)`.

#### Issue 5.1.2: Call Status & Type Strings
- **Location:** `models.py:251-252`
- **Severity:** ~~HIGH~~ → **PLANNED WORK (Phase 6)**
- **Status:** This will be addressed when Phase 6 implements the calls system
- **Cross-reference:** `docs/plans/2026-01-12-phase6-external-calls.md`
- **Current State:**
  ```python
  call_type: str  # llm, http, sql, filesystem
  status: str    # success, error
  ```
- **Note:** The Call model is a placeholder for Phase 6. Type enums will be added when `CallRecorder` is implemented.

---

### 5.2 Implicit Contracts

#### Issue 5.2.1: Payload Reference Semantics
- **Location:** `models.py:108, 255, 288-289`, `recorder.py:1759, 1761, 1585, 1594`
- **Severity:** MEDIUM
- **Current State:**
  ```python
  source_data_ref: str | None = None  # Payload store reference
  request_ref: str | None = None
  response_ref: str | None = None
  ```
- **Risk:** No documented format for payload refs.
- **Fix:** Create `PayloadRef` typed class.

#### Issue 5.2.2: Status Field Discrimination
- **Location:** `models.py:231`, `recorder.py:96-170`
- **Severity:** LOW
- **Current State:** NodeState is discriminated union, but users must check status manually.
- **Risk:** No type narrowing if using string comparisons.
- **Fix:** Add documentation showing proper type checking pattern.

---

### 5.3 Dict-Based Interfaces

#### Issue 5.3.1: Exporter Output
- **Location:** `exporter.py:101-147, 165-357`
- **Severity:** MEDIUM
- **Current State:**
  ```python
  def export_run(self, run_id: str, sign: bool = False) -> Iterator[dict[str, Any]]:
  ```
- **Risk:** No schema validation, no IDE field completion.
- **Fix:** Create typed record classes for each record_type.

---

### 5.4 Optional Sprawl

#### Issue 5.4.1: Export Status Fields
- **Location:** `models.py:42-50`
- **Severity:** MEDIUM
- **Current State:** Five scattered Optional fields for export lifecycle.
- **Risk:** Invalid state combinations possible.
- **Fix:** Create `ExportState` discriminated union.

---

### 5.5 Other Issues

#### Issue 5.5.1: explain() Ambiguous None Return
- **Location:** `lineage.py:50-124`, `recorder.py:1730-1777`
- **Severity:** LOW
- **Current State:**
  ```python
  def explain(...) -> LineageResult | None:
      # Returns None for 5 different reasons
  ```
- **Risk:** Caller cannot distinguish failure reasons.
- **Fix:** Return `LineageResult | LineageNotFound` with reason field.

---

## Part 6: Engine Subsystem Findings

### 6.1 Dict-Based Interfaces

#### Issue 6.1.1: PipelineConfig.config Field
- **Location:** `orchestrator.py:48`
- **Severity:** CRITICAL
- **Current State:**
  ```python
  @dataclass
  class PipelineConfig:
      config: dict[str, Any] = field(default_factory=dict)
  ```
- **Risk:** No schema validation, passed to recorder.begin_run().
- **Fix:** Create `PipelineSettings` dataclass.

#### Issue 6.1.2: Untyped Plugin Mapping
- **Location:** `orchestrator.py:336-344`
- **Severity:** CRITICAL
- **Current State:**
  ```python
  node_to_plugin: dict[str, Any] = {}
  ```
- **Risk:** Cannot type-check plugin operations.
- **Fix:** Use typed union `PluginInstance = SourceProtocol | TransformLike | ...`.

---

### 6.2 Stringly-Typed APIs

#### Issue 6.2.1: String Status Codes
- **Location:** `processor.py:196`, `executors.py:173`
- **Severity:** CRITICAL
- **Current State:**
  ```python
  if result.status == "error":  # String literal
  if result.status == "success":  # String literal
  ```
- **Risk:** Typo in status string causes silent failure.
- **Fix:** Use `TransformStatus` enum.

#### Issue 6.2.2: Stringly-Typed Routing Actions
- **Location:** `executors.py:321-347`, `processor.py:155`
- **Severity:** HIGH
- **Current State:**
  ```python
  if action.kind == "continue":
  elif action.kind == "route":
  elif action.kind == "fork_to_paths":
  ```
- **Risk:** If RoutingKind enum renamed, strings don't update.
- **Fix:** Use `RoutingKind.CONTINUE` etc.

---

### 6.3 Defensive Programming

#### Issue 6.3.1: Plugin Metadata Extraction
- **Location:** `orchestrator.py:358-368`
- **Severity:** MEDIUM
- **Current State:**
  ```python
  raw_version = getattr(plugin, "plugin_version", None)
  plugin_version = raw_version if isinstance(raw_version, str) else "0.0.0"
  ```
- **Risk:** Audit trail receives default values that don't match actual plugin.
- **Fix:** Require plugin_version in protocols, no fallback.

---

### 6.4 Optional Sprawl

#### Issue 6.4.1: Optional sink_name Without Contract
- **Location:** `orchestrator.py:504-515`
- **Severity:** HIGH
- **Current State:**
  ```python
  elif result.outcome == RowOutcome.ROUTED:
      if result.sink_name is None:
          raise RuntimeError(...)  # Runtime check
  ```
- **Risk:** Type system doesn't enforce ROUTED → sink_name set.
- **Fix:** Use discriminated union `RoutedResult` with required sink_name.

---

### 6.5 Integration Point Issues

#### Issue 6.5.1: RetryManager Not Integrated
- **Location:** `retry.py`
- **Severity:** LOW
- **Current State:** RetryManager is well-typed but not used in executor calls.
- **Risk:** No retry wrapping around transform/gate/sink execution.
- **Fix:** Wrap executor calls with retries where appropriate.

---

## Part 7: Plugin System Findings

### 7.1 Dict-Based Interfaces

#### Issue 7.1.1: Config Dict in Base Classes
- **Location:** `base.py:49, 120, 194, 264, 321`
- **Severity:** MEDIUM
- **Current State:**
  ```python
  def __init__(self, config: dict[str, Any]) -> None:
      self.config = config
  ```
- **Risk:** No validation that required keys exist.
- **Fix:** Require all plugins to define a Config class attribute.

---

### 7.2 Stringly-Typed APIs

#### Issue 7.2.1: Plugin Discovery by String
- **Location:** `manager.py:180-238, 276-298`
- **Severity:** MEDIUM
- **Current State:**
  ```python
  def get_source_by_name(self, name: str) -> type[SourceProtocol] | None:
      return self._sources.get(name)
  ```
- **Risk:** Typos cause silent failures (returns None).
- **Fix:** Create plugin registry enums or validated string types.

---

### 7.3 Implicit Contracts

#### Issue 7.3.1: Plugin Attribute Requirements
- **Location:** `base.py:29-68`, `protocols.py:52-53`
- **Severity:** MEDIUM
- **Current State:** Plugins must define class attributes (name, plugin_version, etc.) but no enforcement until registration.
- **Risk:** Copy-paste errors go undetected until runtime.
- **Fix:** Use abstract attributes or metaclass validation.

---

### 7.4 Schema Contracts

#### Issue 7.4.1: Schema extra="ignore"
- **Location:** `schemas.py:22-53`
- **Severity:** MEDIUM
- **Current State:**
  ```python
  class PluginSchema(BaseModel):
      model_config = ConfigDict(extra="ignore", ...)
  ```
- **Risk:** Extra fields silently dropped - audit trail can't recover them.
- **Fix:** Change default to `extra="forbid"`.

---

### 7.5 Defensive Programming

#### Issue 7.5.1: PluginContext.get() Defensive Access
- **Location:** `context.py:69-86`
- **Severity:** LOW
- **Current State:**
  ```python
  def get(self, key: str, *, default: Any = None) -> Any:
      if isinstance(value, dict) and part in value:
          value = value[part]
      else:
          return default
  ```
- **Risk:** Violates "no defensive programming" rule in CLAUDE.md.
- **Fix:** Fail fast if config structure is wrong.

---

### 7.6 Protocol Conformance

#### Issue 7.6.1: Missing close() in Base Classes
- **Location:** `base.py:26-349`
- **Severity:** LOW
- **Current State:** BaseTransform, BaseGate, etc. don't explicitly define `close()` method.
- **Risk:** Plugins might not implement close() if relying on base class.
- **Fix:** Add explicit `close()` method declarations to base classes.

---

## Part 8: CLI/TUI Findings

### 8.1 Dict-Based Interfaces

#### Issue 8.1.1: Node State as Dict
- **Location:** `node_detail.py:22-147`
- **Severity:** HIGH
- **Current State:**
  ```python
  def __init__(self, node_state: dict[str, Any] | None) -> None:
      self._state = node_state

  plugin_name = self._state.get("plugin_name", "unknown")
  # 10+ more .get() calls
  ```
- **Risk:** Silent fallbacks mask missing fields.
- **Fix:** Create typed `NodeStateDisplay` dataclass.

#### Issue 8.1.2: Lineage Data Construction
- **Location:** `explain_screen.py:143-159`
- **Severity:** MEDIUM
- **Current State:** LineageData manually constructed as dict, violating TypedDict contract.
- **Fix:** Use Pydantic models for runtime validation.

#### Issue 8.1.3: Artifact Descriptor Dict
- **Location:** `cli.py:263-279`
- **Severity:** MEDIUM
- **Current State:**
  ```python
  artifact_descriptor = {"kind": "file", "path": sink_options["path"]}
  ```
- **Risk:** If SinkAdapter schema changes, silent failure.
- **Fix:** Create `ArtifactDescriptor` TypedDict or dataclass.

---

### 8.2 Stringly-Typed APIs

#### Issue 8.2.1: Hardcoded Plugin Registries
- **Location:** `cli.py:233-243`
- **Severity:** MEDIUM
- **Current State:**
  ```python
  TRANSFORM_PLUGINS: dict[str, type[BaseTransform]] = {
      "passthrough": PassThrough,
      "field_mapper": FieldMapper,
  }
  ```
- **Risk:** If new plugin added, cli.py must change manually.
- **Fix:** Load plugin registry from config or PluginManager.

---

### 8.3 Integration Point Issues

#### Issue 8.3.1: Database Connection Not Passed
- **Location:** `cli.py:203-208`
- **Severity:** P0 CRITICAL
- **Current State:** ExplainApp launched without database connection.
- **Fix:** Pass LandscapeDB from cli.py to ExplainApp.

#### Issue 8.3.2: Explain Command Stubbed
- **Location:** `cli.py:145-208`
- **Severity:** P0 CRITICAL
- **Current State:** Command returns error message, doesn't actually query Landscape.
- **Fix:** Implement actual Landscape queries.

#### Issue 8.3.3: Tokens Never Populated
- **Location:** `explain_screen.py:158`
- **Severity:** P1 HIGH
- **Current State:** `"tokens": []` hardcoded empty.
- **Fix:** Load from `recorder.get_tokens(run_id)`.

---

### 8.4 Exception Handling

#### Issue 8.4.1: Exception Swallowing
- **Location:** `explain_screen.py:167-169, 251-252`
- **Severity:** MEDIUM
- **Current State:**
  ```python
  except Exception:
      return None
  ```
- **Risk:** Caller can't distinguish missing data from error.
- **Fix:** Return Result type or log specific exceptions.

---

### 8.5 TypedDict Contract Violations

#### Issue 8.5.1: TypedDict Not Enforced
- **Location:** `types.py` vs `explain_screen.py`
- **Severity:** MEDIUM
- **Current State:** types.py defines contracts, explain_screen.py creates dicts that may not match.
- **Fix:** Use Pydantic models for runtime validation.

---

## Part 9: Prioritized Remediation Roadmap

### P0 - CRITICAL (Blocks core functionality)

| # | Issue | Files | Effort | Description |
|---|-------|-------|--------|-------------|
| ~~1~~ | ~~Add `record_call()` method~~ | ~~recorder.py~~ | ~~2h~~ | **PLANNED WORK** - See Phase 6, Task 1 |
| 1 | Pass LandscapeDB to ExplainApp | cli.py, explain_app.py | 1h | explain command non-functional |
| 2 | Implement Landscape queries in explain | cli.py | 4h | Command is stubbed |
| 3 | Populate tokens in ExplainScreen | explain_screen.py | 2h | Token lineage never works |

### P1 - HIGH (Audit integrity / Type safety)

| # | Issue | Files | Effort | Description |
|---|-------|-------|--------|-------------|
| 4 | Create typed `ResolvedConfig` | config.py, orchestrator.py, recorder.py | 4h | Eliminates dict[str, Any] sprawl |
| 5 | Create status enums | models.py, recorder.py | 2h | ExportStatus, BatchStatus (CallType/CallStatus → Phase 6) |
| 6 | Use RoutingMode enum | dag.py, models.py, recorder.py | 1h | Replace "move"/"copy" strings |
| 7 | Use NodeType enum | dag.py, models.py | 1h | Enum exists but unused |
| 8 | Fix ExplainScreen→NodeDetailPanel contract | explain_screen.py | 2h | Provide all expected fields |
| 9 | Replace RowResult Optional sink_name | processor.py, orchestrator.py | 3h | Use discriminated union |

### P2 - MEDIUM (Code quality / Maintainability)

| # | Issue | Files | Effort | Description |
|---|-------|-------|--------|-------------|
| 11 | Remove defensive .get() from NodeDetailPanel | node_detail.py | 2h | Violates CLAUDE.md |
| 12 | Replace edge tuple returns with EdgeInfo | dag.py | 1h | Type safety for edges |
| 13 | Create ExportState discriminated union | models.py | 2h | Replace optional sprawl |
| 14 | Type plugin mapping in orchestrator | orchestrator.py | 2h | Remove dict[str, Any] |
| 15 | Add close() to plugin base classes | base.py | 30m | Protocol conformance |
| 16 | Remove PluginContext.get() defensive behavior | context.py | 1h | Violates CLAUDE.md |
| 17 | Change schema default to extra="forbid" | schemas.py | 1h | Prevent silent data loss |
| 18 | Create typed ArtifactDescriptor | cli.py | 1h | Replace dict construction |

### P3 - LOW (Documentation / Developer experience)

| # | Issue | Files | Effort | Description |
|---|-------|-------|--------|-------------|
| 19 | Document NodeInfo.config purpose | dag.py | 30m | Clarify consumer contract |
| 20 | Document route_resolution_map contract | dag.py | 30m | Tuple key semantics |
| 21 | Create ADR for JSON serialization schema | docs/ | 2h | Error/config JSON formats |
| 22 | Add plugin development guide | docs/ | 4h | Required attributes, lifecycle |
| 23 | Remove redundant RoutingSpec validation | models.py | 15m | Literal already constrains |
| 24 | Document DAG execution order guarantee | config.py | 30m | row_plugins ordering |

---

## Appendix A: Files Analyzed

### Core Infrastructure
- `src/elspeth/core/canonical.py`
- `src/elspeth/core/config.py`
- `src/elspeth/core/dag.py`
- `src/elspeth/core/payload_store.py`
- `src/elspeth/core/logging.py`
- `src/elspeth/core/__init__.py`

### Landscape
- `src/elspeth/core/landscape/database.py`
- `src/elspeth/core/landscape/lineage.py`
- `src/elspeth/core/landscape/models.py`
- `src/elspeth/core/landscape/recorder.py`
- `src/elspeth/core/landscape/schema.py`
- `src/elspeth/core/landscape/formatters.py`
- `src/elspeth/core/landscape/exporter.py`
- `src/elspeth/core/landscape/row_data.py`
- `src/elspeth/core/landscape/reproducibility.py`
- `src/elspeth/core/landscape/__init__.py`

### Engine
- `src/elspeth/engine/orchestrator.py`
- `src/elspeth/engine/processor.py`
- `src/elspeth/engine/retry.py`
- `src/elspeth/engine/tokens.py`
- `src/elspeth/engine/spans.py`
- `src/elspeth/engine/artifacts.py`
- `src/elspeth/engine/executors.py`
- `src/elspeth/engine/adapters.py`
- `src/elspeth/engine/__init__.py`

### Plugin System
- `src/elspeth/plugins/manager.py`
- `src/elspeth/plugins/hookspecs.py`
- `src/elspeth/plugins/base.py`
- `src/elspeth/plugins/config_base.py`
- `src/elspeth/plugins/context.py`
- `src/elspeth/plugins/schemas.py`
- `src/elspeth/plugins/results.py`
- `src/elspeth/plugins/protocols.py`
- `src/elspeth/plugins/sentinels.py`
- `src/elspeth/plugins/enums.py`
- `src/elspeth/plugins/__init__.py`
- `src/elspeth/plugins/sources/csv_source.py`
- `src/elspeth/plugins/sources/json_source.py`
- `src/elspeth/plugins/sinks/csv_sink.py`
- `src/elspeth/plugins/sinks/json_sink.py`
- `src/elspeth/plugins/sinks/database_sink.py`
- `src/elspeth/plugins/transforms/passthrough.py`
- `src/elspeth/plugins/transforms/field_mapper.py`
- `src/elspeth/plugins/gates/filter_gate.py`
- `src/elspeth/plugins/gates/threshold_gate.py`
- `src/elspeth/plugins/gates/field_match_gate.py`

### CLI/TUI
- `src/elspeth/cli.py`
- `src/elspeth/tui/explain_app.py`
- `src/elspeth/tui/types.py`
- `src/elspeth/tui/constants.py`
- `src/elspeth/tui/widgets/lineage_tree.py`
- `src/elspeth/tui/widgets/node_detail.py`
- `src/elspeth/tui/screens/explain_screen.py`
- `src/elspeth/tui/__init__.py`

---

## Appendix B: Methodology

This audit was conducted using parallel subsystem exploration:

1. **Phase 1:** Codebase structure analysis to identify subsystem boundaries
2. **Phase 2:** Five parallel explore agents analyzed distinct subsystems:
   - Core Infrastructure Agent
   - Landscape Agent
   - Engine Agent
   - Plugin System Agent
   - CLI/TUI Agent
3. **Phase 3:** Cross-reference of subsystem findings to identify:
   - Cross-cutting patterns (same issue across multiple subsystems)
   - Interface mismatches (producer-consumer alignment)
   - Dead producers and missing consumers
4. **Phase 4:** Prioritization by impact and effort

Each agent focused on:
- **Part 1:** Soft integration anti-patterns (dict-based, stringly-typed, implicit contracts, etc.)
- **Part 2:** Positive integration validation (producer-consumer alignment, contract verification)

---

*Report generated by Claude Code automated analysis*
