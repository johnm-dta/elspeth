# ELSPETH RC-2 Comprehensive Remediation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Date:** 2026-01-27
**Source:** Architecture Analysis by 17+ parallel agents (archive/2026-01-27-arch-analysis/)
**Total Issues Identified:** 75+
**Recommendation:** Relabel as Alpha until Phase 1-2 complete

---

## Executive Summary

ELSPETH has sound architecture but incomplete implementation. This plan captures **every finding** from comprehensive analysis, organized into 6 phases with effort estimates and dependencies.

| Phase | Focus | Duration | Items |
|-------|-------|----------|-------|
| 0 | Quick Wins | 1 week | 12 |
| 1 | Critical Fixes | 2 weeks | 4 |
| 2 | Core Feature Completion | 2 weeks | 6 |
| 3 | Production Hardening | 2 weeks | 12 |
| 4 | Architecture Cleanup | 3 weeks | 18 |
| 5 | Quality & Documentation | Ongoing | 23+ |

**Verdict:** RC-2 ready once Phase 1-2 complete.

---

## Phase 0: Quick Wins (High Impact, Low Effort)

*Complete these first - maximum ROI, builds momentum*

### QW-01: Fix Memory Leak in `_completed_keys`
**Source:** TD-007 | **Effort:** 0.5 days | **Impact:** Prevents OOM

**File:** `src/elspeth/engine/coalesce_executor.py:172-199`
```python
# CURRENT: _completed_keys grows unbounded
self._completed_keys.add(key)

# FIX: Remove key after merge result emitted
self._completed_keys.discard(key)  # Add after emit
```

**Test:** Long-running pipeline simulation verifies stable memory.

---

### QW-02: Fix Silent JSON Parse Fallback
**Source:** TD-008 | **Effort:** 0.5 days | **Impact:** Better diagnostics

**File:** `src/elspeth/plugins/clients/http.py:164-169`
```python
# CURRENT:
except Exception:
    response_body = response.text  # Silent fallback

# FIX:
except json.JSONDecodeError as e:
    logger.warning("JSON parse failed", content_type=response.headers.get("content-type"), body_preview=response.text[:200])
    return TransformResult.error({"reason": "json_parse_failed", "error": str(e), "body_preview": response.text[:200]})
```

---

### QW-03: Consolidate Duplicate PayloadStore Protocols
**Source:** TD-011 | **Effort:** 0.5 days | **Impact:** Single source of truth

**Files:**
- `src/elspeth/core/payload_store.py:28-83` (full protocol)
- `src/elspeth/core/retention/purge.py:28-41` (minimal protocol)

**Fix:** Extract to `src/elspeth/contracts/payload_store.py`, import from both locations.

---

### QW-04: Add BatchStatus Enum Validation
**Source:** TD-020 | **Effort:** 0.5 days | **Impact:** Data integrity

**File:** `src/elspeth/core/landscape/recorder.py:1319-1348`
```python
# CURRENT: accepts raw string
def update_batch_status(self, batch_id: str, status: str) -> None:

# FIX: validate against enum
def update_batch_status(self, batch_id: str, status: BatchStatus) -> None:
    if not isinstance(status, BatchStatus):
        raise ValueError(f"status must be BatchStatus enum, got {type(status)}")
```

---

### QW-05: Replace Hardcoded Checkpoint Date with Version
**Source:** TD-028 | **Effort:** 0.5 days | **Impact:** Future-proofs checkpoints

**File:** `src/elspeth/core/checkpoint/manager.py:202-233`
```python
# CURRENT:
cutoff_date = datetime(2026, 1, 24, tzinfo=UTC)

# FIX: Use version field
CHECKPOINT_FORMAT_VERSION = 2
if checkpoint.format_version < CHECKPOINT_FORMAT_VERSION:
    raise IncompatibleCheckpointError(...)
```

---

### QW-06: Fix Broken Example (multi_query_assessment)
**Source:** Examples Analysis | **Effort:** 0.25 days | **Impact:** Example works

**File:** `examples/multi_query_assessment/suite.yaml`
```yaml
# CURRENT (broken):
source:
  plugin: csv_source
sinks:
  - name: results
    plugin: csv_sink

# FIX:
source:
  plugin: csv
sinks:
  results:
    plugin: csv
```

---

### QW-07: Add DSN Format Validation
**Source:** Config Validation Analysis | **Effort:** 0.5 days | **Impact:** Fail-fast on bad URLs

**File:** `src/elspeth/core/config.py` (LandscapeSettings)
```python
@field_validator("url")
def validate_database_url(cls, v: str) -> str:
    try:
        from sqlalchemy.engine.url import make_url
        make_url(v)
    except ArgumentError as e:
        raise ValueError(f"Invalid database URL format: {e}")
    return v
```

---

### QW-08: Validate Required Environment Variables
**Source:** Config Validation Analysis | **Effort:** 0.5 days | **Impact:** Clear error messages

**File:** `src/elspeth/core/config.py` (_expand_env_vars function)
```python
# CURRENT: Unset ${VAR} kept as literal string
return match.group(0)

# FIX: Fail on missing required env var
if env_value is None and default is None:
    raise ValueError(f"Required environment variable '{var_name}' not set. Use ${{{var_name}:-default}} for optional vars.")
```

---

### QW-09: Remove Dead Code (models.py)
**Source:** TD-021 | **Effort:** 0.5 days | **Impact:** Less maintenance

**File:** `src/elspeth/core/landscape/models.py` (393 LOC)

**Verification:** Confirm no imports, then delete file.

---

### QW-10: Fix CLI Code Duplication
**Source:** TD-014 | **Effort:** 1 day | **Impact:** 50% less event handler code

**File:** `src/elspeth/cli.py:471-594` and `683-806`

**Fix:** Extract shared event handling to `_create_event_handlers()` helper function, call from all three locations.

---

### QW-11: Fix Hardcoded Plugin Lookup Tables
**Source:** TD-019 | **Effort:** 0.5 days | **Impact:** Easier plugin extensibility

**File:** `src/elspeth/plugins/validation.py:85-109`

**Fix:** Generate lookup table from plugin registry instead of hardcoding.

---

### QW-12: Fix Example README CLI Command
**Source:** Examples Analysis | **Effort:** 0.25 days | **Impact:** Docs correctness

**File:** `examples/large_scale_test/README.md:76`
```bash
# CURRENT (wrong):
elspeth explain --run latest --row 1234

# FIX:
uv run elspeth explain -s examples/large_scale_test/settings.yaml --run latest --row 1234
```

---

## Phase 1: Critical Fixes (Blocks Production)

*Must complete before any production deployment*

### CRIT-01: Wire Rate Limiting to Engine
**Source:** TD-001 | **Effort:** 3-5 days | **Priority:** CRITICAL

**Current State:**
- `src/elspeth/core/rate_limit/registry.py` - Complete implementation (~250 LOC)
- `src/elspeth/engine/orchestrator.py` - No rate limiting imports

**Implementation:**

**Step 1:** Add rate_limit parameter to Orchestrator
```python
# orchestrator.py
def __init__(
    self,
    ...
    rate_limit_registry: RateLimitRegistry | None = None,
):
    self._rate_limit_registry = rate_limit_registry
```

**Step 2:** Pass registry to PluginContext
```python
# context.py
rate_limiter: RateLimitRegistry | None = None
```

**Step 3:** Use in LLM transforms before API calls
```python
# azure.py
if ctx.rate_limiter:
    ctx.rate_limiter.acquire("azure_openai", weight=1)
```

**Step 4:** Wire from CLI
```python
# cli.py - in _execute_pipeline
if config.rate_limits:
    registry = RateLimitRegistry.from_config(config.rate_limits)
else:
    registry = None
orchestrator = Orchestrator(..., rate_limit_registry=registry)
```

**Test:**
```python
def test_rate_limiting_actually_limits():
    registry = RateLimitRegistry({"azure": {"requests_per_second": 1}})
    # Make 10 calls, verify they take ~10 seconds
```

**Exit Criteria:**
- [ ] `RateLimitRegistry` instantiated in Orchestrator
- [ ] LLM calls respect configured limits
- [ ] Integration test proves rate limiting works

---

### CRIT-02: Replace Defensive `.get()` with Boundary Validation
**Source:** TD-002 | **Effort:** 1-2 days | **Priority:** CRITICAL

**Files:**
- `src/elspeth/plugins/llm/azure_batch.py:768-774`
- `src/elspeth/plugins/llm/azure.py`
- `src/elspeth/plugins/llm/azure_multi_query.py`

**Current (violates Three-Tier Trust Model):**
```python
response = result.get("response", {})
body = response.get("body", {})
choices = body.get("choices", [])
```

**Fix (validate at boundary):**
```python
if "response" not in result:
    return TransformResult.error({
        "reason": "malformed_api_response",
        "detail": "missing 'response' key",
        "available_keys": list(result.keys())
    })

response = result["response"]
if not isinstance(response, dict) or "body" not in response:
    return TransformResult.error({
        "reason": "malformed_api_response",
        "detail": "invalid response structure",
        "response_type": type(response).__name__
    })

body = response["body"]
# Continue with validated data - no more .get() needed
```

**Exit Criteria:**
- [ ] No `.get()` chains on external API responses
- [ ] Schema changes produce specific error messages
- [ ] All LLM transforms follow same pattern

---

### CRIT-03: Call Coalesce Timeout in Processor Loop
**Source:** TD-003 | **Effort:** 2-3 days | **Priority:** CRITICAL

**Current State:**
- `coalesce_executor.py:371-440` - `check_timeouts()` exists
- `processor.py` - Never calls it

**Implementation:**

**Step 1:** Add timeout check interval config
```python
# config.py
class ProcessorSettings(BaseModel):
    coalesce_timeout_check_interval_ms: int = 1000
```

**Step 2:** Check timeouts periodically in _process_loop
```python
# processor.py - in main loop
last_timeout_check = time.monotonic()
while work_queue:
    # ... existing processing ...

    if time.monotonic() - last_timeout_check > timeout_check_interval:
        for coalesce_name in self._coalesce_configs:
            outcomes = self._coalesce_executor.check_timeouts(coalesce_name, current_step)
            for outcome in outcomes:
                self._handle_coalesce_outcome(outcome)
        last_timeout_check = time.monotonic()
```

**Test:**
```python
def test_coalesce_timeout_fires_during_processing():
    # Configure 3-branch fork with 1s timeout
    # Process 2 branches, delay 3rd
    # Verify timeout fires before end-of-source
```

**Exit Criteria:**
- [ ] `check_timeouts()` called in processor main loop
- [ ] Configurable check interval
- [ ] Test with deliberate branch failure times out correctly

---

### CRIT-04: Fix HTTP JSON Parse Error Handling
**Source:** TD-008 (extended) | **Effort:** 1 day | **Priority:** CRITICAL

**File:** `src/elspeth/plugins/clients/http.py:164-169`

Already covered in QW-02, but elevating to critical for Phase 1.

---

## Phase 2: Core Feature Completion

*Deliver the core value proposition*

### FEAT-01: Implement `explain` Command
**Source:** TD-004 | **Effort:** 5-10 days | **Priority:** HIGH

**Current State:**
- `cli.py:291-365` - Returns "not_implemented"
- `tui/screens/explain_screen.py` - 314 LOC of working screen
- `tui/widgets/lineage_tree.py` - 198 LOC of working widget
- `core/landscape/lineage.py` - Query logic exists

**Implementation:**

**Step 1:** JSON output mode (do first - simplest)
```python
# cli.py - in explain command
lineage = explain_token_lineage(landscape, run_id, token_id)
if json_output:
    typer.echo(json.dumps(lineage.to_dict(), indent=2))
    return
```

**Step 2:** Wire TUI widgets
```python
# explain_app.py - replace placeholders
def compose(self) -> ComposeResult:
    yield Header()
    yield LineageTree(self.lineage_data, id=WidgetIDs.LINEAGE_TREE)
    yield NodeDetailPanel(id=WidgetIDs.DETAIL_PANEL)
    yield Footer()
```

**Step 3:** Add query methods if missing
```python
# lineage.py
def explain_token_lineage(
    landscape: LandscapeDB,
    run_id: str,
    token_id: str | None = None,
    row_id: str | None = None,
) -> LineageResult:
    # Query source row, tokens, node_states, calls, routing_events
    # Return structured LineageResult
```

**Test:**
```python
def test_explain_returns_complete_lineage():
    # Run simple pipeline
    # Call explain on a token
    # Verify source_row, tokens, node_states, calls present
```

**Exit Criteria:**
- [ ] `elspeth explain --run X --row Y` returns lineage tree
- [ ] JSON output mode works
- [ ] TUI mode shows navigable lineage

---

### FEAT-02: Wire TUI Widgets
**Source:** TD-009 | **Effort:** 3-5 days | **Priority:** HIGH

**Files:**
- `src/elspeth/tui/screens/explain_screen.py`
- `src/elspeth/tui/widgets/lineage_tree.py` (198 LOC)
- `src/elspeth/tui/widgets/node_detail.py` (166 LOC)

**Current:** Uses `Static("placeholder")` instead of working widgets.

**Fix:** Replace placeholders with actual widget instances, wire to Landscape queries.

---

### FEAT-03: Implement Checkpoints
**Source:** TD-005 | **Effort:** 5-10 days | **Priority:** HIGH

**Current State:**
- Schema: `checkpoints_table` defined in `schema.py:373-400`
- Missing: `create_checkpoint()`, `get_latest_checkpoint()`, `CheckpointRepository`

**Implementation:**

**Step 1:** Create CheckpointRepository
```python
class CheckpointRepository:
    def create(self, checkpoint: Checkpoint) -> str: ...
    def get_latest(self, run_id: str) -> Checkpoint | None: ...
    def list_for_run(self, run_id: str) -> list[Checkpoint]: ...
```

**Step 2:** Add recorder methods
```python
# recorder.py
def create_checkpoint(self, run_id: str, state: CheckpointState) -> str: ...
def get_latest_checkpoint(self, run_id: str) -> Checkpoint | None: ...
```

**Step 3:** Integrate with Orchestrator
```python
# orchestrator.py
if self._checkpoint_interval and rows_processed % self._checkpoint_interval == 0:
    self._recorder.create_checkpoint(run_id, current_state)
```

**Exit Criteria:**
- [ ] Checkpoints written periodically during processing
- [ ] `get_latest_checkpoint()` returns valid checkpoint
- [ ] Long pipeline can resume from checkpoint

---

### FEAT-04: Add Missing CLI Commands
**Source:** TD-027 | **Effort:** 5 days | **Priority:** MEDIUM

**Commands:**
- `elspeth status` - Show run status (documented in CLAUDE.md)
- `elspeth export` - Export audit trail
- `elspeth db migrate` - Run Alembic migrations

---

### FEAT-05: Add Graceful Shutdown
**Source:** TD-017 | **Effort:** 3-5 days | **Priority:** HIGH

**Implementation:**
- Add signal handlers for SIGTERM, SIGINT
- Create checkpoint on shutdown
- Flush pending aggregations
- Complete in-flight writes

**Test:**
```python
def test_sigterm_creates_checkpoint():
    # Start long pipeline
    # Send SIGTERM
    # Verify checkpoint created
    # Verify no data loss
```

---

### FEAT-06: Add Circuit Breaker to Retry Logic
**Source:** Discovery Findings | **Effort:** 3-5 days | **Priority:** MEDIUM

**Current:** No circuit breaker - 10,000 rows against dead service = 500+ hours blocked.

**Fix:** After N consecutive failures to same endpoint, fail fast for M seconds.

---

## Phase 3: Production Hardening

*Make it reliable at scale*

### PERF-01: Fix Exporter N+1 Queries
**Source:** TD-006 | **Effort:** 3-5 days | **Priority:** HIGH

**Current:** 21,001 queries for 1000 rows
**Target:** Export 10k rows in < 60s

**Fix:** Batch-load tokens, node_states, calls per run. Join in memory.

---

### PERF-02: Add Missing Export Record Types
**Source:** Landscape Analysis | **Effort:** 2 days | **Priority:** HIGH

**Current:** Exporter misses `validation_errors`, `transform_errors`, `token_outcomes`
**Impact:** Compliance audit incomplete

**Fix:** Add export methods for all table types.

---

### PERF-03: Fix SinkExecutor O(N) Operations
**Source:** TD-016 | **Effort:** 1 day | **Priority:** MEDIUM

**File:** `src/elspeth/engine/executors.py:1563-1572`

---

### PERF-04: Add Composite Index on token_outcomes
**Source:** TD-024 | **Effort:** 0.5 days | **Priority:** MEDIUM

---

### SAFE-01: Fix Retention Race Condition
**Source:** Core Analysis | **Effort:** 2 days | **Priority:** HIGH

**Issue:** Concurrent resume + purge can delete active payloads

**Fix:** Use database transaction with row locking or optimistic concurrency.

---

### SAFE-02: Fix Swallowed Checkpoint Callback Exception
**Source:** Silent Failures Analysis | **Effort:** 1 day | **Priority:** HIGH

**File:** `src/elspeth/engine/executors.py:1644-1656`

**Current:** Exception logged but not raised, pipeline continues with corrupted state.
**Fix:** Re-raise or halt pipeline on checkpoint callback failure.

---

### SAFE-03: Fix batch_adapter Race Condition
**Source:** Engine Analysis | **Effort:** 2 days | **Priority:** HIGH

**File:** `src/elspeth/engine/batch_adapter.py:106-125`

**Issue:** `emit()` stores result during race window, gets discarded.

---

### SAFE-04: Fix DatabaseOps Non-Atomic Operations
**Source:** Landscape Analysis | **Effort:** 2 days | **Priority:** MEDIUM

**File:** `src/elspeth/core/landscape/_database_ops.py:25-45`

**Current:** Opens new connection per operation - multi-step operations not atomic.

---

### SAFE-05: Fix Call Index Counter Persistence
**Source:** TD-018 | **Effort:** 1-2 days | **Priority:** MEDIUM

**Issue:** In-memory counter resets on recorder recreation - resume may have conflicts.

---

### SAFE-06: Enable PostgreSQL Schema Validation
**Source:** Schema Evolution Analysis | **Effort:** 1-2 days | **Priority:** HIGH

**Current:** `database.py:105-106` skips ALL validation for PostgreSQL
**Fix:** Query `information_schema` to validate required columns.

---

### SAFE-07: Add Schema Version Tracking
**Source:** Schema Evolution Analysis | **Effort:** 2-3 days | **Priority:** HIGH

**Fix:** Add `schema_version` to metadata table, check on startup.

---

### SAFE-08: Fix repr_hash Audit Integrity Risk
**Source:** Core Analysis | **Effort:** 1 day | **Priority:** MEDIUM

**File:** `src/elspeth/core/canonical.py:263-287`

**Issue:** Fallback `repr()` hash not stable across Python versions.
**Fix:** Track which hash method was used, or eliminate fallback.

---

## Phase 4: Architecture Cleanup

*Reduce maintenance burden*

### ARCH-01: Add BaseCoalesce and Audit Protocol/Base Drift
**Source:** TD-010 | **Effort:** 3-5 days | **Priority:** MEDIUM

**Files:**
- `src/elspeth/plugins/base.py`
- `src/elspeth/plugins/protocols.py`

**Fix:**
1. Add `BaseCoalesce` class
2. Audit all Protocol/Base pairs for drift (`_on_error` already drifted)
3. Consider code generation or single-source-of-truth

---

### ARCH-02: Fix LLM Transform LSP Violation
**Source:** TD-013 | **Effort:** 3-5 days | **Priority:** MEDIUM

**Current:** `process()` raises `NotImplementedError` but extends `BaseTransform`

**Options:**
a) Create `BaseStreamingTransform` subclass for accept()-based transforms
b) Remove inheritance from `BaseTransform`
c) Make `process()` work as `accept()` wrapper

---

### ARCH-03: Fix Layer Violations
**Source:** TD-026 | **Effort:** 3-5 days | **Priority:** MEDIUM

**Violations:**
- `contracts/results.py` imports `MaxRetriesExceeded` from `engine/retry.py`
- `core/config.py` imports `ExpressionParser` from `engine/expression_parser.py`

**Fix:**
- Move `MaxRetriesExceeded` to contracts
- Move `ExpressionParser` to core

---

### ARCH-04: Remove or Implement OpenTelemetry
**Source:** TD-012 | **Effort:** 1-2 weeks | **Priority:** LOW

**Current:** Docstring claims OTel integration; doesn't exist.

**Options:**
a) Remove false claims from docstring (immediate)
b) Implement OTel integration (Phase 5)

---

### ARCH-05: Fix TransformExecutor Monkey-Patching
**Source:** Engine Analysis | **Effort:** 2 days | **Priority:** MEDIUM

**File:** `src/elspeth/engine/executors.py:149`

**Issue:** Dynamically attaches `_executor_batch_adapter` to transform objects.
**Fix:** Use composition instead of monkey-patching.

---

### ARCH-06: Fix AggregationExecutor Parallel Dictionaries
**Source:** Engine Analysis | **Effort:** 3 days | **Priority:** MEDIUM

**File:** `src/elspeth/engine/executors.py:870-888`

**Issue:** 5 parallel dictionaries (`_buffers`, `_counts`, `_triggers`, etc.) must stay synchronized.
**Fix:** Consolidate into single `AggregationState` dataclass.

---

### ARCH-07: Fix Lifecycle Hook Ambiguity
**Source:** Plugin Analysis | **Effort:** 1 day | **Priority:** LOW

**Issues:**
- `on_start()`/`on_complete()` ordering undefined
- `close()` vs `on_complete()` semantics overlap

**Fix:** Document lifecycle contract clearly.

---

### ARCH-08: Investigate Gate Discovery Disabled
**Source:** Plugin Analysis | **Effort:** 1 day | **Priority:** LOW

**File:** `src/elspeth/plugins/discovery.py:159-163`

**Issue:** Full infrastructure exists but gates aren't discovered. Intentional or oversight?

---

### ARCH-09: Fix Validation Subsystem Hardcoding
**Source:** Plugin Analysis | **Effort:** 2 days | **Priority:** MEDIUM

**File:** `src/elspeth/plugins/validation.py:85-109`

**Issue:** Adding plugin type requires modifying validation.py.
**Fix:** Generate from plugin registry.

---

### ARCH-10: Fix BatchTransformMixin Two-Phase Init
**Source:** Plugin Analysis | **Effort:** 2 days | **Priority:** MEDIUM

**Issue:** `connect_output()` must be called between `__init__()` and `accept()`. Missing call = silent failure.

**Fix:** Validate in `accept()` that output is connected, or restructure init.

---

### ARCH-11: Add Batch.trigger_type Enum Validation
**Source:** Contracts/CLI Analysis | **Effort:** 0.5 days | **Priority:** LOW

**Current:** `Batch.trigger_type` typed as `str` instead of `TriggerType` enum.

---

### ARCH-12: Add ExecutionResult.status Enum Validation
**Source:** Contracts/CLI Analysis | **Effort:** 0.5 days | **Priority:** LOW

**Current:** `ExecutionResult.status` typed as `str` instead of `RunStatus` enum.

---

### ARCH-13: Fix Repository Session Parameter
**Source:** TD-022 | **Effort:** 1 day | **Priority:** LOW

**Issue:** All repositories receive `None` for session and never use it.

---

### ARCH-14: Fix Resume Schema Verification Gap
**Source:** Core Analysis | **Effort:** 2 days | **Priority:** MEDIUM

**File:** `src/elspeth/core/checkpoint/recovery.py:133-226`

**Issue:** Requires `source_schema_class` but can't verify it matches original run.

---

### ARCH-15: Fix NodeInfo Mutability
**Source:** Core Analysis | **Effort:** 0.5 days | **Priority:** LOW

**File:** `src/elspeth/core/dag.py:41-55` (mutated at line 754)

**Issue:** Docstring claims immutable, code mutates it.

---

### ARCH-16: Fix Non-Atomic Payload Writes
**Source:** Core Analysis | **Effort:** 1 day | **Priority:** LOW

**File:** `src/elspeth/core/payload_store.py:113-117`

---

### ARCH-17: Fix Global Thread Exception Hook
**Source:** Core Analysis | **Effort:** 1 day | **Priority:** LOW

**File:** `src/elspeth/core/rate_limit/limiter.py:27-75`

**Issue:** Modified at import time.

---

### ARCH-18: Fix in_memory() Schema Validation Bypass
**Source:** TD-023 | **Effort:** 1 day | **Priority:** MEDIUM

**File:** `src/elspeth/core/landscape/database.py:188-202`

**Issue:** Tests using in-memory database may miss schema issues.

---

## Phase 5: Quality & Documentation

*Ongoing improvements*

### TEST-01: Fix Test Path Integrity Violations
**Source:** TD-015 | **Effort:** 1-2 weeks | **Priority:** HIGH

**Evidence:** 62+ instances of `graph._` private access in tests.

**Fix:**
1. Audit all tests using `graph._` private access
2. Refactor to use `from_plugin_instances()` factory
3. Add CI check to flag private attribute access

---

### TEST-02: Increase Property Testing Coverage
**Source:** TD-025 | **Effort:** Ongoing | **Priority:** MEDIUM

**Current:** Only 1.2% (3 files) use property testing.

**Priority invariants:**
- Audit trail completeness (every token reaches terminal state)
- Fork-join balance
- Schema compatibility transitivity
- DAG routing consistency

---

### TEST-03: Increase CLI Test Density
**Source:** Discovery Findings | **Effort:** 1 week | **Priority:** MEDIUM

**Current:** 5.96% test coverage for 1778 lines of CLI code.

---

### TEST-04: Add TUI Tests for ExplainScreen
**Source:** Test Gaps Analysis | **Effort:** 2 days | **Priority:** MEDIUM

---

### OBS-01: Add Exception Recording to Spans
**Source:** Observability Analysis | **Effort:** 2-3 hours | **Priority:** MEDIUM

---

### OBS-02: Add Correlation IDs
**Source:** Observability Analysis | **Effort:** 2-3 hours | **Priority:** MEDIUM

**Current:** structlog has contextvars processor but ContextVars never populated.

---

### OBS-03: Add Structured Logging to Major Operations
**Source:** Observability Analysis | **Effort:** 3-5 hours | **Priority:** MEDIUM

**Current:** Only 4 files use logging.

---

### OBS-04: Add Metrics/Prometheus Integration
**Source:** Observability Analysis | **Effort:** 1 week | **Priority:** LOW

---

### DOC-01: Add Missing Example READMEs
**Source:** Examples Analysis | **Effort:** 2 days | **Priority:** LOW

**Missing:** batch_aggregation, boolean_routing, deaggregation, json_explode, threshold_gate, threshold_gate_container

---

### DOC-02: Add Fork/Coalesce Examples
**Source:** Examples Analysis | **Effort:** 2 days | **Priority:** MEDIUM

---

### DOC-03: Document Access Control Limitations
**Source:** Security Analysis | **Effort:** 0.5 days | **Priority:** LOW

**Add:** "ELSPETH is not multi-user. Assumes single-user or fully trusted network."

---

### DOC-04: Document Checkpoint Breaking Change
**Source:** Schema Evolution Analysis | **Effort:** 0.5 days | **Priority:** MEDIUM

**Communicate:** All checkpoints before 2026-01-24 are invalid due to node ID change.

---

### DOC-05: Enable Audit Export Signing in Production
**Source:** Security Analysis | **Effort:** 0.5 days | **Priority:** LOW

**Document:** Enable `landscape.export.sign = true` for legal-grade integrity.

---

### SEC-01: Add Symlink Defense (Optional)
**Source:** Security Analysis | **Effort:** 1 day | **Priority:** LOW

**Fix:** Add `resolve(strict=True)` to path validation.

---

## Effort Summary

| Phase | Total Effort | Items |
|-------|--------------|-------|
| Phase 0: Quick Wins | ~6 days | 12 |
| Phase 1: Critical Fixes | ~8-12 days | 4 |
| Phase 2: Core Features | ~25-40 days | 6 |
| Phase 3: Production Hardening | ~20-30 days | 12 |
| Phase 4: Architecture Cleanup | ~25-35 days | 18 |
| Phase 5: Quality & Documentation | Ongoing | 23+ |

**Minimum to Production (Phases 0-2):** ~40-60 days
**Full Remediation (Phases 0-4):** ~85-130 days

---

## Exit Criteria by Phase

### Phase 0 Complete When:
- [ ] All 12 quick wins implemented and tested
- [ ] No OOM risk in long-running pipelines
- [ ] Better error diagnostics from JSON parse failures
- [ ] Broken example fixed

### Phase 1 Complete When:
- [ ] Production pipeline runs without Azure rate-limit errors
- [ ] Pipeline with failed branch times out correctly (not hangs)
- [ ] API response schema changes produce actionable error messages
- [ ] All changes have integration tests

### Phase 2 Complete When:
- [ ] `elspeth explain --run <id> --row <id>` returns meaningful output
- [ ] TUI shows lineage tree that can be navigated
- [ ] Long-running pipeline can be stopped and resumed
- [ ] Graceful shutdown creates checkpoint

### Phase 3 Complete When:
- [ ] 10,000 row export completes in < 60 seconds
- [ ] 100,000 row pipeline maintains stable memory
- [ ] SIGTERM results in clean shutdown with checkpoint
- [ ] 1000 calls to dead endpoint completes in < 10 seconds (circuit breaker)
- [ ] PostgreSQL deployments have schema validation

### Phase 4 Complete When:
- [ ] No layer violation warnings in CI
- [ ] No test files with `graph._` access
- [ ] Single PayloadStore protocol
- [ ] All plugin types have matching Protocol/Base
- [ ] LLM transforms have clear execution model

---

## Risk Mitigation

### Risk: Rate Limiting Changes Break Existing Pipelines
**Mitigation:** Rate limiting remains optional (None if not configured). Existing configs without rate_limits continue to work.

### Risk: Coalesce Timeout Changes Break Fork/Join Logic
**Mitigation:** Add feature flag for timeout behavior. Default to current behavior (timeout only at end-of-source). New behavior opt-in via config.

### Risk: Explain Implementation Takes Too Long
**Mitigation:** JSON output first (simplest). TUI wiring second. Incremental PRs, not one big change.

### Risk: Schema Changes Break Existing Databases
**Mitigation:** Use Alembic migrations. Test upgrade and downgrade paths. Add schema version tracking.

---

## Appendix: Evidence Commands

```bash
# Verify rate limiting disconnected
grep -r "RateLimitRegistry" src/elspeth/engine/

# Verify check_timeouts not called
grep -r "check_timeouts" src/elspeth/engine/processor.py

# Count test path violations
grep -r "graph\._" tests/engine/ --include="*.py" | grep -v "graph\._graph" | wc -l

# Find layer violations
grep -r "from.*engine" src/elspeth/contracts/
grep -r "from.*engine" src/elspeth/core/

# Find duplicate protocols
grep -r "class.*Protocol" src/elspeth/core/

# Verify explain returns not_implemented
grep -r "not_implemented" src/elspeth/cli.py
```

---

## Document History

| Date | Version | Changes |
|------|---------|---------|
| 2026-01-27 | 1.0 | Initial comprehensive remediation plan from 17+ agent analysis |
