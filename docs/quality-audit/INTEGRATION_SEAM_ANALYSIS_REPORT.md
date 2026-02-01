# ELSPETH Integration Seam Analysis Report

**Project:** ELSPETH Rapid (RC-1)
**Analysis Date:** 2026-01-25
**Report Date:** 2026-01-26
**Git Commit:** 708ea26
**Analysis Duration:** 1453.1s (24.2 minutes)
**Analyst:** Automated Integration Seam Hunt (codex)

---

## Executive Summary

This report documents a systematic architectural quality audit examining **10 engine modules** for integration seam defects—places where subsystems fail to connect cleanly through well-defined contracts. The analysis identified **9 significant architectural issues** requiring remediation before RC-1 release.

### Key Findings

| Priority | Count | Description |
|----------|-------|-------------|
| **P1** | 7 | Critical architectural defects (contract violations, leaky abstractions) |
| **P2** | 1 | Important impedance mismatches |
| **P3** | 1 | Technical debt (observability drift) |

**Critical Discovery:** Most defects stem from **implicit boundary contracts**—assumptions about how subsystems interact that aren't enforced by types, protocols, or interfaces. This represents the gap between code that "works today" and code that "cannot break tomorrow."

### Impact Assessment

**Pre-RC-1 Risk:** 6 findings pose direct threats to ELSPETH's core auditability guarantees:
- Silent audit trail gaps (missing failure records)
- Architectural layering violations (engine depends on plugin packs)
- Contract contradictions (protocol vs inheritance dispatch)
- Boundary leaks (engine accessing database schema directly)

**Estimated Remediation:** 2-3 developer-days to resolve all P1 blockers.

**Recommendation:** Fix 6 critical issues (Items #1-6 in Action Plan) before RC-1 release. Remaining 3 issues can be tracked as technical debt.

---

## Table of Contents

1. [Methodology](#methodology)
2. [Detailed Findings](#detailed-findings)
   - [P1 Critical Issues](#p1-critical-issues)
   - [P2 Important Issues](#p2-important-issues)
   - [P3 Technical Debt](#p3-technical-debt)
3. [Thematic Analysis](#thematic-analysis)
4. [Action Plan](#action-plan)
5. [Verification Strategy](#verification-strategy)
6. [Long-Term Recommendations](#long-term-recommendations)
7. [Appendix: Evidence Details](#appendix-evidence-details)

---

## Methodology

### Scope

**Files Analyzed:** 10 engine modules from `src/elspeth/engine/`
- `orchestrator.py` - SDA pipeline orchestration
- `processor.py` - Row-by-row processing
- `executors.py` - Transform/sink execution
- `coalesce_executor.py` - Join/merge operations
- `tokens.py` - Row token identity tracking
- `triggers.py` - Aggregation trigger evaluation
- `expression_parser.py` - Gate condition AST parsing
- `retry.py` - Retry management with backoff
- `spans.py` - OpenTelemetry tracing
- `artifacts.py` - Output file handling (clean)

### Analysis Technique

The integration seam hunt examines **boundary interactions** between subsystems, looking for:

1. **Parallel Type Evolution** - Duplicate definitions of same concept
2. **Impedance Mismatch** - Complex translation at boundaries
3. **Leaky Abstraction** - Implementation details crossing boundaries
4. **Contract Violation** - Undocumented assumptions between components
5. **Shared Mutable State** - Unclear ownership
6. **God Object** - Excessive coupling through large context
7. **Stringly-Typed Interface** - Magic strings instead of types
8. **Missing Facade** - Complex subsystem without simple interface
9. **Protocol Drift** - Incompatible versions of same contract
10. **Callback Hell** - Complex async chains
11. **Missing Error Translation** - Low-level errors leaking
12. **Implicit State Dependencies** - Call order matters but not enforced

### Quality Gates

- **Evidence Requirement:** Each finding must cite specific file paths and line numbers from both sides of the seam
- **Evidence Gate:** 15 potential findings were downgraded for insufficient evidence
- **Priority Calibration:** Severity based on auditability impact, not just code aesthetics

### Clean Files

- `artifacts.py` - No integration seam defects detected

---

## Detailed Findings

### P1 Critical Issues

#### Finding #1: Coalesce Failure Outcomes Never Recorded in Audit Trail

**File:** `coalesce_executor.py`
**Severity:** Major
**Anti-Pattern:** Contract Violation (undocumented assumptions)
**Boundary:** engine/coalesce_executor ↔ engine/orchestrator

**Problem:**

Coalesce failure outcomes (quorum_not_met, incomplete_branches) are returned from `flush_pending()` but **never written to the audit trail**. The orchestrator assumes the executor already logged the failure, but inspection shows CoalesceExecutor only records successful merges.

**Evidence:**

*Side A - Executor returns failure metadata:*
```python
# src/elspeth/engine/coalesce_executor.py:422-434
results.append(
    CoalesceOutcome(
        held=False,
        failure_reason="quorum_not_met",
        coalesce_metadata={
            "policy": settings.policy,
            "quorum_required": settings.quorum_count,
            "branches_arrived": list(pending.arrived.keys()),
        },
    )
)
```

*Side B - Orchestrator assumes recording happened:*
```python
# src/elspeth/engine/orchestrator.py:1063-1068
elif outcome.failure_reason:
    # Coalesce failed (timeout, missing branches, etc.)
    # Failure is recorded in audit trail by executor.  # ← THIS IS FALSE
    # Not counted as rows_failed since the individual fork children
    # were already counted when they reached their terminal states.
    pass
```

*Evidence of success-only recording:*
```python
# src/elspeth/engine/coalesce_executor.py:236-249
# Record node states for consumed tokens
for token in consumed_tokens:
    state = self._recorder.begin_node_state(...)
    self._recorder.complete_node_state(
        state_id=state.state_id,
        status="completed",  # ← Only success path
        output_data={"merged_into": merged_token.token_id},
        duration_ms=0,
    )
```

**Impact:**

- **Auditability Violation:** Fails ELSPETH's core promise that every outcome is traceable
- **Debugging Blind Spot:** Operators cannot query why quorum wasn't met
- **Data Loss:** Metadata about failed coalesces never persisted

**Root Cause:**

Failure handling responsibility was split between executor and orchestrator without a concrete audit-recording contract; the executor returns failure metadata but never writes audit records, while the orchestrator assumes it does.

**Recommended Fix:**

1. Define the canonical audit contract for coalesce failures (which tokens get terminal outcomes and which outcome values to use)
2. In `CoalesceExecutor.flush_pending()`, record node states and token outcomes with `failure_reason`/metadata via `LandscapeRecorder`
3. Update orchestrator failure handling to either assert executor recording or perform it if the executor is chosen owner
4. Add tests asserting audit records exist for `quorum_not_met` and `incomplete_branches` cases

**Effort Estimate:** 2 hours (implementation + tests)

---

#### Finding #2: Engine Imports BatchPendingError from LLM Plugin Pack

**File:** `executors.py`
**Severity:** Major
**Anti-Pattern:** Leaky Abstraction (implementation details cross boundaries)
**Boundary:** engine ↔ plugins/llm

**Problem:**

The **core engine** imports an exception from the **optional LLM plugin pack**, creating a reverse dependency that makes the engine unable to function without LLM plugins:

```python
# src/elspeth/engine/executors.py:33
from elspeth.plugins.llm.batch_errors import BatchPendingError
```

**Evidence:**

*Side A - Engine depends on plugin exception:*
```python
# src/elspeth/engine/executors.py:949-952
except BatchPendingError:
    # BatchPendingError is a CONTROL-FLOW SIGNAL, not an error.
    # The batch has been submitted but isn't complete yet.
    # Complete node_state with PENDING status and link batch for audit trail, then re-raise.
```

*Side B - Exception defined in plugin pack:*
```python
# src/elspeth/plugins/llm/batch_errors.py:14-18
class BatchPendingError(Exception):
    """Raised when batch is submitted but not yet complete.

    This is NOT an error condition - it's a control flow signal
    telling the engine to schedule a retry check later.
```

**Impact:**

- **Architectural Layering Violation:** Core engine cannot be used without LLM plugin pack
- **Modularity Loss:** Makes LLM plugins mandatory instead of optional
- **Dependency Arrow Reversal:** Core → plugins (should be plugins → core)
- **Future Extensibility Block:** Other plugin packs can't use batch patterns without importing LLM

**Root Cause:**

Batch-aware retry signaling was introduced in the LLM plugin pack and later adopted by the engine without promoting the exception to a shared contracts/engine module.

**Recommended Fix:**

1. Create `elspeth/contracts/exceptions.py` or use `elspeth/engine/errors.py`
2. Move `BatchPendingError` to the shared module as the canonical batch control-flow signal
3. Update `src/elspeth/engine/executors.py` and `src/elspeth/engine/orchestrator.py` to import from shared location
4. Update LLM batch transforms to import from shared location
5. Delete `src/elspeth/plugins/llm/batch_errors.py` (no legacy shim per CLAUDE.md)
6. Add contract-level test verifying `BatchPendingError` produces PENDING node_state during batch flush

**Effort Estimate:** 30 minutes (move file, update imports, verify tests pass)

**Priority Rationale:** This is the easiest P1 fix and prevents architectural rot from spreading.

---

#### Finding #3: Boolean Expression Classification Mismatch

**File:** `expression_parser.py`
**Severity:** Major
**Anti-Pattern:** Contract Violation (undocumented assumptions)
**Boundary:** core (config) ↔ engine (gate execution)

**Problem:**

`ExpressionParser.is_boolean_expression()` classifies any `BoolOp` (and/or) as boolean for config validation, but gate execution routes based on runtime result type. Non-boolean BoolOps can pass validation but fail or mis-route at runtime.

**Evidence:**

*Side A - Parser accepts "boolean-ish" expressions:*
```python
# src/elspeth/engine/expression_parser.py:426-430
# Boolean operators (and, or) always return truthy/falsy value
# Note: In Python, `x and y` returns y if x is truthy, not necessarily bool
# But for gate routing purposes, we treat this as boolean-ish
if isinstance(node, ast.BoolOp):
    return True
```

*Side B - Config validation requires strict boolean routing:*
```python
# src/elspeth/core/config.py:264-273
@model_validator(mode="after")
def validate_boolean_routes(self) -> "GateSettings":
    """Validate route labels match the condition's return type.

    Boolean expressions (comparisons, and/or, not) must use "true"/"false"
    as route labels.
    """
    parser = ExpressionParser(self.condition)
    if parser.is_boolean_expression():
        route_labels = set(self.routes.keys())
        expected_labels = {"true", "false"}
```

*Side C - Runtime routing uses actual result type:*
```python
# src/elspeth/engine/executors.py:553-560
# Convert evaluation result to route label
if isinstance(eval_result, bool):
    route_label = "true" if eval_result else "false"
elif isinstance(eval_result, str):
    route_label = eval_result
else:
    # Unexpected result type - convert to string
    route_label = str(eval_result)
```

**Impact:**

- **Config Validation Lies:** Passes expressions that will fail at runtime
- **Runtime Surprise:** `"foo" and "bar"` returns `"bar"` (string), not boolean, causing route mismatch
- **Silent Routing Errors:** Row goes to wrong destination without error

**Example Failure:**
```yaml
gate:
  condition: "row['status'] and row['priority']"  # Returns string, not bool!
  routes:
    true: high_priority_sink    # Never matches
    false: low_priority_sink    # Never matches
```

**Root Cause:**

ExpressionParser's boolean classification is intentionally broad ("boolean-ish") while config validation treats it as strict boolean contract, and execution uses runtime result typing without coercion; the shared contract between validation and execution is not enforced.

**Recommended Fix:**

1. **Option A (Stricter Validation):** Tighten boolean classification in `ExpressionParser._is_boolean_node()` so `ast.BoolOp` returns True only when all operands are boolean expressions (recursive check)
2. **Option B (Runtime Coercion):** If "boolean-ish" is intended, enforce it in execution by coercing `eval_result` with `bool()` when `parser.is_boolean_expression()` is True, and update validation docs accordingly
3. Add tests covering `and/or` with non-boolean operands to ensure validation and runtime routing agree

**Effort Estimate:** 2 hours (decide approach, implement, test edge cases)

---

#### Finding #4: Orchestrator Directly Queries Landscape Database Schema

**File:** `orchestrator.py`
**Severity:** Major
**Anti-Pattern:** Leaky Abstraction (implementation details cross boundaries)
**Boundary:** engine ↔ landscape

**Problem:**

Orchestrator resume path directly queries SQLAlchemy tables (`runs_table`, `edges_table`) instead of using the `LandscapeRecorder` interface, leaking persistence implementation details into the engine.

**Evidence:**

*Side A - Engine accesses database schema directly:*
```python
# src/elspeth/engine/orchestrator.py:1487-1492
from sqlalchemy import select
from elspeth.core.landscape.schema import runs_table

with self._db.engine.connect() as conn:
    run_row = conn.execute(
        select(runs_table.c.source_schema_json)
        .where(runs_table.c.run_id == run_id)
    ).fetchone()

# src/elspeth/engine/orchestrator.py:1600-1606
from elspeth.core.landscape.schema import edges_table

with self._db.engine.connect() as conn:
    edges = conn.execute(
        select(edges_table)
        .where(edges_table.c.run_id == run_id)
    ).fetchall()
```

*Side B - Recorder already provides these methods:*
```python
# src/elspeth/core/landscape/recorder.py:344-355
def get_run(self, run_id: str) -> Run | None:
    with self._db.connection() as conn:
        result = conn.execute(
            select(runs_table)
            .where(runs_table.c.run_id == run_id)
        )

# src/elspeth/core/landscape/recorder.py:694-704
def get_edges(self, run_id: str) -> list[Edge]:
    query = select(edges_table) \
        .where(edges_table.c.run_id == run_id) \
        .order_by(edges_table.c.created_at, edges_table.c.edge_id)
```

**Impact:**

- **Tight Coupling:** Engine now depends on Landscape SQLAlchemy schema details
- **Fragility:** Schema changes require engine modifications
- **Boundary Violation:** Breaks the Landscape facade abstraction
- **Future Migration Risk:** Moving to Postgres, encryption, or sharding requires engine changes

**Root Cause:**

Resume needed access to source schema and edge IDs, and the engine bypassed the LandscapeRecorder facade, letting SQLAlchemy schema details leak into the engine layer.

**Recommended Fix:**

1. Use `LandscapeRecorder.get_run()` to read `source_schema_json` (or add dedicated accessor if needed)
2. Use `LandscapeRecorder.get_edges()` to build the `edge_map` instead of querying `edges_table` directly
3. Remove all SQLAlchemy imports from `src/elspeth/engine/orchestrator.py`
4. Keep all database schema access inside the landscape subsystem

**Effort Estimate:** 1 hour (replace direct queries with recorder calls)

---

#### Finding #5: RowProcessor Rejects Protocol-Only Plugins

**File:** `processor.py`
**Severity:** Major
**Anti-Pattern:** Contract Violation (undocumented assumptions)
**Boundary:** engine ↔ plugins

**Problem:**

RowProcessor requires `BaseGate`/`BaseTransform` subclasses for dispatch, but plugin documentation states "Plugins can implement protocols directly." This creates a contract contradiction.

**Evidence:**

*Side A - Engine requires concrete base classes:*
```python
# src/elspeth/engine/processor.py:656-662
# Type-safe plugin detection using base classes
if isinstance(transform, BaseGate):
    # Gate transform
    outcome = self._gate_executor.execute_gate(
        gate=transform,
        token=current_token,
        ctx=ctx,
    )

# src/elspeth/engine/processor.py:728-862
elif isinstance(transform, BaseTransform):
    ...
else:
    raise TypeError(
        f"Unknown transform type: {type(transform).__name__}. "
        f"Expected BaseTransform or BaseGate."
    )
```

*Side B - Plugin contract says protocols are valid:*
```python
# src/elspeth/plugins/base.py:2-6
"""Base classes for plugin implementations.

These provide common functionality and ensure proper interface compliance.
Plugins can subclass these for convenience, or implement protocols directly.
```

*Side C - Plugin factory returns protocol types:*
```python
# src/elspeth/plugins/manager.py:333-358
def create_transform(
    self,
    transform_type: str,
    config: dict[str, Any]
) -> TransformProtocol:  # ← Returns protocol, not Base*
    """Create transform plugin instance with validated config."""
    plugin_cls = self.get_transform_by_name(transform_type)
    return plugin_cls(config)
```

**Impact:**

- **Contract Violation:** Documentation promises protocols work, runtime rejects them
- **Extensibility Block:** Future plugins (e.g., protocol-only Azure integrations) will fail
- **Type System Confusion:** Factory signature says `TransformProtocol`, dispatcher requires `BaseTransform`

**Root Cause:**

Engine dispatch logic was implemented around concrete base classes for runtime checks, while the plugin API documentation and factory paths evolved to protocol-based contracts, leaving the processor's runtime gate out of sync with the plugin contract.

**Recommended Fix:**

1. **Decide canonical contract:** Either enforce `Base*` inheritance everywhere OR support protocol-only plugins
2. **If protocol-only is valid:** Update `src/elspeth/engine/processor.py` to dispatch using `isinstance(transform, TransformProtocol)` or duck-typing checks (`hasattr(transform, 'process')`)
3. **If Base* is required:** Update all documentation and type hints to reflect this, remove "implement protocols directly" language
4. Add integration test that runs a protocol-only transform/gate through the processor to prevent regressions

**Effort Estimate:** 3 hours (decide contract, update dispatch logic, add tests)

**Recommendation:** Choose protocol-based dispatch for future extensibility.

---

#### Finding #6: TokenManager Leaks Payload Storage Details into Engine

**File:** `tokens.py`
**Severity:** Major
**Anti-Pattern:** Leaky Abstraction (implementation details cross boundaries)
**Boundary:** engine ↔ core/landscape

**Problem:**

TokenManager performs payload storage itself (canonicalization + store) and passes `payload_ref` to `LandscapeRecorder.create_row()`, leaking payload persistence implementation details across the boundary.

**Evidence:**

*Side A - Engine handles payload persistence:*
```python
# src/elspeth/engine/tokens.py:75-90
# Store payload if payload_store is configured (audit requirement)
payload_ref = None
if self._payload_store is not None:
    # Use canonical_json to handle pandas/numpy types, Decimal, datetime, etc.
    payload_bytes = canonical_json(row_data).encode("utf-8")
    payload_ref = self._payload_store.store(payload_bytes)

# Create row record with payload reference
row = self._recorder.create_row(
    run_id=run_id,
    source_node_id=source_node_id,
    row_index=row_index,
    data=row_data,
    payload_ref=payload_ref,  # ← Engine pre-computed this
)
```

*Side B - Recorder already handles payload storage for other records:*
```python
# src/elspeth/core/landscape/recorder.py:2072-2082
# Auto-persist request to payload store if available and ref not provided
# This enables replay/verify modes to retrieve the original request
if request_ref is None and self._payload_store is not None:
    request_bytes = canonical_json(request_data).encode("utf-8")
    request_ref = self._payload_store.store(request_bytes)
```

**Impact:**

- **Knowledge Leak:** Engine must know about canonical_json and payload storage
- **Duplication:** Payload persistence logic exists in both engine and recorder
- **Inconsistency:** Request payloads handled by recorder, row payloads by engine
- **Coupling:** Changes to payload strategy require engine updates

**Root Cause:**

Payload storage was added as an engine-side responsibility for source rows while the recorder already owns payload persistence for other audit records (calls), leaving `create_row()` without a unified façade and forcing engine code to know recorder internals.

**Recommended Fix:**

1. Move source-row payload persistence into `LandscapeRecorder.create_row()` using its `_payload_store` (if configured) so recorder owns serialization and storage
2. Remove `payload_store` handling and `canonical_json` calls from `TokenManager.create_initial_token()`, passing only `row_data`
3. Update call sites constructing `TokenManager` to stop passing `payload_store` once recorder handles persistence
4. Make the `payload_ref` parameter on `create_row()` internal or remove it to prevent bypassing recorder-owned persistence
5. Add tests asserting `create_row()` stores `source_data_ref` when payload_store is configured

**Effort Estimate:** 4 hours (refactor recorder API, update engine call sites, add tests)

---

#### Finding #7: Trigger Condition Context Mismatch

**File:** `triggers.py`
**Severity:** Major
**Anti-Pattern:** Contract Violation (undocumented assumptions)
**Boundary:** core/config ↔ engine/triggers

**Problem:**

`TriggerConfig` documentation shows examples with row-level field access (`row['type'] == 'flush_signal'`), but `TriggerEvaluator` evaluates conditions against batch-only context (`batch_count`, `batch_age_seconds`). Row-based trigger expressions are invalid at runtime.

**Evidence:**

*Side A - Config example uses row fields:*
```python
# src/elspeth/core/config.py:37-60
class TriggerConfig(BaseModel):
    """Trigger configuration for aggregation batches.

    Example YAML (combined triggers):
        trigger:
          count: 1000
          timeout_seconds: 3600
          condition: "row['type'] == 'flush_signal'"  # ← Row-level access!
```

*Side B - Evaluator provides batch-only context:*
```python
# src/elspeth/engine/triggers.py:117-125
# Check condition trigger
if self._condition_parser is not None:
    # ExpressionParser.evaluate() accepts a dict that becomes "row" in expressions.
    # So row['batch_count'] accesses this dict directly.
    context = {
        "batch_count": self._batch_count,
        "batch_age_seconds": self.batch_age_seconds,
    }
    result = self._condition_parser.evaluate(context)
```

**Impact:**

- **Documentation Lies:** Example code will fail at runtime
- **User Confusion:** Users expect row fields, get batch metrics
- **Runtime Errors:** `row['type']` raises KeyError (key doesn't exist)

**Root Cause:**

Trigger condition guidance and examples were inherited from row-based gate expressions without defining or enforcing the batch-specific context that TriggerEvaluator actually supplies.

**Recommended Fix:**

1. **Define trigger condition contract explicitly:** Row-based vs batch-metric-based
2. **If row-based:** Change `TriggerEvaluator.should_trigger()` to accept the last accepted row (and optionally batch metrics) and pass it to `ExpressionParser.evaluate()`
3. **If batch-only:** Update docs/examples to use `row['batch_count']`/`row['batch_age_seconds']` and validate allowed keys at config time
4. Add unit tests exercising trigger conditions against the chosen context, including invalid field access
5. Update sample pipeline configs to match finalized contract

**Effort Estimate:** 2 hours (decide contract, update docs or evaluator, add tests)

---

### P2 Important Issues

#### Finding #8: Retry Exponential Base Configuration Ignored

**File:** `retry.py`
**Severity:** Minor
**Anti-Pattern:** Impedance Mismatch (complex translation at boundaries)
**Boundary:** core/config ↔ engine/retry

**Problem:**

`RetrySettings` exposes `exponential_base` field for configuring backoff base, but `RetryConfig.from_settings()` drops it, so configured exponential bases are silently ignored.

**Evidence:**

*Side A - Config defines exponential_base:*
```python
# src/elspeth/core/config.py:557-565
class RetrySettings(BaseModel):
    """Retry behavior configuration."""
    max_attempts: int = Field(default=3, gt=0)
    initial_delay_seconds: float = Field(default=1.0, gt=0)
    max_delay_seconds: float = Field(default=60.0, gt=0)
    exponential_base: float = Field(
        default=2.0,
        gt=1.0,
        description="Exponential backoff base"
    )
```

*Side B - Mapping drops exponential_base:*
```python
# src/elspeth/engine/retry.py:86-101
@classmethod
def from_settings(cls, settings: "RetrySettings") -> "RetryConfig":
    return cls(
        max_attempts=settings.max_attempts,
        base_delay=settings.initial_delay_seconds,
        max_delay=settings.max_delay_seconds,
        jitter=1.0,  # Fixed jitter, not exposed in settings
        # ← exponential_base is MISSING
    )
```

*Side C - Tenacity uses default (2.0):*
```python
# src/elspeth/engine/retry.py:152-159
wait=wait_exponential_jitter(
    initial=self._config.base_delay,
    max=self._config.max_delay,
    jitter=self._config.jitter,
    # ← exp_base not passed, uses tenacity default
),
```

**Impact:**

- **Config Field Ignored:** Users set `exponential_base: 1.5` expecting gentler backoff, get 2.0
- **Silent Misconfiguration:** No warning that field is unused
- **Dead Code:** Field exists but has no effect

**Root Cause:**

RetrySettings gained `exponential_base` field but RetryConfig/RetryManager were not updated, so the configuration contract drifted from engine behavior.

**Recommended Fix:**

1. Add `exponential_base: float` to `RetryConfig`
2. Map `RetrySettings.exponential_base` in `RetryConfig.from_settings()`
3. Pass `exp_base=self._config.exponential_base` to `wait_exponential_jitter()`
4. Add/adjust tests asserting `exponential_base` affects backoff timing

**Effort Estimate:** 30 minutes (wire field through config → retry → tenacity)

---

### P3 Technical Debt

#### Finding #9: Aggregation Span Definition Drift

**File:** `spans.py`
**Severity:** Minor
**Anti-Pattern:** Protocol Drift (versions incompatible)
**Boundary:** engine/observability ↔ engine/aggregation execution

**Problem:**

`SpanFactory` still defines `aggregation_span()` method labeled for "aggregation plugins," but aggregation execution was refactored to structural batch-aware transforms. Tracing metadata and hierarchy now drift from actual execution.

**Evidence:**

*Side A - SpanFactory defines aggregation span:*
```python
# src/elspeth/engine/spans.py:193-217
@contextmanager
def aggregation_span(
    self,
    aggregation_name: str,
    *,
    batch_id: str | None = None,
) -> Iterator["Span | NoOpSpan"]:
    """Create a span for an aggregation flush.

    Args:
        aggregation_name: Name of the aggregation plugin
        batch_id: Optional batch identifier
    """
    with self._tracer.start_as_current_span(f"aggregation:{aggregation_name}") as span:
        span.set_attribute("plugin.name", aggregation_name)
        span.set_attribute("plugin.type", "aggregation")  # ← Stale metadata
        if batch_id:
            span.set_attribute("batch.id", batch_id)
```

*Side B - Aggregation execution uses transform spans:*
```python
# src/elspeth/engine/processor.py:724-733
# NOTE: BaseAggregation branch was DELETED in aggregation structural cleanup.
# Aggregation is now handled by batch-aware transforms (is_batch_aware=True).
elif isinstance(transform, BaseTransform):
    # Check if this is a batch-aware transform at an aggregation node
    node_id = transform.node_id
    if transform.is_batch_aware and node_id in self._aggregation_settings:
        # Use engine buffering for aggregation
        return self._process_batch_aggregation_node(...)

# src/elspeth/engine/executors.py:943-945
with self._spans.transform_span(transform.name, input_hash=input_hash):
    # ← Uses transform_span, not aggregation_span
    result = transform.process(buffered_rows, ctx)
```

**Impact:**

- **Observability Confusion:** Traces label aggregations as "transform" instead of "aggregation"
- **Dead Code:** `aggregation_span()` method exists but is never called
- **Dashboard Drift:** Monitoring queries for `plugin.type == "aggregation"` find nothing

**Root Cause:**

Aggregation was refactored from dedicated `BaseAggregation` to structural batch-aware transforms, but the SpanFactory API and span hierarchy documentation were not updated, leaving a stale aggregation span contract.

**Recommended Fix:**

1. **Option A (Remove):** Delete `aggregation_span()` method entirely since aggregations are now transforms
2. **Option B (Use Consistently):** Update aggregation execution to call `aggregation_span()` instead of `transform_span()` for batch flushes
3. Align span hierarchy documentation with chosen approach
4. Update monitoring dashboards if they depend on `plugin.type == "aggregation"`

**Effort Estimate:** 2 hours (decide approach, update code/docs, verify dashboards)

---

## Thematic Analysis

### Pattern 1: Protocol vs Base Class Confusion

**What Happened:**

ELSPETH's architecture evolved in phases. Early implementation (Phase 1-3) used `BaseTransform`/`BaseGate` concrete inheritance. Later design (Phase 6) shifted to protocol-based contracts for extensibility. The engine's runtime type checks never updated.

**Evidence Across Findings:**

- Finding #5: Processor requires `isinstance(transform, BaseTransform)` but docs say "implement protocols"
- Finding #2: Engine imports plugin-specific exception instead of using shared protocol

**Why It Matters:**

This blocks future plugin extensibility. Azure plugins, custom transforms, and third-party extensions will hit undocumented inheritance requirements.

**Root Pattern:** "Implementation First, Contract Later" - code written before formalizing the plugin contract, then contract documentation added without updating implementation.

---

### Pattern 2: The Landscape Boundary Leak

**What Happened:**

The Landscape subsystem provides a `LandscapeRecorder` facade, but some engine code bypasses it and accesses SQLAlchemy schema directly.

**Evidence Across Findings:**

- Finding #4: Orchestrator queries `runs_table` and `edges_table` directly
- Finding #6: TokenManager handles payload storage instead of delegating to recorder

**Why It Matters:**

SQLAlchemy implementation details now leak into the engine. Future changes (Postgres migration, encryption, sharding) will break engine code instead of being isolated to Landscape subsystem.

**Architectural Principle Violated:** "Program to an interface, not an implementation."

---

### Pattern 3: Audit Trail Gaps

**What Happened:**

Audit recording logic is split between executors and orchestrator without clear ownership contracts. Some failure paths return metadata but never write audit records.

**Evidence:**

- Finding #1: Coalesce failures return `CoalesceOutcome(failure_reason=...)` but never call `recorder.complete_node_state()`
- Orchestrator comments claim "Failure is recorded by executor" when it isn't

**Why It Matters:**

This violates ELSPETH's core auditability guarantee: "Every decision traceable to source." Silent failures create blind spots in the audit trail.

**CLAUDE.md Violation:**

> "I don't know what happened" is never an acceptable answer for any output.

Missing audit records mean exactly this—operators cannot explain why rows failed.

---

### Pattern 4: Configuration Drift

**What Happened:**

Configuration schema (`RetrySettings`, `TriggerConfig`) evolved independently from engine implementation, creating fields that are ignored or contexts that don't match docs.

**Evidence:**

- Finding #8: `exponential_base` field exists but is never wired to tenacity
- Finding #7: `TriggerConfig` examples show row fields, evaluator provides batch metrics

**Why It Matters:**

Users configure settings expecting behavior changes, but get silent no-ops. This erodes trust in the configuration system.

---

## Action Plan

### Phase 1: Pre-RC-1 Blockers (Fix Before Release)

**Estimated Total Effort:** 12 hours (1.5 developer-days)

| Priority | Finding | Effort | Justification |
|----------|---------|--------|---------------|
| **1** | BatchPendingError (#2) | 30 min | Easiest fix, prevents architectural rot |
| **2** | Orchestrator DB Access (#4) | 1 hour | Use existing recorder methods |
| **3** | Coalesce Audit Gap (#1) | 2 hours | Critical for auditability |
| **4** | Boolean Expression Routing (#3) | 2 hours | Prevents silent routing failures |
| **5** | Protocol vs Base Dispatch (#5) | 3 hours | Unblocks plugin extensibility |
| **6** | Trigger Context (#7) | 2 hours | Prevents user confusion |

**Completion Criteria:**

- [ ] All P1 findings have passing tests
- [ ] No engine imports from `plugins/llm/`
- [ ] No engine imports from `core/landscape/schema`
- [ ] All failure paths create audit records
- [ ] Config validation matches runtime behavior
- [ ] Plugin protocols documented and enforced

---

### Phase 2: Post-RC-1 Improvements (Track as Technical Debt)

**Estimated Total Effort:** 6.5 hours

| Priority | Finding | Effort | GitHub Issue |
|----------|---------|--------|--------------|
| **7** | Payload Storage Encapsulation (#6) | 4 hours | TBD |
| **8** | Retry Exponential Base (#8) | 30 min | TBD |
| **9** | Aggregation Span Drift (#9) | 2 hours | TBD |

---

### Implementation Sequence (Phase 1)

#### Step 1: BatchPendingError Relocation (30 minutes)

```bash
# 1. Create shared exception module
mkdir -p src/elspeth/contracts
cat > src/elspeth/contracts/exceptions.py << 'EOF'
"""Shared exception types for cross-subsystem contracts."""

class BatchPendingError(Exception):
    """Control flow signal: batch submitted but not complete.

    This is NOT an error condition—it tells the engine to
    schedule a retry check later.
    """
    pass
EOF

# 2. Update imports in engine
sed -i 's|from elspeth.plugins.llm.batch_errors|from elspeth.contracts.exceptions|g' \
    src/elspeth/engine/executors.py \
    src/elspeth/engine/orchestrator.py

# 3. Update imports in LLM plugins
sed -i 's|from elspeth.plugins.llm.batch_errors|from elspeth.contracts.exceptions|g' \
    src/elspeth/plugins/llm/*.py

# 4. Delete old file
rm src/elspeth/plugins/llm/batch_errors.py

# 5. Run tests
pytest tests/engine/ tests/plugins/llm/
```

**Verification:** No import errors, all tests pass.

---

#### Step 2: Orchestrator Landscape Boundary Fix (1 hour)

```python
# File: src/elspeth/engine/orchestrator.py

# BEFORE (lines 1487-1492):
from sqlalchemy import select
from elspeth.core.landscape.schema import runs_table

with self._db.engine.connect() as conn:
    run_row = conn.execute(
        select(runs_table.c.source_schema_json)
        .where(runs_table.c.run_id == run_id)
    ).fetchone()

# AFTER:
run = self._recorder.get_run(run_id)
if run is None:
    raise ValueError(f"Run {run_id} not found")
source_schema_json = run.source_schema_json


# BEFORE (lines 1600-1606):
from elspeth.core.landscape.schema import edges_table

with self._db.engine.connect() as conn:
    edges = conn.execute(
        select(edges_table)
        .where(edges_table.c.run_id == run_id)
    ).fetchall()

# AFTER:
edges = self._recorder.get_edges(run_id)
```

**Verification:** Remove all SQLAlchemy imports from orchestrator.py, tests pass.

---

#### Step 3: Coalesce Failure Audit Recording (2 hours)

```python
# File: src/elspeth/engine/coalesce_executor.py

def flush_pending(self, step_in_pipeline: int) -> list[CoalesceOutcome]:
    """Flush pending coalesces at end-of-source."""
    results = []

    for key, pending in list(self._pending.items()):
        settings = self._settings_by_node[key[0]]
        node_id = key[0]

        if settings.policy == "quorum":
            if len(pending.arrived) >= settings.quorum_count:
                outcome = self._execute_merge(...)
                results.append(outcome)
            else:
                # Quorum not met - RECORD FAILURE
                del self._pending[key]

                # NEW: Record node states for tokens that failed to merge
                for token in pending.arrived.values():
                    state = self._recorder.begin_node_state(
                        token_id=token.token_id,
                        node_id=node_id,
                        step_index=step_in_pipeline,
                        input_data=token.row_data,
                    )
                    self._recorder.complete_node_state(
                        state_id=state.state_id,
                        status="failed",
                        output_data={
                            "failure_reason": "quorum_not_met",
                            "quorum_required": settings.quorum_count,
                            "branches_arrived": list(pending.arrived.keys()),
                        },
                        duration_ms=0,
                    )
                    # Record terminal token outcome
                    self._recorder.record_token_outcome(
                        token_id=token.token_id,
                        outcome="QUARANTINED",
                        metadata={"reason": "coalesce_quorum_not_met"},
                    )

                results.append(CoalesceOutcome(...))  # Existing
```

**Test:**
```python
def test_coalesce_quorum_failure_recorded_in_audit():
    """Verify quorum failures create audit records."""
    # Setup coalesce with quorum=3, only 2 branches arrive
    # ...

    # Flush pending at end-of-source
    outcomes = executor.flush_pending(step=2)

    # Assert failure outcome returned
    assert len(outcomes) == 1
    assert outcomes[0].failure_reason == "quorum_not_met"

    # Assert audit records exist
    for token in arrived_tokens:
        node_states = recorder.get_node_states(token.token_id)
        assert any(
            ns.status == "failed" and
            "quorum_not_met" in ns.output_data.get("failure_reason", "")
            for ns in node_states
        )
```

---

### Remaining Steps (Abbreviated)

**Step 4: Boolean Expression Classification** - Tighten `_is_boolean_node()` or add runtime coercion
**Step 5: Protocol-Based Dispatch** - Update processor to accept `TransformProtocol`
**Step 6: Trigger Context Alignment** - Update docs to match batch-only context

---

## Verification Strategy

For each fix:

1. **Read Both Sides** - Understand current coupling by reading evidence from both files
2. **Write Failing Test** - Prove defect exists (e.g., protocol-only transform fails)
3. **Implement Fix** - Minimal change to enforce contract
4. **Verify Test Passes** - Green test proves fix works
5. **Run Full Suite** - Ensure no regressions (`pytest tests/`)
6. **Update Finding** - Mark status as "fixed" in FINDINGS_INDEX.md

---

## Long-Term Recommendations

### 1. Contract Testing Framework

Add tests enforcing subsystem boundaries:

```python
# tests/contracts/test_plugin_protocols.py
def test_protocol_only_transform_executes():
    """Plugins can implement protocols without Base* inheritance."""
    class ProtocolTransform:  # No BaseTransform!
        name = "test"
        is_batch_aware = False
        determinism = Determinism.DETERMINISTIC
        plugin_version = "1.0"

        def process(self, row, ctx):
            return TransformResult.success(row)

    # Should not raise TypeError
    processor.execute(ProtocolTransform(...))


# tests/contracts/test_boundary_enforcement.py
def test_engine_cannot_import_from_plugin_packs():
    """Engine modules must not import from plugins/llm, plugins/azure."""
    import ast

    for engine_file in Path("src/elspeth/engine").glob("*.py"):
        tree = ast.parse(engine_file.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not node.module.startswith("elspeth.plugins.llm")
                assert not node.module.startswith("elspeth.plugins.azure")
```

---

### 2. Dependency Linting (CI Check)

Add pre-commit hook preventing reverse dependencies:

```yaml
# .pre-commit-config.yaml
- repo: local
  hooks:
    - id: check-dependency-direction
      name: Check dependency arrows
      entry: python scripts/check_imports.py
      language: system
      types: [python]
```

```python
# scripts/check_imports.py
"""Enforce dependency direction: core ← engine ← plugins."""

FORBIDDEN_IMPORTS = [
    ("elspeth.engine", "elspeth.plugins.llm"),
    ("elspeth.engine", "elspeth.plugins.azure"),
    ("elspeth.core", "elspeth.engine"),
    ("elspeth.core", "elspeth.plugins"),
]

def check_file(path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for importing, forbidden in FORBIDDEN_IMPORTS:
                if path.match(f"**/{importing.replace('.', '/')}/**"):
                    if node.module and node.module.startswith(forbidden):
                        raise ValueError(
                            f"{path}: Cannot import {node.module} from {importing}"
                        )
```

---

### 3. Boundary Documentation

Create `docs/contracts/subsystem-boundaries.md`:

```markdown
# Subsystem Boundary Contracts

## Dependency Graph

```
┌─────────────────┐
│   Contracts     │  ← Shared types, protocols, exceptions
└────────┬────────┘
         │
    ┌────▼─────┐
    │   Core   │  ← Landscape, Config, Canonical
    └────┬─────┘
         │
    ┌────▼─────┐
    │  Engine  │  ← Orchestrator, Processor, Executors
    └────┬─────┘
         │
    ┌────▼─────┐
    │ Plugins  │  ← Sources, Transforms, Sinks, LLM pack, Azure pack
    └──────────┘
```

**Arrows point UP** - dependencies flow upward, knowledge flows downward.

## Import Rules

| From ↓ / To → | Contracts | Core | Engine | Plugins |
|---------------|-----------|------|--------|---------|
| **Contracts** | ✅ | ❌ | ❌ | ❌ |
| **Core** | ✅ | ✅ | ❌ | ❌ |
| **Engine** | ✅ | ✅ | ✅ | ❌ |
| **Plugins** | ✅ | ✅ | ✅ | ✅ |

✅ = Allowed
❌ = Forbidden (reverse dependency)

## Public Facades

### Landscape (Core)

**Public Interface:** `LandscapeRecorder`

**Forbidden:** Direct access to `schema.py` tables, SQLAlchemy imports outside `landscape/`

**Methods:**
- `create_run()`
- `get_run()` / `get_runs()`
- `get_edges()`
- `create_row()` / `get_row()`
- `begin_node_state()` / `complete_node_state()`
- `record_token_outcome()`

### Plugin System

**Public Interface:** `TransformProtocol`, `SourceProtocol`, `SinkProtocol`, `GateProtocol`

**Dispatch:** Must accept protocol implementations (not just `Base*` subclasses)

**Shared Exceptions:** Use `contracts.exceptions`, never plugin-specific exceptions in engine

## Enforcement

- **Pre-commit hook:** `scripts/check_imports.py`
- **CI check:** `make check-boundaries`
- **Contract tests:** `tests/contracts/`
```

---

## Appendix: Evidence Details

### Coalesce Timeout Issue (Secondary Finding in coalesce_executor.py.md)

**Status:** P2 (downgraded from P1)
**Finding:** `check_timeouts()` method implemented but never called by orchestrator

**Evidence:**
```python
# src/elspeth/engine/coalesce_executor.py:303-311
def check_timeouts(
    self,
    coalesce_name: str,
    step_in_pipeline: int,
) -> list[CoalesceOutcome]:
    """Check for timed-out pending coalesces and merge them.

    For best_effort policy, merges whatever has arrived when timeout expires.
```

**Impact:** `best_effort` and `quorum` timeout policies never trigger during execution

**Fix:** Add periodic `check_timeouts()` calls in orchestrator main loop

**Effort:** 2 hours (orchestrator loop + timeout tests)

---

### Batch-Aware Transform Type Signature Mismatch (Secondary Finding in executors.py.md)

**Status:** P2
**Finding:** `TransformProtocol.process()` signature is `row: dict`, but executor passes `list[dict]` for batches

**Evidence:**
```python
# src/elspeth/engine/executors.py:946
result = transform.process(buffered_rows, ctx)  # type: ignore[arg-type]

# src/elspeth/plugins/protocols.py:173
def process(self, row: dict[str, Any], ctx: "PluginContext") -> TransformResult:
```

**Impact:** Type checker disabled with `# type: ignore`, hiding signature mismatch

**Fix:** Update `TransformProtocol` to `row: dict[str, Any] | list[dict[str, Any]]` or create `BatchTransformProtocol`

---

### CSV Export Sink Config Assumption (Secondary Finding in orchestrator.py.md)

**Status:** P1
**Finding:** Export assumes all sinks have `.config["path"]`, but `SinkProtocol` doesn't require `config` attribute

**Evidence:**
```python
# src/elspeth/engine/orchestrator.py:1217-1221
if "path" not in sink.config:
    raise ValueError(
        f"CSV export requires file-based sink with 'path' in config"
    )
artifact_path: str = sink.config["path"]
```

**Impact:** Protocol-only sinks without `config` will crash at export time

**Fix:** Add `config: dict[str, Any]` to `SinkProtocol` or create `FileSinkProtocol`

---

### Route Destination Magic Strings (Secondary Finding in orchestrator.py.md)

**Status:** P2
**Finding:** Route destinations use string sentinels `"continue"` and `"fork"` instead of typed enum

**Evidence:**
```python
# src/elspeth/engine/orchestrator.py:264-271
if destination == "continue":
    continue
if destination == "fork":
    continue

# src/elspeth/core/dag.py:466-472
if target == "continue":
    graph._route_resolution_map[(gid, route_label)] = "continue"
elif target == "fork":
    graph._route_resolution_map[(gid, route_label)] = "fork"
```

**Impact:** Typos cause runtime failures, no type safety

**Fix:** Create `RouteDestination` enum or `Literal["continue", "fork"] | str` union type

---

### TriggerEvaluator Private Field Mutation (Secondary Finding in triggers.py.md)

**Status:** P1
**Finding:** Checkpoint restore mutates `_first_accept_time` private field directly

**Evidence:**
```python
# src/elspeth/engine/executors.py:1258-1265
elapsed_seconds = node_state.get("elapsed_age_seconds", 0.0)
if elapsed_seconds > 0.0:
    # Adjust timer: make it think first accept was N seconds ago
    evaluator._first_accept_time = time.monotonic() - elapsed_seconds
```

**Impact:** Fragile coupling, breaks if `TriggerEvaluator` internal changes

**Fix:** Add public `restore_elapsed_seconds(float)` method

---

### Retry Result-Level Retryability Ignored (Secondary Finding in retry.py.md)

**Status:** P2
**Finding:** Plugins return `TransformResult.error(retryable=True)` but only exceptions trigger retry

**Evidence:**
```python
# src/elspeth/plugins/llm/base.py:234-254
except RateLimitError as e:
    return TransformResult.error(
        {"reason": "rate_limited"},
        retryable=True,  # ← This is ignored!
    )

# src/elspeth/engine/retry.py:128-161
def execute_with_retry(
    self,
    operation: Callable[[], T],
    *,
    is_retryable: Callable[[BaseException], bool],  # ← Exception-only
```

**Impact:** Rate limits and transient LLM failures route to error sinks instead of retrying

**Fix:** Extend retry logic to check `result.retryable` or convert to exception

---

## Conclusion

This analysis reveals a **fundamentally sound architecture** with **fixable boundary issues**. The defects aren't symptoms of "bad code"—they're evidence that:

1. **Rapid evolution** (Phase 1 → Phase 6) created contract drift
2. **Some boundaries weren't formalized** as design evolved
3. **Strong quality discipline** exists (running this analysis proves it)

**Final Assessment:**

- **Total P1 Effort:** ~12 hours (6 issues × 2 hours average)
- **ROI:** Prevents audit trail gaps, architectural rot, and plugin contract violations
- **Risk of Shipping Without Fixes:** Medium-High
  - Customer discovers missing audit records
  - Plugin developers hit undocumented constraints
  - Landscape schema changes break engine

**Recommendation:** **Fix findings #1-6 before RC-1 release.** Track findings #7-9 as technical debt with GitHub issues.

---

**Report Prepared By:** ELSPETH Quality Engineering
**Contact:** See `docs/quality-audit/findings-integration/` for detailed evidence
**Archive:** `docs/quality-audit/findings-integration.tar.gz`

---

*This report represents a snapshot of integration seam quality as of commit 708ea26. Findings may become stale as code evolves. Re-run analysis after major architectural changes.*
