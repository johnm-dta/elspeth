# ELSPETH Architecture Quality Assessment

**Date:** 2026-01-27
**Assessor:** Claude Opus 4.5 (Architecture Critic)
**Scope:** RC-1 codebase quality evaluation
**Method:** Evidence-based critique against documented standards

---

## Executive Summary

ELSPETH claims to be an "auditable SDA pipeline framework for high-stakes accountability." This assessment evaluates whether the architecture delivers on that claim.

**Overall Quality Score:** 2.5 / 5

**Verdict:** The architecture has strong foundations but contains multiple High-severity issues that contradict documented design principles. The codebase is not production-ready for an audit system.

| Category | Score | Critical | High | Medium |
|----------|-------|----------|------|--------|
| Architectural Coherence | 2/5 | 1 | 4 | 3 |
| Separation of Concerns | 3/5 | 0 | 2 | 2 |
| SOLID Principles | 2/5 | 0 | 3 | 2 |
| Pattern Consistency | 3/5 | 1 | 2 | 3 |
| Production Readiness | 2/5 | 2 | 3 | 4 |

---

## 1. Architectural Coherence

**Question:** Does the implementation match the documented architecture?

### 1.1 Core Value Proposition Undelivered - Critical

**Claim (CLAUDE.md:15-20):**
> "Every decision must be traceable to source data, configuration, and code version"
> "The Landscape audit trail is the source of truth, not logs or metrics"

**Reality:**
The `explain` command - the primary interface for audit trail access - returns "not_implemented".

**Evidence:**
- `/home/john/elspeth-rapid/src/elspeth/cli.py:336-348`
```python
if json_output:
    result = {
        ...
        "status": "not_implemented",
        ...
    }
    raise typer.Exit(2)
```

**Impact:** An audit system that cannot explain its decisions is not an audit system. Users cannot verify traceability claims. Auditors cannot investigate outcomes.

**Severity:** Critical

---

### 1.2 Rate Limiting Subsystem Completely Disconnected - Critical

**Documented (CLAUDE.md:432):**
> "Rate Limiting | pyrate-limiter | Custom leaky buckets"

**Reality:**
The rate limiting subsystem exists (`/home/john/elspeth-rapid/src/elspeth/core/rate_limit/`) but is never imported or used by the engine.

**Evidence:**
```bash
$ grep -r "from.*rate_limit" src/elspeth/engine/
# No matches
$ grep -r "RateLimitRegistry" src/elspeth/engine/
# No matches
```

The registry at `/home/john/elspeth-rapid/src/elspeth/core/rate_limit/registry.py` is complete and functional (122 lines), but disconnected from the LLM transforms that need it.

**Impact:** Azure OpenAI rate limits will cause cascading failures. Users configuring rate limits in settings.yaml get no actual protection.

**Severity:** Critical

---

### 1.3 Coalesce Timeout Never Fires During Processing - High

**Expected behavior:** Coalesce points with timeout policies should time out and proceed when branches fail to arrive.

**Reality:**
`check_timeouts()` is defined in `/home/john/elspeth-rapid/src/elspeth/engine/coalesce_executor.py:371-440` but the processor never calls it during normal row processing.

**Evidence:**
```bash
$ grep "check_timeouts" src/elspeth/engine/processor.py
# No matches
```

Timeouts only fire at end-of-source during `flush_pending()`, not during processing.

**Impact:** A pipeline with quorum policy expecting 3 branches where one branch fails silently will hang indefinitely instead of timing out.

**Severity:** High

---

### 1.4 Checkpoints Table is Dead Code - High

**Schema defines:**
- `/home/john/elspeth-rapid/src/elspeth/core/landscape/schema.py:373-400` - `checkpoints_table`

**Missing:**
- No `create_checkpoint()` method in Recorder
- No `get_latest_checkpoint()` method
- No `CheckpointRepository` class

**Evidence:**
```bash
$ grep -r "create_checkpoint" src/elspeth/core/landscape/
# No matches
$ grep -r "get_latest_checkpoint" src/elspeth/core/landscape/
# No matches
```

**Impact:** Resume functionality cannot work properly without checkpoint management. The schema exists but the feature does not.

**Severity:** High

---

### 1.5 OpenTelemetry Claimed But Not Implemented - High

**Claim (CLAUDE.md:430):**
> "Observability | OpenTelemetry + Jaeger | Custom tracing (immediate visualization)"

**Reality:**
- No tracer configuration in core
- No span creation utilities
- No trace context propagation
- `/home/john/elspeth-rapid/src/elspeth/core/logging.py:3-6` docstring says "complements OpenTelemetry spans" but there are no spans to complement

**Impact:** Operators have no distributed tracing for debugging production issues. The "immediate visualization" claimed doesn't exist.

**Severity:** High

---

### 1.6 TUI Widgets Exist But Aren't Wired - Medium

**Exists:**
- `src/elspeth/tui/screens/explain_screen.py` (314 LOC)
- `src/elspeth/tui/widgets/lineage_tree.py` (198 LOC)
- `src/elspeth/tui/widgets/node_detail.py` (166 LOC)

**Actually used:**
```python
yield Static("Lineage Tree (placeholder)", id=WidgetIDs.LINEAGE_TREE)
yield Static("Detail Panel (placeholder)", id=WidgetIDs.DETAIL_PANEL)
```

**Impact:** 678 lines of code that do nothing. The TUI exists as a scaffold, not a feature.

**Severity:** Medium

---

## 2. Separation of Concerns

### 2.1 Protocol/Base Class Duality - High

**Problem:** Every plugin type has both a Protocol and a Base class that must stay synchronized:

| Protocol | Base Class | Sync Issue? |
|----------|------------|-------------|
| `SourceProtocol` | `BaseSource` | Yes - `_on_validation_failure` docs differ |
| `TransformProtocol` | `BaseTransform` | Yes - `_on_error` docs differ |
| `GateProtocol` | `BaseGate` | No |
| `SinkProtocol` | `BaseSink` | No |
| `CoalesceProtocol` | **None** | Missing base class entirely |

**Evidence:**
- `/home/john/elspeth-rapid/src/elspeth/plugins/protocols.py:171` - `_on_error` docs say "Transforms extending TransformDataConfig set this from config"
- `/home/john/elspeth-rapid/src/elspeth/plugins/base.py:66-67` - `_on_error` docs say "Transforms extending TransformDataConfig override this from config"

Subtle difference: "set this from config" vs "override this from config". This is documentation drift that will compound over time.

**Impact:** Maintenance burden doubles. Changes require updating two files. Bugs emerge when they drift.

**Severity:** High

---

### 2.2 Duplicate PayloadStore Protocols - High

**Two different Protocol definitions for the same abstraction:**

1. `/home/john/elspeth-rapid/src/elspeth/core/payload_store.py:28-83` - Full protocol with 4 methods
2. `/home/john/elspeth-rapid/src/elspeth/core/retention/purge.py:28-41` - "Minimal" protocol with 2 methods

```python
# purge.py:28-31
class PayloadStoreProtocol(Protocol):
    """Protocol for PayloadStore to avoid circular imports.
    Defines the minimal interface required by PurgeManager."""
```

**Impact:** Interface contract is fragmented. Changes to PayloadStore don't update the "minimal" protocol. Type checking sees different protocols.

**Severity:** High

---

### 2.3 Layer Violations - Medium

**Expected layers:**
```
CLI -> Engine -> Core -> Landscape -> Contracts
```

**Violations found:**
1. `contracts/results.py` imports `MaxRetriesExceeded` from `engine/retry.py`
2. `core/config.py` imports `ExpressionParser` from `engine/expression_parser.py`

**Impact:** Contracts should be the foundation layer with no dependencies. Upward imports create import cycles waiting to happen.

**Severity:** Medium

---

### 2.4 Hardcoded Plugin Lookup in Validation - Medium

**Location:** `/home/john/elspeth-rapid/src/elspeth/plugins/validation.py:85-109`

The validation module has hardcoded lookup tables for plugin types instead of using the plugin registry.

**Impact:** Adding a new plugin requires updating the validation code. The registry pattern is bypassed.

**Severity:** Medium

---

## 3. SOLID Principles

### 3.1 Liskov Substitution Violation in LLM Transforms - High

**Problem:** `AzureOpenAITransform` implements `BaseTransform` but rejects `process()`:

**Evidence:** `/home/john/elspeth-rapid/src/elspeth/plugins/llm/azure.py:228-243`
```python
def process(self, row: dict, ctx: PluginContext) -> TransformResult:
    """Not supported - use accept() for row-level pipelining.
    ...
    Raises:
        NotImplementedError: Always, directing callers to use accept()
    """
    raise NotImplementedError(...)
```

A subclass that unconditionally rejects a parent method violates LSP. Code expecting a `BaseTransform` will crash when given an `AzureOpenAITransform`.

**Impact:** Polymorphism is broken. Generic transform handling cannot work.

**Severity:** High

---

### 3.2 Interface Segregation Violation: Missing BaseCoalesce - High

**Problem:** `CoalesceProtocol` exists but there is no `BaseCoalesce` class.

**Evidence:** `/home/john/elspeth-rapid/src/elspeth/plugins/base.py:201-207`
```python
# NOTE: BaseAggregation was DELETED in aggregation structural cleanup.
# ...
# Use is_batch_aware=True on BaseTransform for batch processing.
```

This comment exists for `BaseAggregation`. No equivalent comment for `BaseCoalesce`. The protocol exists, but implementers have no base class.

**Impact:** Coalesce implementations must implement the entire protocol from scratch. No shared lifecycle hooks, no default implementations.

**Severity:** High

---

### 3.3 Single Responsibility Violation: CLI Massive Duplication - Medium

**Location:** `/home/john/elspeth-rapid/src/elspeth/cli.py:471-594` and `683-806`

123 lines duplicated verbatim between `_execute_pipeline()` and `_execute_pipeline_with_instances()`. Same duplication in `_execute_resume_with_instances()`.

**Impact:** Bug fixes must be applied in 3+ places. Divergence is inevitable.

**Severity:** Medium

---

### 3.4 Open/Closed Violation: Hardcoded Date Check - Medium

**Location:** `/home/john/elspeth-rapid/src/elspeth/core/checkpoint/manager.py:202-233`

```python
cutoff_date = datetime(2026, 1, 24, tzinfo=UTC)
```

Hardcoded date for node ID format changes. Future format changes require more date checks.

**Impact:** Every schema change requires a new hardcoded date. Should use version field.

**Severity:** Medium

---

## 4. Pattern Consistency

### 4.1 Three-Tier Trust Model Violation in azure_batch.py - Critical

**CLAUDE.md mandates (lines 70-80):**
> "Validate at the boundary, coerce where possible, record what we got"
> "IMMEDIATELY validate at the boundary - don't let their data travel"

**Reality:** `/home/john/elspeth-rapid/src/elspeth/plugins/llm/azure_batch.py:768-774`
```python
response = result.get("response", {})
body = response.get("body", {})
choices = body.get("choices", [])
if choices:
    content = choices[0].get("message", {}).get("content", "")
```

This is a `.get()` chain with empty dict/list fallbacks - exactly what CLAUDE.md prohibits. When Azure returns unexpected structure, users get cryptic "no_choices_in_response" instead of "malformed API response: missing 'body' key".

**Evidence directly contradicts CLAUDE.md:656-658:**
> "Do not use .get(), getattr(), hasattr(), isinstance(), or silent exception handling to suppress errors"

**Impact:** Debug information is lost. The audit trail records "no_choices_in_response" but not why. Auditors cannot determine if it was API failure, network corruption, or schema change.

**Severity:** Critical

---

### 4.2 Silent JSON Fallback in HTTP Client - High

**Location:** `/home/john/elspeth-rapid/src/elspeth/plugins/clients/http.py:164-169`
```python
if "application/json" in content_type:
    try:
        response_body = response.json()
    except Exception:  # Too broad!
        response_body = response.text  # Silent fallback
```

**Impact:** Content-Type says JSON but body is HTML error page. Downstream transforms receive string instead of dict. Error message is useless for debugging.

**Severity:** High

---

### 4.3 Test Path Integrity Violations - High

**CLAUDE.md:490-542 explicitly documents:**
> "Never bypass production code paths in tests."
> "BUG-LINEAGE-01 hid for weeks because tests manually built graphs"

**Evidence:**
```bash
$ grep -r "graph\._" tests/engine/ --include="*.py" | grep -v "graph\._graph" | wc -l
62
```

62 instances of tests directly accessing private graph attributes instead of using `ExecutionGraph.from_plugin_instances()`.

**Impact:** Same bug pattern that caused BUG-LINEAGE-01 is still present. Production bugs will hide in untested code paths.

**Severity:** High

---

### 4.4 Memory Leak in CoalesceExecutor - Medium

**Location:** `/home/john/elspeth-rapid/src/elspeth/engine/coalesce_executor.py:172-199`
```python
self._completed_keys.add(key)  # Grows unbounded
```

Cleared only in `flush_pending()`, not after each normal merge.

**Impact:** Long-running pipelines with many coalesce operations will OOM.

**Severity:** Medium

---

### 4.5 N+1 Query Pattern in Exporter - High

**Location:** `/home/john/elspeth-rapid/src/elspeth/core/landscape/exporter.py:199-329`
```python
for row in self._recorder.get_rows(run_id):        # 1 query
    for token in self._recorder.get_tokens(...):   # 1000 queries
        for state in self._recorder.get_node_states_for_token(...):  # 2000+ queries
            for call in self._recorder.get_calls(...):  # 6000+ queries
```

For 1000 rows: 21,001+ queries minimum.

**Impact:** Compliance audits requiring full export are impractical. Export of large runs could take hours.

**Severity:** High

---

## 5. Production Readiness

### 5.1 No Graceful Shutdown - High

**Evidence:**
```bash
$ grep -r "shutdown" src/elspeth/engine/
# No matches for shutdown signal handling
```

**Impact:** SIGTERM kills pipeline mid-row. No checkpoint created. Resume cannot recover cleanly.

**Severity:** High

---

### 5.2 No Circuit Breaker - High

The retry subsystem (`/home/john/elspeth-rapid/src/elspeth/engine/retry.py`) implements exponential backoff but no circuit breaker pattern.

**Impact:** If an external service is down, every row will retry to exhaustion before failing. 10,000 rows means 10,000 retry sequences against a dead endpoint.

**Severity:** High

---

### 5.3 In-Memory Call Index Counter - High

**Location:** `/home/john/elspeth-rapid/src/elspeth/core/landscape/recorder.py:1750-1787`

The call index counter is in-memory only. Resume starts at 0 regardless of existing calls.

**Impact:** Resume after crash may create duplicate call indices, violating unique constraints or causing data integrity issues.

**Severity:** High

---

### 5.4 Missing Database Migration CLI - Medium

Alembic migrations exist but no `elspeth db migrate` command.

**Impact:** Users must use Alembic directly. Non-standard deployment. Documentation gap.

**Severity:** Medium

---

### 5.5 Missing CLI Commands Documented in CLAUDE.md - Medium

**Documented (CLAUDE.md:399-407):**
```bash
elspeth explain --run latest --row 42
elspeth validate --settings settings.yaml
elspeth plugins list
```

**Status:**
- `explain` - returns "not_implemented"
- `validate` - exists
- `plugins list` - exists
- `status` - mentioned in CLAUDE.md but missing
- `export` - mentioned in discovery but missing

**Impact:** Documentation promises features that don't work.

**Severity:** Medium

---

### 5.6 BatchStatus Accepts Raw String - Low

**Location:** `/home/john/elspeth-rapid/src/elspeth/core/landscape/recorder.py:1319-1348`

BatchStatus accepts raw string, no enum validation. Invalid status values can enter the audit trail.

**Impact:** Audit trail integrity risk - garbage status values possible.

**Severity:** Low

---

## Confidence Assessment

**Overall Confidence:** High

**Evidence Quality:**
- All findings backed by specific file:line references
- Grep commands verified absence of expected code
- Code reviewed directly, not inferred from documentation

**Information Gaps:**
- Did not execute performance benchmarks
- Did not test under actual Azure API rate limits
- Did not verify all 62 test path violations individually
- Did not analyze Alembic migration correctness

---

## Risk Assessment

**Highest Risk Issues for Production:**

1. **Rate limiting disconnected** - Azure will rate-limit production traffic with no protection
2. **Explain command broken** - Cannot demonstrate audit capability to regulators
3. **Coalesce timeout never fires** - Pipelines will hang on branch failures
4. **N+1 export queries** - Compliance audits unusable at scale

---

## Caveats

1. This assessment is based on code review, not runtime testing
2. Some issues may have mitigations not visible in source code
3. RC-1 status implies known incomplete features
4. Assessment does not cover security vulnerabilities (separate review needed)

---

## Summary: Issues by Severity

| Severity | Count | Key Issues |
|----------|-------|------------|
| Critical | 4 | Rate limiting disconnected, explain broken, Trust Model violation, N/A |
| High | 12 | Coalesce timeout, LSP violation, protocol duality, N+1 queries, no shutdown |
| Medium | 10 | Layer violations, hardcoded dates, CLI duplication, memory leak |
| Low | 1 | BatchStatus validation |

---

## Recommendation

This codebase should NOT be released as RC-1 for an audit system. The core value proposition (auditability via explain) is not implemented. Critical subsystems (rate limiting) are disconnected.

Priority fixes before any production use:
1. Connect rate limiting to LLM transforms
2. Implement explain command
3. Fix coalesce timeout firing
4. Fix azure_batch.py Trust Model violations
5. Add graceful shutdown handling

The architecture is sound. The implementation is incomplete. Label this as Alpha, not RC-1.
