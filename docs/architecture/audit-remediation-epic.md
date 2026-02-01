# EPIC: Audit Trail Completeness Remediation

> **North Star Document for Landscape Audit System Hardening**
>
> Status: ACTIVE
> Priority: P1 - Release Blocker
> Created: 2026-01-31
> Owner: Core Team

---

## Executive Summary

A comprehensive 4-perspective review of ELSPETH's audit entry points architecture identified **critical gaps** that violate the core principle: *"Every token reaches exactly one terminal state with complete audit trail."*

The architecture is **fundamentally sound** (clean 4-layer separation, XOR constraints, atomic lineage operations) but has **systematic exception-path vulnerabilities** where:
- Audit records can be incomplete under failure conditions
- Tokens can exist without terminal states
- The audit trail can "lie" about row processing status

This epic defines the remediation work required to achieve **audit trail integrity under adversarial conditions**.

---

## Table of Contents

1. [Review Panel Findings](#review-panel-findings)
2. [Critical Vulnerabilities](#critical-vulnerabilities)
3. [Gap Inventory](#gap-inventory)
4. [Test Coverage Gaps](#test-coverage-gaps)
5. [Implementation Plan](#implementation-plan)
6. [Acceptance Criteria](#acceptance-criteria)
7. [Risk Matrix](#risk-matrix)

---

## Review Panel Findings

### Panel Composition

| Reviewer | Domain | Verdict |
|----------|--------|---------|
| `axiom-system-architect` | Architecture & Design Patterns | REQUEST CHANGES |
| `axiom-python-engineering` | Code Quality & Concurrency | APPROVE WITH OBSERVATIONS |
| `ordis-quality-engineering` | Test Coverage & Edge Cases | REQUEST CHANGES |
| `yzmir-systems-thinking` | Systemic Risks & Patterns | REQUEST CHANGES |

### Consensus Issues (All 4 Reviewers Agreed)

| Issue | Impact | Priority |
|-------|--------|----------|
| XOR invariant validation missing | Lineage corruption | **P1** |
| Sink flush exception handling | Audit incomplete | **P1** |
| Checkpoint durability violation | Data loss on crash | **P1** |
| Crash scenario test coverage | Undetected regressions | **P1** |

### Key Statistics

- **65** distinct audit entry points across 4 layers
- **17** database tables receiving audit entries
- **8** missing audit entry points identified
- **12** critical test coverage gaps
- **17** historical bugs following same pattern (infrastructure exists, code path missing)

---

## Critical Vulnerabilities

### Vulnerability 1: Checkpoint Pre-Durability

**Severity: CRITICAL**
**Location:** `orchestrator.py:647-658`

**Problem:** Checkpoints are created BEFORE sink writes. If process crashes between checkpoint and sink write, recovery skips unwritten rows.

```
Timeline:
1. Token reaches sink
2. Processor records COMPLETED
3. _maybe_checkpoint() called     ← Checkpoint persisted
4. ═══════════════════════════════ CRASH WINDOW
5. sink.write() executes          ← Never happens on crash
6. Artifact created
```

**Impact:** Rows 8-10 of a 100-row batch could be silently lost. Audit trail shows COMPLETED but no artifact exists.

**Evidence:** P0-2026-01-19 bug documented this as "design decision needed."

**Fix:** Move checkpoint creation to AFTER `sink_executor.write()` succeeds, or maintain separate "sink write complete" flag.

---

### Vulnerability 2: LandscapeRecorder Single Point of Failure

**Severity: HIGH**
**Location:** `core/landscape/recorder.py` (all 60+ methods)

**Problem:** No fallback, no retry, no timeout when recorder methods fail. Exception propagates but audit state is inconsistent.

**Failure Scenarios:**

| Failure Point | Records Lost | Recovery Impact |
|---------------|--------------|-----------------|
| `begin_run()` crashes | Run never created | Resume fails |
| `create_row()` fails | Row data missing | Lineage incomplete |
| `begin_node_state()` fails | Processing boundary missing | Cannot explain state |
| `record_token_outcome()` fails | Terminal state missing | Recovery loops infinitely |
| `complete_batch()` never called | Batch stuck in DRAFT | Aggregation unrecoverable |

**Evidence:** 17 closed P0-P1 bugs follow pattern "infrastructure exists, code path missing."

**Fix:** Wrap all recorder calls with explicit exception handling. Fail fast if audit write fails (don't silently continue).

---

### Vulnerability 3: Sink Flush Exception Path

**Severity: HIGH**
**Location:** `executors.py:1703-1705`

**Problem:** `sink.flush()` is called without try/except. If flush raises, node_states remain OPEN permanently.

```python
# Current code (WRONG)
for token, state in states:
    self._recorder.complete_node_state(..., status=COMPLETED)

sink.flush()  # ← Raises here, but states already marked COMPLETED

# States show COMPLETED, but data not durable
```

**Impact:** Audit trail shows COMPLETED tokens, but sink data may be partially written or corrupt.

**Fix:** Wrap `sink.flush()` in try/except, complete node_states as FAILED on error.

---

### Vulnerability 4: XOR Invariant Not Validated Pre-Insert

**Severity: HIGH**
**Location:** `recorder.py:record_call()`, `recorder.py:record_operation_call()`

**Problem:** Calls must have exactly one parent (`state_id` XOR `operation_id`). Database constraint catches this at commit, but no runtime validation exists.

```python
# Current code (NO VALIDATION)
def record_call(self, state_id: str | None, operation_id: str | None, ...):
    # Could be: both None ✗
    # Could be: both set ✗
    # Only caught at database commit
```

**Impact:** Wrong parent attribution, lineage broken, audit queries return incorrect results.

**Evidence:** Zero tests for XOR invariant under any conditions.

**Fix:** Add pre-insert validation:
```python
if not ((state_id is None) ^ (operation_id is None)):
    raise AuditIntegrityError("XOR violation: exactly one of state_id/operation_id required")
```

---

### Vulnerability 5: Coalesce Failure Tokens Unrecorded

**Severity: MEDIUM**
**Location:** `coalesce_executor.py:421-449`

**Problem:** When coalesce fails (quorum not met, incomplete branches), tokens are deleted from pending dict without any audit record.

```python
# Current code (SILENT DROP)
if quorum_not_met:
    del self._pending[key]  # ← Tokens vanish from audit trail
    return CoalesceOutcome(failure_reason="quorum_not_met", consumed_tokens=[])
```

**Impact:** Audit trail cannot answer "Which tokens waited at coalesce X?" or "What data was in missing branches?"

**Fix:** Record node_states for held tokens before deleting from pending.

---

### Vulnerability 6: Tier 1 Validation Gaps

**Severity: HIGH**
**Location:** `repositories.py` (multiple load methods)

**Problem:** Falsy checks allow invalid values to pass silently, violating CLAUDE.md Tier 1 principle.

```python
# WRONG: Falsy check allows "" to bypass enum validation
export_status=ExportStatus(row.export_status) if row.export_status else None

# WRONG: Coerces any non-1 value to False
is_terminal = row.is_terminal == 1  # NULL, 2, 99 all become False
```

**Impact:** Silent corruption of audit data. Auditors get "confident wrong answers."

**Fix:** Apply `is not None` checks. Add `__post_init__` validation to dataclasses.

---

### Vulnerability 7: Abandoned Run Detection

**Severity: MEDIUM**
**Location:** `orchestrator.py` (run lifecycle)

**Problem:** Runs can remain in RUNNING status indefinitely if process killed before catch clause. No timeout, no ABANDONED state.

**Cascade Effect:**
1. Run A crashes in RUNNING state
2. Checkpoints for Run A persist (only deleted on COMPLETED)
3. Run B resumed with same run_id
4. Recovery finds stale Run A checkpoint
5. Resume from wrong position → data duplication

**Fix:** Add ABANDONED run status, timeout for RUNNING, cleanup stale checkpoints on resume validation.

---

## Gap Inventory

### Missing Audit Entry Points

| # | Gap | Location | Severity | Category |
|---|-----|----------|----------|----------|
| 1 | Sink flush failure leaves OPEN states | `executors.py:1703-1705` | P1 | Exception |
| 2 | Quarantine outcome recorded pre-durability | `orchestrator.py:1185-1191` | P1 | Ordering |
| 3 | Config gate MissingEdgeError leaves OPEN state | `executors.py:843-879` | P2 | Exception |
| 4 | Coalesce failure tokens unrecorded | `coalesce_executor.py:421-449` | P2 | Silent drop |
| 5 | Resume operation not marked in audit | `orchestrator.py:1900-2060` | P2 | Traceability |
| 6 | Aggregation timeout idle flush missing | `orchestrator.py:1340-1345` | P2 | Completeness |
| 7 | Expression parser errors unhandled | `expression_parser.py` + `executors.py:726` | P2 | Exception |
| 8 | Buffered tokens on crash termination | `processor.py` aggregation | P3 | Shutdown |

### Tier 1 Validation Failures

| # | Issue | Location | Fix Pattern |
|---|-------|----------|-------------|
| 1 | `is_terminal` coercion | `repositories.py:478` | Explicit 0/1 check, crash on other |
| 2 | `export_status` falsy check | `RunRepository.load()` | `is not None` check |
| 3 | Missing `__post_init__` | Call, RoutingEvent, Batch, TokenOutcome | Add enum/type validation |
| 4 | Endpoint NULL handling | `clients/http.py` | Crash on NULL base_url |

### Metadata Never Persisted

| # | Data | Computed In | Should Persist To | Status |
|---|------|-------------|-------------------|--------|
| 1 | Coalesce metadata (policy, timing, branches) | `coalesce_executor.py` | `node_states.context_after` | OPEN |
| 2 | Batch trigger reason | `AggregationExecutor` | `batches.trigger_type` | PARTIAL |
| 3 | Resume marker | `orchestrator.resume()` | New event type | MISSING |

---

## Test Coverage Gaps

### Zero Coverage (Critical)

| # | Scenario | Risk | Priority |
|---|----------|------|----------|
| 1 | Crash during `token_outcome` write after `node_state` completes | Orphaned states | P0 |
| 2 | XOR invariant under concurrency | Lineage corruption | P0 |
| 3 | Database connection failure during transaction | Audit loss | P0 |
| 4 | Payload store failure during row creation | Hash without content | P0 |
| 5 | Fork atomicity (parent outcome + all children) | Orphaned tokens | P0 |

### Partial Coverage (High Priority)

| # | Scenario | Current State | Gap |
|---|----------|---------------|-----|
| 6 | Coalesce timeout with missing branch | Success path only | Timeout scenarios |
| 7 | Multiple outcomes for same token | Audit sweep detection | No explicit test |
| 8 | Partial expand recovery | Thin coverage | Orphaned children |
| 9 | Resume with corrupted checkpoint | Version validation only | Missing token scenario |

### Missing Assertions in Code

| Location | Missing Assertion |
|----------|-------------------|
| `recorder.py:fork_token()` | `assert len(children) == len(branches)` |
| `recorder.py:record_call()` | `assert (state_id is None) != (operation_id is None)` |
| `coalesce_executor.py` | `assert all(token recorded) before del pending` |

---

## Implementation Plan

### Phase 1: Critical Fixes (P1 - Blocking RC-2)

**Estimated Effort: 3-4 days**

#### 1.1 Sink Flush Exception Handling
- File: `src/elspeth/engine/executors.py`
- Lines: 1698-1715
- Change: Wrap `sink.flush()` in try/except
- On error: Complete all node_states as FAILED, re-raise
- Test: `test_sink_flush_failure_completes_states_as_failed`

#### 1.2 Checkpoint Durability Fix
- File: `src/elspeth/engine/orchestrator.py`
- Lines: 647-658
- Change: Move `_maybe_checkpoint()` call to AFTER sink writes complete
- Alternative: Add `sink_write_complete` flag to checkpoint data
- Test: `test_checkpoint_not_created_until_sink_write_succeeds`

#### 1.3 XOR Invariant Validation
- File: `src/elspeth/core/landscape/recorder.py`
- Methods: `record_call()`, `record_operation_call()`
- Change: Add pre-insert validation with `AuditIntegrityError`
- Test: `test_xor_violation_raises_immediately`

#### 1.4 Tier 1 Validation Hardening
- File: `src/elspeth/core/landscape/repositories.py`
- Change: Replace falsy checks with `is not None`
- Change: Add `__post_init__` to Call, RoutingEvent, Batch, TokenOutcome
- Test: `test_invalid_is_terminal_crashes`, `test_invalid_enum_crashes`

### Phase 2: Critical Tests (P1 - Blocking RC-2)

**Estimated Effort: 2-3 days**

#### 2.1 Crash Scenario Tests
- File: `tests/engine/test_crash_scenarios.py` (NEW)
- Tests:
  - `test_crash_during_token_outcome_write`
  - `test_crash_during_fork_between_parent_and_children`
  - `test_crash_during_coalesce_merge`
  - `test_crash_during_expand_token`
  - `test_crash_during_batch_flush`

#### 2.2 XOR Invariant Tests
- File: `tests/core/landscape/test_xor_invariant.py` (NEW)
- Tests:
  - `test_call_with_both_parents_raises`
  - `test_call_with_neither_parent_raises`
  - `test_concurrent_calls_same_state_id_unique_indices`

#### 2.3 Database Failure Tests
- File: `tests/core/landscape/test_database_failures.py` (NEW)
- Tests:
  - `test_connection_failure_during_outcome_write`
  - `test_transaction_rollback_on_constraint_violation`
  - `test_payload_store_failure_propagates`

### Phase 3: Medium Priority Fixes (P2 - Should Have)

**Estimated Effort: 2-3 days**

#### 3.1 Coalesce Failure Recording
- File: `src/elspeth/engine/coalesce_executor.py`
- Lines: 421-449
- Change: Record node_states for held tokens before deleting from pending
- Outcome: FAILED with appropriate error_hash

#### 3.2 Resume Audit Marker
- File: `src/elspeth/engine/orchestrator.py`
- Method: `resume()`
- Change: Insert explicit "run_resumed_from_checkpoint" event
- New recorder method: `record_resume_event()`

#### 3.3 Abandoned Run Detection
- File: `src/elspeth/engine/orchestrator.py`
- Changes:
  - Add `RunStatus.ABANDONED`
  - Add configurable RUNNING timeout (default 1 hour)
  - Clean up stale checkpoints on resume validation

#### 3.4 Config Gate Error Handling
- File: `src/elspeth/engine/executors.py`
- Lines: 843-879
- Change: Add try/finally around routing state creation
- Ensure node_state completion on MissingEdgeError

### Phase 4: Completeness Fixes (P3 - Nice to Have)

**Estimated Effort: 1-2 days**

#### 4.1 Coalesce Metadata Persistence
- File: `src/elspeth/engine/coalesce_executor.py`
- Change: Pass metadata to `complete_node_state(context_after=...)`

#### 4.2 Aggregation Idle Flush
- File: `src/elspeth/engine/orchestrator.py`
- Change: Add explicit end-of-source aggregation flush loop

#### 4.3 Expression Parser Error Handling
- File: `src/elspeth/engine/executors.py`
- Change: Wrap template evaluation in try/finally

---

## Acceptance Criteria

### Audit Trail Completeness Invariants

After remediation, these invariants MUST hold under ALL conditions:

1. **Terminal State Invariant**
   - Every token reaches exactly ONE terminal state
   - No token remains in OPEN or BUFFERED state after run completes
   - Verified by: `test_audit_sweep.py` Query 1

2. **Node State Lifecycle Invariant**
   - Every `begin_node_state()` has matching `complete_node_state()`
   - Exception paths complete with FAILED status
   - Verified by: `test_audit_sweep.py` Query 4 (new)

3. **XOR Call Attribution Invariant**
   - Every call has exactly one parent (state_id XOR operation_id)
   - Validated at runtime, not just database constraint
   - Verified by: `test_xor_invariant.py`

4. **Checkpoint Durability Invariant**
   - Checkpoint implies sink write succeeded
   - Recovery never skips unwritten rows
   - Verified by: `test_checkpoint_durability.py`

5. **Crash Recovery Invariant**
   - Partial transactions are rolled back
   - Resume finds all incomplete rows
   - Verified by: `test_crash_scenarios.py`

### Definition of Done

- [ ] All P1 fixes implemented
- [ ] All P1 tests passing
- [ ] `test_audit_sweep.py` passes with new queries
- [ ] No OPEN node_states after any test run
- [ ] No tokens without terminal outcomes after any test run
- [ ] Property-based tests verify invariants under randomized conditions
- [ ] Documentation updated (CLAUDE.md, architecture docs)

---

## Risk Matrix

### If Not Addressed

| Scenario | Probability | Impact | Risk Score |
|----------|-------------|--------|------------|
| Crash during sink flush leaves audit incomplete | Medium | HIGH | **8/10** |
| XOR violation under concurrent calls | Low | HIGH | **6/10** |
| Checkpoint skip causes data loss | Low | CRITICAL | **9/10** |
| Coalesce failure loses token data | Medium | MEDIUM | **5/10** |
| Abandoned run causes duplicate processing | Low | MEDIUM | **4/10** |

### After Remediation (Target)

| Scenario | Probability | Impact | Risk Score |
|----------|-------------|--------|------------|
| Crash during sink flush | Low | LOW (detected, recovered) | **2/10** |
| XOR violation | Near-zero (validated) | LOW (prevented) | **1/10** |
| Checkpoint skip | Near-zero (reordered) | LOW (prevented) | **1/10** |
| Coalesce failure data loss | Low | LOW (recorded) | **2/10** |
| Abandoned run | Low | LOW (timeout, cleanup) | **2/10** |

---

## Historical Context

### Pattern: Infrastructure Exists, Code Path Missing

17 closed P0-P1 bugs followed this exact pattern:

| Bug ID | What Existed | What Was Missing | Detection Lag |
|--------|--------------|------------------|---------------|
| P0-2026-01-19 | `calls` table | No `record_call()` invocation | 1 week |
| P0-2026-01-19 | `source_data_ref` column | No payload storage in `create_row()` | 2 weeks |
| P0-2026-01-19 | `update_batch_status()` | Never called, batches stuck in DRAFT | 1 week |
| P2-2026-01-22 | `context_after` column | Coalesce metadata computed but discarded | STILL OPEN |

**Lesson:** Audit infrastructure is complete. The gaps are in **code paths that use it**, especially exception paths.

### Why Exception Paths Are Undertested

The codebase was built feature-forward with property-based testing for success paths. Exception paths were not systematically tested because:

1. Happy-path tests pass → feature appears complete
2. Exception injection is harder to write
3. Crash windows are narrow and hard to trigger
4. Database constraints catch some issues (but at wrong time)

**This epic addresses the systematic gap.**

---

## References

### Source Files

| File | Lines of Interest | Issue |
|------|-------------------|-------|
| `orchestrator.py` | 647-658 | Checkpoint pre-durability |
| `orchestrator.py` | 1185-1191 | Quarantine outcome ordering |
| `orchestrator.py` | 1900-2060 | Resume without marker |
| `executors.py` | 1698-1715 | Sink flush exception |
| `executors.py` | 843-879 | Config gate error |
| `coalesce_executor.py` | 421-449 | Failure tokens unrecorded |
| `recorder.py` | 1875-1920 | XOR validation missing |
| `repositories.py` | Multiple | Tier 1 validation gaps |

### Test Files

| File | Status | Coverage |
|------|--------|----------|
| `test_terminal_states.py` | EXCELLENT | Property-based |
| `test_recovery_fork_partial.py` | EXCELLENT | Fork/coalesce recovery |
| `test_audit_sweep.py` | GOOD | 6 sweep queries |
| `test_crash_scenarios.py` | **MISSING** | Zero coverage |
| `test_xor_invariant.py` | **MISSING** | Zero coverage |
| `test_database_failures.py` | **MISSING** | Zero coverage |

### Documentation

| File | Section | Update Needed |
|------|---------|---------------|
| `CLAUDE.md` | Audit Invariants | Add formal invariant list |
| `landscape-system.md` | Exception Handling | Document crash semantics |
| `landscape-audit-entry-points.md` | Gap Analysis | Mark fixed items |

---

*Document Version: 1.0*
*Last Updated: 2026-01-31*
*Next Review: After Phase 1 completion*
