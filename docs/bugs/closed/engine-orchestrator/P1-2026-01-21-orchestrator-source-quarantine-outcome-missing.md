# Bug Report: Source quarantined rows are routed without QUARANTINED outcome

## Summary

- When `SourceRow.is_quarantined` is true, the orchestrator creates a token and routes it to the quarantine sink but never records a `RowOutcome.QUARANTINED`.
- The `quarantine_error` payload on `SourceRow` is ignored, so quarantined rows appear as completed sink outputs or lack a terminal outcome in the audit trail.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: Codex
- Date: 2026-01-21
- Related run/issue ID: N/A

## Environment

- Commit/branch: ae2c0e6 (fix/rc1-bug-burndown-session-2)
- OS: unknown
- Python version: unknown
- Config profile / env vars: N/A
- Data set or fixture: any source that yields `SourceRow.quarantined(...)`

## Agent Context (if relevant)

- Goal or task prompt: deep dive into src/elspeth/engine/orchestrator.py
- Model/version: Codex (GPT-5)
- Tooling and permissions (sandbox/approvals): workspace-write sandbox, no escalations
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code review only

## Steps To Reproduce

1. Configure a source with schema validation and `on_validation_failure` pointing to a quarantine sink.
2. Supply at least one invalid row so the source yields `SourceRow.quarantined(...)`.
3. Run the pipeline and inspect `token_outcomes` for the quarantined token.

## Expected Behavior

- Quarantined rows should record `RowOutcome.QUARANTINED` with an error hash derived from `SourceRow.quarantine_error`, while still routing to the quarantine sink for storage.

## Actual Behavior

- The orchestrator routes the quarantined row to a sink without calling `record_token_outcome`, so the audit trail lacks a QUARANTINED terminal outcome (or shows only a completed sink node_state).

## Evidence

- Quarantined rows are routed directly to sinks without recording outcomes in `src/elspeth/engine/orchestrator.py:780-795`.
- The standard QUARANTINED outcome path records `record_token_outcome` in `src/elspeth/engine/processor.py:779-790`, but this path is bypassed.
- `SourceRow.quarantine_error` exists but is unused in this flow (`src/elspeth/contracts/results.py:283-322`).

## Impact

- User-facing impact: quarantine metrics and audit queries cannot reliably identify quarantined rows.
- Data integrity / security impact: terminal state guarantees are violated; audit trail misrepresents failure handling.
- Performance or cost impact: none.

## Root Cause Hypothesis

- Source quarantine handling bypasses RowProcessor logic and does not record a terminal outcome or validation error for the quarantined token.

## Proposed Fix

- Code changes (modules/files):
  - In `src/elspeth/engine/orchestrator.py`, when `source_item.is_quarantined` is true, compute an error_hash from `source_item.quarantine_error` and call `record_token_outcome(..., outcome=RowOutcome.QUARANTINED, error_hash=...)` before routing to the quarantine sink.
  - Optionally record validation error details using `PluginContext.record_validation_error()` if available.
- Config or schema changes: N/A
- Tests to add/update:
  - Source quarantine flow records QUARANTINED outcome and error hash.
- Risks or migration steps:
  - Ensure QUARANTINED outcomes remain terminal even when a quarantine sink is used.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): CLAUDE.md "Terminal Row States" and "Auditability Standard".
- Observed divergence: quarantined rows do not reach a terminal QUARANTINED outcome.
- Reason (if known): quarantine handling is in orchestrator without outcome recording.
- Alignment plan or decision needed: define required audit records for source quarantine flows.

## Acceptance Criteria

- Every quarantined source row records a QUARANTINED outcome with an error hash.
- Audit queries can distinguish quarantined rows from completed rows written to normal sinks.

## Tests

- Suggested tests to run: `pytest tests/engine/test_orchestrator.py -k quarantine -v`
- New tests required: yes, source quarantine outcome recording.

## Notes / Links

- Related issues/PRs: P1-2026-01-21-validation-error-recording-crashes-on-nondict
- Related design docs: CLAUDE.md auditability standard

---

## Verification Report

**Verification Date:** 2026-01-24
**Verifier:** Claude Sonnet 4.5
**Commit Verified:** 36e17f2 (fix/rc1-bug-burndown-session-4)

**Status: STILL VALID**

### Current Code Analysis

**File:** `/home/john/elspeth-rapid/src/elspeth/engine/orchestrator.py`
**Lines:** 895-931 (in `_execute_run` method)

The source quarantine handling code:

```python
# Handle quarantined source rows - route directly to sink
if source_item.is_quarantined:
    rows_quarantined += 1
    # Route quarantined row to configured sink if it exists
    quarantine_sink = source_item.quarantine_destination
    if quarantine_sink and quarantine_sink in config.sinks:
        # Create a token for the quarantined row
        quarantine_token = processor.token_manager.create_initial_token(
            run_id=run_id,
            source_node_id=source_id,
            row_index=row_index,
            row_data=source_item.row,
        )
        pending_tokens[quarantine_sink].append(quarantine_token)
    # ... progress emission ...
    continue
```

**Missing:** No call to `recorder.record_token_outcome()` with `RowOutcome.QUARANTINED`

### Evidence of Bug

1. **Token created without outcome recorded** (lines 901-907)
   - Token is created via `token_manager.create_initial_token()`
   - Token is added to `pending_tokens[quarantine_sink]`
   - Token will be written to sink (line 1071-1082)
   - **But no outcome is recorded in `token_outcomes` table**

2. **Counter incremented without audit trail** (line 896)
   - `rows_quarantined` counter incremented
   - RunResult will show non-zero `rows_quarantined`
   - **But no corresponding `token_outcomes` records with `outcome='quarantined'`**

3. **Error information ignored** (`source_item.quarantine_error`)
   - SourceRow carries `quarantine_error` field (see `/home/john/elspeth-rapid/src/elspeth/contracts/results.py:321`)
   - This field contains validation error details
   - **Never used to compute error_hash or record error**

### Comparison with Transform Quarantine (Working Code)

The processor DOES record QUARANTINED outcomes for transform failures (processor.py:775-788):

```python
# QUARANTINED path - error sink is "discard"
error_detail = str(result.reason) if result.reason else "unknown_error"
quarantine_error_hash = hashlib.sha256(error_detail.encode()).hexdigest()[:16]
self._recorder.record_token_outcome(
    run_id=self._run_id,
    token_id=current_token.token_id,
    outcome=RowOutcome.QUARANTINED,
    error_hash=quarantine_error_hash,
)
return (
    RowResult(
        token=current_token,
        final_data=current_token.row_data,
        outcome=RowOutcome.QUARANTINED,
    ),
    child_items,
)
```

**Source quarantine bypasses this entire pathway.**

### Git History Check

**Searched for fixes since 2026-01-21:**

- Commit `5063fc0` (2026-01-21): Added `_validate_source_quarantine_destination()` validation - ensures quarantine sink exists, but does NOT add outcome recording
- Commit `b90e9d5` (2026-01-23): Fixed CSVSource to quarantine malformed rows - improved source-level quarantine, but orchestrator handling unchanged
- Commit `c774dfe` (2026-01-23): Fixed AzureBlobSource JSONL quarantine - source-level only
- Commit `cec7dbb` (2026-01-22): Fixed JSONSource quarantine - source-level only

**No commits address the orchestrator outcome recording gap.**

### Impact Verification

**Audit Trail Violations:**

1. **Terminal outcome guarantee broken**
   - CLAUDE.md specifies all rows must reach exactly one terminal state
   - Source-quarantined rows have NO terminal outcome recorded
   - They appear in sink output but not in outcome metrics

2. **Audit queries unreliable**
   - `SELECT * FROM token_outcomes WHERE outcome = 'quarantined'` misses source quarantine
   - RunResult.rows_quarantined != COUNT of quarantined outcomes in DB
   - Cannot distinguish "quarantined at source" from "quarantined in transform"

3. **Error traceability lost**
   - `source_item.quarantine_error` contains validation failure details
   - This is never persisted to audit trail
   - Cannot answer "why was this row quarantined?" for source failures

### Proposed Fix Validation

The proposed fix is architecturally sound:

```python
if source_item.is_quarantined:
    rows_quarantined += 1
    quarantine_sink = source_item.quarantine_destination
    if quarantine_sink and quarantine_sink in config.sinks:
        # Create token
        quarantine_token = processor.token_manager.create_initial_token(...)

        # MISSING: Compute error hash from quarantine_error
        error_detail = source_item.quarantine_error or "unknown_validation_error"
        error_hash = hashlib.sha256(error_detail.encode()).hexdigest()[:16]

        # MISSING: Record QUARANTINED outcome
        recorder.record_token_outcome(
            run_id=run_id,
            token_id=quarantine_token.token_id,
            outcome=RowOutcome.QUARANTINED,
            error_hash=error_hash,
        )

        # Then route to sink
        pending_tokens[quarantine_sink].append(quarantine_token)
    continue
```

This mirrors the transform quarantine pattern and ensures audit compliance.

### Test Coverage Gap

**No existing tests for source quarantine outcome recording:**

```bash
$ grep -r "test.*quarantine.*outcome" tests/
# (no results)
```

The test suggested in acceptance criteria does not exist:
```bash
$ pytest tests/engine/test_orchestrator.py -k quarantine -v
# (would run 0 tests - no quarantine outcome tests exist)
```

### Conclusion

**BUG STATUS: STILL VALID**

The bug persists in the current codebase (commit 36e17f2). Source-quarantined rows:
- ✅ Get routed to quarantine sink (working)
- ✅ Increment `rows_quarantined` counter (working)
- ❌ DO NOT record `RowOutcome.QUARANTINED` in audit trail (BUG)
- ❌ DO NOT persist validation error details (BUG)
- ❌ Violate terminal outcome guarantee (ARCHITECTURE VIOLATION)

**Priority remains P1** due to audit compliance violation.
