# Bug Report: Azure multi-query batch uses synthetic state_id not tied to node_state

## Summary

- Batch processing appends "_row{i}" to ctx.state_id and records external calls under those synthetic IDs, but only the original ctx.state_id exists in node_states. This violates the calls -> node_states FK and breaks audit traceability for batch runs.

## Severity

- Severity: critical
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
- Data set or fixture: any batch run using azure_multi_query_llm

## Agent Context (if relevant)

- Goal or task prompt: deep dive into src/elspeth/plugins/llm for bugs
- Model/version: Codex (GPT-5)
- Tooling and permissions (sandbox/approvals): workspace-write sandbox, no escalations
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code review only

## Steps To Reproduce

1. Configure an aggregation node that flushes to `azure_multi_query_llm` with landscape enabled.
2. Run a batch that triggers aggregation flush.
3. Observe external call recording or DB insert behavior for LLM calls.

## Expected Behavior

- All external LLM calls are recorded under the batch node's actual `state_id` so they link to a real `node_states` row.

## Actual Behavior

- LLM calls are recorded under synthetic IDs like `<state_id>_row0`, which do not exist in `node_states`, causing FK violations or orphaned call records.

## Evidence

- Per-row state IDs are constructed in `src/elspeth/plugins/llm/azure_multi_query.py:492`.
- External calls require a valid `node_states.state_id` per FK in `src/elspeth/core/landscape/schema.py:191`.

## Impact

- User-facing impact: batch runs can crash when recording calls.
- Data integrity / security impact: audit trail loses linkage between calls and node state.
- Performance or cost impact: failed batches may require reruns.

## Root Cause Hypothesis

- Batch path uses synthetic state IDs for per-row isolation without creating matching node_states entries.

## Proposed Fix

- Code changes (modules/files):
  - Use `ctx.state_id` for all per-row LLM calls in batch mode, or create per-row node_states before calling the LLM and use those IDs.
- Config or schema changes: N/A
- Tests to add/update:
  - Add a batch-mode test that asserts calls are recorded against an existing state_id.
- Risks or migration steps:
  - If switching to shared state_id, verify call_index uniqueness and audit ordering.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): CLAUDE.md "External calls - Full request AND response recorded" and calls FK to node_states.
- Observed divergence: external calls recorded under nonexistent state_id.
- Reason (if known): attempt to isolate per-row calls in batch without audit model support.
- Alignment plan or decision needed: confirm whether batch uses shared state_id or supports per-row node_states.

## Acceptance Criteria

- Batch-mode multi-query runs record all calls under valid node_states IDs with no FK errors.

## Tests

- Suggested tests to run: `pytest tests/plugins/llm/test_azure_multi_query.py -k batch -v`
- New tests required: yes, audit linkage test for calls -> node_states.

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: CLAUDE.md auditability standard

---

## Verification (2026-01-24)

### Status: **STILL VALID**

### Verification Method

1. **Code inspection** of `/home/john/elspeth-rapid/src/elspeth/plugins/llm/azure_multi_query.py`
2. **Schema verification** of FK constraints in `/home/john/elspeth-rapid/src/elspeth/core/landscape/schema.py`
3. **Git history analysis** of state_id handling changes
4. **Test coverage review** to check if FK violations are exercised

### Findings

#### 1. Bug Still Present in Code (Lines 462-510)

The `_process_batch()` method at line 462 creates synthetic state_ids:

```python
# Line 493 in azure_multi_query.py
row_state_id = f"{ctx.state_id}_row{i}"
```

This synthetic ID is then passed to `_process_single_row_internal()` (line 496), which:
- Creates an `AuditedLLMClient` with the synthetic state_id (via `_get_llm_client()` line 146-158)
- Records external LLM calls under this synthetic state_id (line 210 in `chat_completion()`)

**However**, only `ctx.state_id` exists in the `node_states` table. The synthetic `_row{i}` variants are never recorded as node_states.

#### 2. FK Constraint Exists and is Enforced

Schema verification confirms:
- FK constraint exists: `/home/john/elspeth-rapid/src/elspeth/core/landscape/schema.py:192`
  ```python
  Column("state_id", String(64), ForeignKey("node_states.state_id"), nullable=False),
  ```
- SQLite FK enforcement is enabled: `/home/john/elspeth-rapid/src/elspeth/core/landscape/database.py:86`
  ```python
  cursor.execute("PRAGMA foreign_keys=ON")
  ```
- Recent FK enforcement commit: `1f057c0` (2026-01-23) added comprehensive FK validation

This means **FK violations WILL crash** when landscape recording is active.

#### 3. Git History Confirms Bug Introduction

- Bug introduced in commit `bf22a43` (2026-01-21): "feat(llm): implement batch processing for azure_multi_query"
- Original implementation included synthetic state_id pattern from the start
- No subsequent fixes found in git history

#### 4. Test Coverage Does Not Exercise FK Constraint

Review of `/home/john/elspeth-rapid/tests/plugins/llm/test_azure_multi_query.py`:
- `test_process_batch_uses_per_row_state_ids` (line 505) verifies synthetic IDs are created
- **BUT** all tests use mocked landscape recorder, so FK constraints are never validated
- No integration tests exist that combine azure_multi_query batch mode with real landscape database

### Why This Hasn't Crashed Yet

**Critical insight**: The bug is **latent** because:

1. Unit tests mock the landscape recorder (no real database)
2. Integration tests don't cover azure_multi_query in batch mode with landscape enabled
3. No production usage has triggered aggregation -> azure_multi_query batch path with landscape recording

### Reproduction Scenario

To trigger the bug, you would need:

```yaml
# pipeline.yaml
aggregations:
  - plugin: collect_batch
    flush_to: llm_eval

transforms:
  - id: llm_eval
    plugin: azure_multi_query_llm
    # ... config ...
```

With landscape enabled, when the aggregation flushes a batch to `azure_multi_query_llm`:
1. Engine creates ONE node_state with `state_id = "state-abc123"`
2. `azure_multi_query._process_batch()` creates synthetic IDs: `"state-abc123_row0"`, `"state-abc123_row1"`, etc.
3. `AuditedLLMClient.chat_completion()` tries to record calls with synthetic state_id
4. **CRASH**: FK constraint violation `(sqlite3.IntegrityError) FOREIGN KEY constraint failed`

### Evidence Line References

- Synthetic state_id creation: `/home/john/elspeth-rapid/src/elspeth/plugins/llm/azure_multi_query.py:493`
- LLM client creation with synthetic ID: `/home/john/elspeth-rapid/src/elspeth/plugins/llm/azure_multi_query.py:206`
- FK constraint definition: `/home/john/elspeth-rapid/src/elspeth/core/landscape/schema.py:192`
- FK enforcement enabled: `/home/john/elspeth-rapid/src/elspeth/core/landscape/database.py:86`
- Recorder expects valid state_id: `/home/john/elspeth-rapid/src/elspeth/core/landscape/recorder.py:2056`
  > "Invalid state_id will raise IntegrityError due to foreign key constraint."

### Severity Assessment

**Severity: P1 (Critical)** - Confirmed valid
- **Impact**: Complete pipeline crash when batch mode + landscape are combined
- **Data loss risk**: Partial batch results lost on crash
- **Audit integrity**: Zero audit trail for failed batch runs
- **Workaround**: None (avoid batch mode or disable landscape)

### Recommended Next Steps

1. **Fix implementation** using one of these approaches:
   - **Option A**: Use shared `ctx.state_id` for all batch rows (simpler, but loses per-row call isolation)
   - **Option B**: Record per-row node_states before LLM calls (maintains isolation, more complex)

2. **Add integration test** that exercises:
   - Aggregation flush to azure_multi_query_llm
   - Real landscape database (not mocked)
   - Verify calls are recorded without FK violations

3. **Update existing test** `test_process_batch_uses_per_row_state_ids`:
   - Replace mock with real landscape database
   - Assert FK constraint is satisfied after batch processing

### Verification Confidence: **HIGH**

- Code inspection confirms synthetic state_id pattern
- FK constraints verified in schema and enforced at runtime
- No test coverage for this failure path
- Git history shows no fixes since introduction

---

## RESOLUTION: 2026-01-25 (via systematic debugging review 2026-01-26)

**Status:** FIXED

**Fixed by:** Commit `b5f3f50` (2026-01-25) - "fix(infra): thread safety, integration tests, and Azure audit trail"

**Root Cause Analysis:**
Systematic debugging (Phase 1) revealed the bug was introduced in commit `bf22a43` (2026-01-21) and fixed 4 days later in commit `b5f3f50`. The original implementation created synthetic state_ids like `"state-abc123_row0"` that violated FK constraints because only the parent `"state-abc123"` existed in `node_states` table.

**Implementation:**
The fix removed synthetic per-row state_id creation and uses shared `ctx.state_id` for all rows in a batch:

**Before (buggy):**
```python
for i, row in enumerate(rows):
    row_state_id = f"{ctx.state_id}_row{i}"  # FK violation!
    result = self._process_single_row_internal(row, row_state_id)
```

**After (fixed):**
```python
# All rows share ctx.state_id, ensuring FK constraint satisfaction
for _i, row in enumerate(rows):
    result = self._process_single_row_internal(row, ctx.state_id)
```

**Benefits of shared state_id approach:**
- FK constraints satisfied (all calls reference valid state_id)
- Single LLM client cached per batch (not per row)
- Call index continuity across all queries
- Simpler implementation

**Tests added:**
- `test_process_batch_uses_shared_state_id` (line 514-577) - Verifies all rows use shared state_id
- `test_process_batch_cleans_up_shared_client` (line 579+) - Verifies client cleanup

**Remaining gap:**
Unit tests use mocked landscape (no real database). An integration test with real landscape database + FK enforcement would provide additional confidence but is not critical since the fix is architecturally sound.

**Files changed:**
- `src/elspeth/plugins/llm/azure_multi_query.py` (lines 511-534)
- `tests/plugins/llm/test_azure_multi_query.py` (new tests)

**Verification confidence:** HIGH - Fix is correct, tested, and matches CLAUDE.md audit requirements.

### Git Diff Evidence

```bash
$ git show b5f3f50 -- src/elspeth/plugins/llm/azure_multi_query.py | grep -A 5 -B 5 "row_state_id\|ctx.state_id"
```

```diff
-        for i, row in enumerate(rows):
-            row_state_id = f"{ctx.state_id}_row{i}"
+        # BUG-AZURE-02 FIX: Use ONE LLM client for entire batch
+        # All rows share ctx.state_id, ensuring FK constraint satisfaction
+
+        try:
+            for _i, row in enumerate(rows):
+                # BUG-AZURE-02 FIX: Use shared state_id for all rows in batch
+                # (removed synthetic row_state_id creation)
+                result = self._process_single_row_internal(row, ctx.state_id)
```

### Current Code Verification

**File:** `src/elspeth/plugins/llm/azure_multi_query.py`

**Line 522:** Uses `ctx.state_id` directly (no synthetic suffix)
```python
result = self._process_single_row_internal(row, ctx.state_id)
```

**Grep verification:**
```bash
$ grep -n "row_state_id" src/elspeth/plugins/llm/azure_multi_query.py
521:                # (removed synthetic row_state_id creation)
```
Only appears in comment documenting removal - pattern is gone from code.

### Test Evidence

**File:** `tests/plugins/llm/test_azure_multi_query.py:514-577`

Key assertions:
```python
# Line 572: Verify ALL calls use shared state_id
unique_state_ids = set(created_state_ids)
assert len(unique_state_ids) == 1

# Line 577: Verify NO synthetic IDs created
for state_id in created_state_ids:
    assert "_row" not in state_id
```

Test explicitly validates the fix by checking no `_row` suffix exists.
