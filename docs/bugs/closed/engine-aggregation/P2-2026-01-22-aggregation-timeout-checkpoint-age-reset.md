# Bug Report: Timeout age resets after checkpoint restore

## Summary

- Crash recovery restores trigger counts by replaying `record_accept()`, which resets `first_accept_time` to the recovery time. Any elapsed batch age before the crash is lost, delaying timeout triggers beyond their configured window.

## Severity

- Severity: major
- Priority: P2

## Reporter

- Name or handle: codex
- Date: 2026-01-22
- Related run/issue ID: N/A

## Environment

- Commit/branch: `fix/rc1-bug-burndown-session-2` @ `ae2c0e6f088f467276582fa8016f91b4d3bb26c7`
- OS: Linux 6.8.0-90-generic
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: deep dive into `src/elspeth/engine/triggers.py` and file bugs
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: inspected checkpoint restore path for trigger state

## Steps To Reproduce

1. Configure an aggregation with `timeout_seconds: 10`.
2. Accept a row and wait ~9 seconds.
3. Simulate a crash, restore from checkpoint, then wait another ~2 seconds.
4. Observe that the batch does not flush even though total age exceeds 10 seconds.

## Expected Behavior

- Timeout should account for elapsed time before the crash and trigger shortly after recovery.

## Actual Behavior

- Timeout age is reset during recovery, effectively extending the timeout window.

## Evidence

- Recovery replays `record_accept()` without restoring timestamps: `src/elspeth/engine/executors.py:1072-1097`
- `record_accept()` initializes `first_accept_time` to `time.monotonic()`: `src/elspeth/engine/triggers.py:74-82`

## Impact

- User-facing impact: delayed batch flushes after recovery.
- Data integrity / security impact: trigger timing is inconsistent across crashes.
- Performance or cost impact: prolonged buffering increases memory usage.

## Root Cause Hypothesis

- Trigger evaluator lacks a persisted `first_accept_time` (or elapsed age) in checkpoint state.

## Proposed Fix

- Code changes (modules/files):
  - Include `first_accept_time` (or elapsed seconds) in checkpoint state.
  - Add a `restore` method on `TriggerEvaluator` to set count and age explicitly.
- Config or schema changes: extend checkpoint payload with trigger timing metadata.
- Tests to add/update:
  - Add checkpoint/restore test for timeout triggers preserving elapsed age.
- Risks or migration steps:
  - Define how to handle clock differences across process restarts.

## Architectural Deviations

- Spec or doc reference: `docs/contracts/plugin-protocol.md:1208-1210`
- Observed divergence: timeout semantics are not preserved through recovery.
- Reason (if known): checkpoint only stores buffered rows/token IDs.
- Alignment plan or decision needed: define timeout behavior across crashes.

## Acceptance Criteria

- Timeout-triggered flushes occur based on total elapsed batch age, including pre-crash time.

## Tests

- Suggested tests to run: `pytest tests/engine/test_executors.py -k checkpoint`
- New tests required: yes

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `docs/contracts/plugin-protocol.md`

---

## Verification (2026-01-25)

**Status: STILL VALID**

### Investigation Summary

Performed comprehensive verification by examining:
1. Current checkpoint state structure (`src/elspeth/engine/executors.py:1081-1149`)
2. Checkpoint restoration logic (`src/elspeth/engine/executors.py:1151-1223`)
3. TriggerEvaluator implementation (`src/elspeth/engine/triggers.py`)
4. Git history for any timeout-related checkpoint fixes
5. Test coverage for timeout trigger restoration

### Findings

#### 1. Checkpoint State Does NOT Include Timing Information

**Location:** `src/elspeth/engine/executors.py:1118-1129`

The checkpoint state format only includes:
```python
state[node_id] = {
    "tokens": [
        {
            "token_id": t.token_id,
            "row_id": t.row_id,
            "branch_name": t.branch_name,
            "row_data": t.row_data,
        }
        for t in tokens
    ],
    "batch_id": self._batch_ids.get(node_id),
}
```

**Missing:** No `first_accept_time`, `elapsed_age`, or any timing metadata for timeout triggers.

#### 2. Restoration Replays record_accept() Which Resets Timer

**Location:** `src/elspeth/engine/executors.py:1217-1222`

During restoration:
```python
# Restore trigger evaluator count (so next row triggers at correct count)
evaluator = self._trigger_evaluators.get(node_id)
if evaluator is not None:
    # Record each restored row as "accepted" to advance the count
    for _ in reconstructed_tokens:
        evaluator.record_accept()
```

**Problem:** `record_accept()` is called for each buffered row.

#### 3. record_accept() Initializes Timer to Current Time

**Location:** `src/elspeth/engine/triggers.py:74-82`

```python
def record_accept(self) -> None:
    """Record that a row was accepted into the batch.

    Call this after each successful accept. Updates batch_count and
    starts the timer on first accept.
    """
    self._batch_count += 1
    if self._first_accept_time is None:
        self._first_accept_time = time.monotonic()
```

**Issue:** The first call to `record_accept()` during restoration sets `_first_accept_time = time.monotonic()`, which is the **recovery time**, not the original batch start time.

#### 4. Concrete Example of the Bug

**Scenario:**
1. **T=0s:** First row accepted, `first_accept_time = 0`, timeout configured for 10s
2. **T=9s:** Process crashes with 1 row buffered (timeout has 1s remaining)
3. **T=30s:** Process restarts, restores checkpoint
4. **Restoration:** Calls `record_accept()` which sets `first_accept_time = 30` (monotonic clock)
5. **T=32s:** Second row arrives, timeout check happens
6. **Actual age:** `32 - 30 = 2s` (but should be `32 - 0 = 32s`)
7. **Result:** Timeout does NOT fire even though 32s > 10s

**Expected:** Timeout should fire immediately at recovery (9s pre-crash + time since recovery > 10s)

**Actual:** Timeout clock resets to 0 at recovery time, extending the timeout window indefinitely.

#### 5. No Fix in Git History

**Commands run:**
```bash
git log --all --oneline --grep="timeout" --grep="checkpoint" --grep="first_accept_time" --since="2026-01-22"
git log --all --oneline -- src/elspeth/engine/triggers.py
```

**Result:** No commits addressing timeout age preservation in checkpoints.

Recent checkpoint work (`3e25073`, `260b9a7`, `59bb35f`) focused on storing full `TokenInfo` metadata but did NOT address trigger timing state.

#### 6. No Tests for Timeout Restoration

**Location:** `tests/engine/test_executors.py`

Examined checkpoint restoration tests:
- `test_restore_from_checkpoint_restores_buffers` - Tests count restoration only
- `test_checkpoint_roundtrip` - Tests token metadata preservation
- `test_restore_from_checkpoint_restores_trigger_count` - Tests **count** trigger restoration

**Gap:** No test verifies timeout trigger age is preserved across checkpoint restore.

The existing test at line 2987 shows restoration works for **count triggers**:
```python
# Process 1 more row - should trigger flush (2 restored + 1 new = 3)
result_list = processor.process_row(...)
assert result.outcome == RowOutcome.COMPLETED  # Count trigger fired
```

But there's no equivalent test for **timeout triggers** where the batch age matters.

#### 7. Related Bug Confirms Pattern

**Related Issue:** `docs/bugs/open/engine-orchestrator/P1-2026-01-22-aggregation-timeout-idle-never-fires.md`

This P1 bug (verified 2026-01-25, status STILL VALID) shows that timeout triggers have **systemic issues**:
- Timeouts only check on new row arrivals (no periodic checking)
- Same architectural gap affects coalesce timeouts

The checkpoint age reset bug is **another manifestation** of incomplete timeout trigger infrastructure.

### Code Evidence

**Current Flow (HEAD at 7540e57):**

1. **Before Crash:**
   - Row accepted → `record_accept()` → `_first_accept_time = time.monotonic()` (say, 100.0)
   - Batch age = `time.monotonic() - 100.0` (grows over time)
   - Crash at T=109.0 (9 seconds elapsed)

2. **Checkpoint Saved:**
   ```json
   {
     "node_id": {
       "tokens": [...],
       "batch_id": "batch-123"
       // NO first_accept_time or elapsed_age
     }
   }
   ```

3. **After Recovery (say T=200.0 in monotonic clock):**
   - `restore_from_checkpoint()` called
   - Calls `record_accept()` for each buffered row
   - First call: `_first_accept_time = 200.0` (RESET!)
   - Batch age now = `time.monotonic() - 200.0` = 0.0
   - **Lost:** Original 9 seconds of elapsed age

4. **Next Row Arrival (say T=202.0):**
   - Timeout check: `batch_age_seconds >= timeout_seconds`
   - Actual: `(202.0 - 200.0) = 2.0 >= 10.0` → FALSE
   - Expected: `(202.0 - 100.0) = 102.0 >= 10.0` → TRUE
   - **Bug:** Timeout doesn't fire even though actual elapsed time is 102s > 10s

### Impact Analysis

**Affected Scenarios:**
1. **Time-based aggregations:** Batches intended to flush every N seconds will be delayed by crash recovery
2. **Real-time pipelines:** SLA violations for latency-sensitive processing
3. **Resource management:** Extended buffering increases memory pressure

**Severity Assessment:**
- **Data Correctness:** ✓ Not affected (rows still eventually flush at end-of-source or count trigger)
- **Timing Guarantees:** ✗ Violated (timeout contracts not honored across crashes)
- **Audit Trail:** ✓ Correct (trigger type recorded accurately when it finally fires)

**Why P2 (not P1):**
- Workaround exists: Use count triggers instead of timeout triggers
- Data integrity not compromised
- Most aggregations use count triggers (timeout is less common)
- Related P1 bug (timeout-idle-never-fires) is more critical (affects non-crash scenarios too)

### Proposed Fix Details

**Option 1: Store Elapsed Age (Simpler)**

Extend checkpoint state:
```python
state[node_id] = {
    "tokens": [...],
    "batch_id": ...,
    "elapsed_age_seconds": evaluator.batch_age_seconds,  # NEW
}
```

Restoration:
```python
if "elapsed_age_seconds" in node_state:
    # Restore age by backdat ing first_accept_time
    evaluator._first_accept_time = time.monotonic() - node_state["elapsed_age_seconds"]
```

**Pros:** Clock-independent, simple
**Cons:** Doesn't preserve absolute timestamps

**Option 2: Store Absolute Timestamp (Clock-Dependent)**

Store `first_accept_time` directly, but this requires handling:
- Monotonic clock differences across restarts (clocks reset)
- System clock changes (unreliable)

**Verdict:** Option 1 (elapsed age) is more robust.

**Option 3: Hybrid Approach**

Store **both** elapsed age and wall-clock timestamp:
- Use elapsed age for timeout calculations (reliable)
- Store wall-clock timestamp for audit trail (human-readable)

This matches the audit trail's philosophy: "Record everything, even if redundant."

### Test Requirements

**New Test Needed:**
```python
def test_timeout_age_preserved_across_checkpoint_restore(self) -> None:
    """Timeout trigger age is preserved through checkpoint/restore cycle."""
    # Buffer rows with timeout trigger
    # Wait 2 seconds (batch age = 2s, timeout = 5s)
    # Checkpoint + restore
    # Wait 4 more seconds (total = 6s > 5s timeout)
    # Verify timeout fires (not count trigger)
```

This test would **FAIL** with current implementation and **PASS** after fix.

### Conclusion

**Bug Status: STILL VALID**

The bug remains unfixed as of commit `7540e57` (current HEAD on `fix/rc1-bug-burndown-session-4`):

1. ✅ Bug confirmed through code inspection
2. ✅ Concrete failure scenario documented
3. ❌ No fix implemented
4. ❌ No timing metadata in checkpoint state
5. ❌ Restoration resets `first_accept_time` to recovery time
6. ❌ No test coverage for timeout age restoration

**Root Cause:** `TriggerEvaluator` timing state (`_first_accept_time`) is not included in checkpoint persistence, and restoration path inadvertently resets it.

**Impact:** Timeout triggers extend their window by the crash duration, violating timing contracts for time-based aggregations.

**Recommended Fix:** Store `elapsed_age_seconds` in checkpoint state, restore by backdating `_first_accept_time` to `time.monotonic() - elapsed_age`. Add test verifying timeout fires correctly after restoration.

**Priority Rationale (P2):** While confirmed valid, this is less critical than P1-2026-01-22-aggregation-timeout-idle-never-fires (timeouts never fire during idle periods), which affects all timeout-based aggregations even without crashes. Fix that P1 issue first, then address checkpoint restoration in the same refactor.
