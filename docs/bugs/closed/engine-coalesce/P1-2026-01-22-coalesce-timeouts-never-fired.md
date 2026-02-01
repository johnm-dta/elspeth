# Bug Report: Coalesce timeouts are never checked (check_timeouts is unused)

## Summary

- `CoalesceExecutor.check_timeouts()` is defined but never invoked, so `best_effort` and `quorum` timeouts do not fire; coalesce groups wait until end-of-source (or forever in streaming sources).

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: codex
- Date: 2026-01-22
- Related run/issue ID: N/A

## Environment

- Commit/branch: `ae2c0e6f088f467276582fa8016f91b4d3bb26c7` (fix/rc1-bug-burndown-session-2)
- OS: Linux (Ubuntu kernel 6.8.0-90-generic)
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: deep dive into coalesce_executor, identify bugs, create bug docs
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code inspection of coalesce executor and orchestrator; rg search for check_timeouts usage

## Steps To Reproduce

1. Configure a pipeline with a fork and a coalesce set to `policy: best_effort` (or `quorum`) and a short `timeout_seconds`.
2. Use a long-running or streaming source so end-of-source is not reached quickly.
3. Run the pipeline and wait past the timeout.

## Expected Behavior

- After `timeout_seconds`, the coalesce should merge (best_effort) or resolve (quorum) without waiting for end-of-source.

## Actual Behavior

- No timeout-driven merge occurs; pending coalesces only resolve on full arrival or end-of-source flush.

## Evidence

- Timeout handler exists but has no callers: `src/elspeth/engine/coalesce_executor.py:303`
- Comment claims a timeout loop exists, but no loop calls it: `src/elspeth/engine/coalesce_executor.py:112`
- Orchestrator only flushes at end-of-source: `src/elspeth/engine/orchestrator.py:866`

## Impact

- User-facing impact: pipelines with timeouts hang or emit delayed results.
- Data integrity / security impact: none directly, but audit timing data is wrong.
- Performance or cost impact: unbounded pending state growth on long-running sources.

## Root Cause Hypothesis

- `check_timeouts()` was implemented but never wired into orchestrator/processor scheduling.

## Proposed Fix

- Code changes (modules/files):
  - Call `CoalesceExecutor.check_timeouts()` periodically during processing (e.g., per row or on a timer) and enqueue any merged tokens.
  - Ensure timeout-driven failures are handled for policies that require it.
- Config or schema changes: none.
- Tests to add/update:
  - Add an integration test with `best_effort` timeout verifying merge without end-of-source.
- Risks or migration steps:
  - Ensure timeout checks do not introduce excessive overhead; consider batching or interval checks.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `docs/contracts/plugin-protocol.md#L1092`
- Observed divergence: timeouts are defined but never triggered.
- Reason (if known): missing orchestration loop wiring.
- Alignment plan or decision needed: implement periodic timeout checks in execution loop.

## Acceptance Criteria

- A coalesce with `best_effort` or `quorum` timeout resolves when timeout elapses during streaming runs.
- No pending coalesce remains past its timeout unless policy explicitly allows it.

## Tests

- Suggested tests to run:
  - `pytest tests/engine/test_coalesce_executor.py -k timeout`
- New tests required: yes (timeout merge in streaming mode)

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `docs/contracts/plugin-protocol.md`

---

## Verification Report (2026-01-24)

**Status: STILL VALID**

### Investigation Summary

Performed comprehensive verification of the bug by examining:
1. Current orchestrator code (`src/elspeth/engine/orchestrator.py`)
2. CoalesceExecutor implementation (`src/elspeth/engine/coalesce_executor.py`)
3. Git history from bug report date (2026-01-22) to present (2026-01-24)
4. Test coverage for `check_timeouts()` functionality

### Findings

#### 1. `check_timeouts()` Method Exists But Is Never Called

**Location:** `src/elspeth/engine/coalesce_executor.py:303-370`

The method is fully implemented with timeout logic for both `best_effort` and `quorum` policies:
- Tracks elapsed time using `time.monotonic()`
- Identifies timed-out entries based on `settings.timeout_seconds`
- Executes merge operations for timed-out coalesces
- Returns list of `CoalesceOutcome` objects

**Problem:** Zero invocations found in production code (orchestrator or processor).

#### 2. Misleading Comment Suggests Missing Implementation

**Location:** `src/elspeth/engine/coalesce_executor.py:115`

```python
def get_registered_names(self) -> list[str]:
    """Get names of all registered coalesce points.

    Used by processor for timeout checking loop.  # <-- This loop does not exist

    Returns:
        List of registered coalesce names
    """
    return list(self._settings.keys())
```

The comment claims this method is "used by processor for timeout checking loop," but no such loop exists in the codebase.

#### 3. Orchestrator Main Loop - No Timeout Checks

**Location:** `src/elspeth/engine/orchestrator.py:891-1016`

The main processing loop (lines 891-1016) processes rows with these checkpoints:
- Line 936: `processor.process_row()` - processes individual rows
- Lines 995-1016: Progress emission every 100 rows or 5 seconds
- **MISSING:** No calls to `coalesce_executor.check_timeouts()`

The only coalesce interaction is at **end-of-source** (line 1037-1053):
```python
# Flush pending coalesce operations at end-of-source
if coalesce_executor is not None:
    flush_step = len(config.transforms) + len(config.gates)
    pending_outcomes = coalesce_executor.flush_pending(flush_step)
```

This confirms coalesces only resolve at end-of-source, not on timeout.

#### 4. Test Coverage Confirms Functionality (But Not Integration)

**Unit Tests Found:**
- `tests/engine/test_coalesce_executor.py:484-486` - Tests `check_timeouts()` returns empty list when quorum not met
- `tests/engine/test_coalesce_executor.py:569-572` - Tests `check_timeouts()` triggers best_effort merge after timeout
- `tests/engine/test_coalesce_executor.py:576-596` - Tests error handling for unregistered coalesce
- `tests/engine/test_processor.py:2000-2003` - Manual invocation in test (not via orchestrator loop)

**Gap:** All tests manually call `check_timeouts()`. No integration test verifies timeout-driven merges happen automatically during pipeline execution.

#### 5. Git History - No Fix Applied

**Commands run:**
```bash
git log --oneline --grep="timeout\|coalesce" --since="2026-01-22" --all
git diff ae2c0e6f..HEAD -- src/elspeth/engine/orchestrator.py src/elspeth/engine/coalesce_executor.py
```

**Result:** No commits added `check_timeouts()` calls to orchestrator since original bug report (commit `ae2c0e6f`).

Recent timeout-related commits were unrelated:
- `f819d19` - Rate limiter thread-safety fix
- `0c225ee` - Test configuration fixes for fork/coalesce
- `e60a7e1` - Copilot review feedback

### Code Evidence

**Current state (HEAD):**
```python
# orchestrator.py lines 995-1016: Progress emission loop
# EXPECTED: check_timeouts() call here every 5 seconds
current_time = time.perf_counter()
time_since_last_progress = current_time - last_progress_time
should_emit = (
    rows_processed == 1
    or rows_processed % progress_interval == 0
    or time_since_last_progress >= progress_time_interval  # Every 5 seconds
)
if should_emit:
    elapsed = current_time - start_time
    self._events.emit(ProgressEvent(...))
    last_progress_time = current_time
    # MISSING: No timeout check here
```

**Where timeout check should be added:**

The progress emission block (lines 995-1016) runs every 5 seconds, making it the ideal location for periodic timeout checks. The fix would look like:

```python
if should_emit:
    # ... existing progress emission ...

    # Check for timed-out coalesces (every 5 seconds)
    if coalesce_executor is not None:
        for coalesce_name in coalesce_executor.get_registered_names():
            timed_out = coalesce_executor.check_timeouts(
                coalesce_name=coalesce_name,
                step_in_pipeline=len(config.transforms) + len(config.gates)
            )
            for outcome in timed_out:
                if outcome.merged_token is not None:
                    rows_coalesced += 1
                    pending_tokens[output_sink_name].append(outcome.merged_token)
```

### Conclusion

**Bug Status: STILL VALID**

The bug remains unfixed as of commit `36e17f2` (current HEAD on `fix/rc1-bug-burndown-session-4`):
1. `check_timeouts()` method exists and is tested in isolation
2. No integration into orchestrator's main processing loop
3. Timeouts will never fire during pipeline execution
4. Coalesces only resolve at end-of-source (line 1037)

**Impact:** Pipelines with streaming sources or long-running batches will never honor `timeout_seconds` configuration, causing indefinite waits for best_effort/quorum coalesces.

**Recommended Fix Location:** Add timeout checks to the progress emission block (lines 995-1016) which already runs every 5 seconds.

**Verification Method:** Create integration test with streaming source, fork to 3 branches, coalesce with `timeout_seconds: 2`, verify merge happens after 2 seconds without end-of-source.
