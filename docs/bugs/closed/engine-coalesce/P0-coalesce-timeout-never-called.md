# Bug Report: Coalesce Timeout Never Called in Processor Loop

## Summary

- The `CoalesceExecutor.check_timeouts()` method exists but is never called from the processor main loop. This means coalesce timeout policies (`best_effort`, timed `require_all`) never fire during processing - only at end-of-source. If a fork branch fails or hangs, the pipeline will wait indefinitely.

## Severity

- Severity: critical
- Priority: P0 (RC-1 Blocker)

## Reporter

- Name or handle: Release Validation Analysis
- Date: 2026-01-29
- Related run/issue ID: CRIT-03, TD-003

## Environment

- Commit/branch: fix/P2-aggregation-metadata-hardcoded
- OS: Linux
- Python version: 3.12+
- Config profile / env vars: Any pipeline with fork/coalesce and timeout
- Data set or fixture: Fork with one branch that fails or delays

## Agent Context (if relevant)

- Goal or task prompt: RC-1 release validation - identify blockers
- Model/version: Claude Opus 4.5
- Tooling and permissions: Read-only analysis
- Determinism details: N/A
- Notable tool calls or steps: Cross-referenced RC1-remediation.md, coalesce_executor.py, processor.py

## Steps To Reproduce

1. Configure a pipeline with:
   - A gate that forks to 2+ paths
   - A coalesce with `policy: best_effort` and `timeout_seconds: 5`
   - One branch that intentionally delays or fails
2. Run the pipeline with sufficient rows
3. Observe behavior when one branch doesn't complete within timeout

## Expected Behavior

- After 5 seconds, coalesce should fire with available branches
- Timed-out branches should be recorded as missing
- Pipeline continues without waiting indefinitely

## Actual Behavior

- Pipeline waits indefinitely for all branches
- Timeout only fires at end-of-source (if source ever completes)
- For streaming sources or long-running pipelines, this is effectively a hang

## Evidence

- `src/elspeth/engine/coalesce_executor.py:371-440` - `check_timeouts()` method exists, fully implemented
- `src/elspeth/engine/processor.py` - grep for "check_timeouts" returns no results
- `src/elspeth/engine/processor.py:639-653` - end-of-source flush calls coalesce but not timeout check

```bash
# Verification command
grep -r "check_timeouts" src/elspeth/engine/processor.py
# Returns: nothing
```

## Impact

- User-facing impact: Fork/join pipelines hang if any branch fails
- Data integrity / security impact: Incomplete audit trails for timed-out branches
- Performance or cost impact: Indefinite resource consumption

## Root Cause Hypothesis

- `check_timeouts()` was implemented in CoalesceExecutor but never integrated into the processor main loop
- The aggregation timeout check pattern exists but wasn't replicated for coalesce
- Missing integration test that exercises timeout during processing (not just end-of-source)

## Proposed Fix

- Code changes (modules/files):
  ```python
  # src/elspeth/engine/processor.py - in main processing loop

  # Add timeout check interval config
  # In ProcessorSettings or hardcoded initially:
  COALESCE_TIMEOUT_CHECK_INTERVAL_MS = 1000

  # In _process_loop(), add periodic timeout check:
  last_timeout_check = time.monotonic()

  while work_queue:
      # ... existing processing ...

      # Check coalesce timeouts periodically
      now = time.monotonic()
      if now - last_timeout_check > COALESCE_TIMEOUT_CHECK_INTERVAL_MS / 1000:
          for coalesce_name in self._coalesce_configs:
              outcomes = self._coalesce_executor.check_timeouts(
                  coalesce_name,
                  current_step
              )
              for outcome in outcomes:
                  self._handle_coalesce_outcome(outcome)
          last_timeout_check = now
  ```

- Config or schema changes:
  - Optional: Add `coalesce_timeout_check_interval_ms` to ProcessorSettings

- Tests to add/update:
  - `test_coalesce_timeout_fires_during_processing`
  - Test with 3-branch fork, 2 complete quickly, 1 delays past timeout
  - Verify timeout fires before end-of-source

- Risks or migration steps:
  - Low risk - adds behavior that was always intended
  - Existing pipelines without timeouts unaffected
  - Pipelines relying on "wait forever" behavior will now timeout (arguably correct)

## Architectural Deviations

- Spec or doc reference: plugin-protocol.md:830 - "Policy: `best_effort` - wait until timeout"
- Observed divergence: `best_effort` actually waits forever during processing
- Reason (if known): Implementation incomplete - method exists but not called

## Verification Criteria

- [x] `check_timeouts()` called periodically in processor main loop
- [x] `best_effort` policy fires at configured timeout
- [x] Timed-out branches recorded in audit trail
- [x] Integration test with deliberate branch delay

## Resolution

**Already fixed prior to bug report creation.**

The fix was implemented on 2026-01-22 (see comments in `orchestrator.py:1147-1149` and `orchestrator.py:2094-2096`).

**Evidence:**
- `orchestrator.py:1150-1159` - `coalesce_executor.check_timeouts()` called after processing each row in `_execute_run()`
- `orchestrator.py:2096-2106` - Same fix in `_process_resumed_rows()` for resume path
- Test `tests/engine/test_coalesce_integration.py::TestCoalesceTimeoutIntegration::test_best_effort_timeout_merges_during_processing` - PASSED
- Test `tests/engine/test_audit_sweep.py::TestAuditSweepForkCoalesce::test_timeout_triggered_coalesce_records_completed_outcome` - PASSED

**Note:** This bug report was likely generated by an automated release validation scan that examined code state before the fix commits were applied. The bug was already resolved when this report was filed.

## Cross-References

- RC1-remediation.md: CRIT-03
- requirements.md: SOP-015 (best_effort policy)
- docs/release/rc1-checklist.md: Section 2.3
