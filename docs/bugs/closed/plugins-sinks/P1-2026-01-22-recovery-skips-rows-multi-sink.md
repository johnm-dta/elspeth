# Bug Report: Recovery skips rows for sinks written later due to row_index checkpoint boundary

## Summary

`RecoveryManager.get_unprocessed_rows` uses the row_index of the latest checkpointed token as a single boundary. Because checkpoints are created after sink writes in sink order, the latest checkpoint can correspond to an earlier row than some rows written to other sinks, causing resume to skip rows routed to a later/failed sink and leaving outputs missing.

## Severity

- Severity: critical
- Priority: P1

## Reporter

- Name or handle: Codex (static analysis agent)
- Date: 2026-01-22
- Related run/issue ID: Unknown

## Environment

- Commit/branch: main (d8df733)
- OS: Linux
- Python version: 3.12+
- Config profile / env vars: Pipeline with multiple sinks and checkpoint frequency `every_row`
- Data set or fixture: Rows routed to multiple sinks

## Agent Context (if relevant)

- Goal or task prompt: Static analysis agent doing a deep bug audit of `src/elspeth/core/checkpoint/recovery.py`
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): Read-only sandbox, approvals never
- Determinism details (seed, run ID): Unknown
- Notable tool calls or steps: Read recovery.py, manager.py, orchestrator.py, executors.py, enums.py, CLAUDE.md

## Steps To Reproduce

1. Configure a pipeline with two sinks (`sink_a` default, `sink_b` via gate) and checkpoint frequency `every_row`
2. Run with rows routed to both sinks; force `sink_b.write()` to raise after `sink_a` succeeds (simulate sink failure)
3. Call `RecoveryManager.get_unprocessed_rows(run_id)` and resume
4. Observe rows routed to `sink_b` are not returned/resumed

## Expected Behavior

- Recovery should include rows whose tokens never reached a completed sink node_state (including rows routed to `sink_b`)
- Resume should write the missing sink outputs

## Actual Behavior

- `get_unprocessed_rows` uses the latest checkpoint's token row_index and returns only rows with row_index greater than that
- Rows routed to the failed/later sink are skipped, leaving their outputs missing

## Evidence

- Logs or stack traces: Unknown
- Artifacts (paths, IDs, screenshots):
  - `src/elspeth/core/checkpoint/recovery.py:223`
  - `src/elspeth/core/checkpoint/recovery.py:250`
  - `src/elspeth/core/checkpoint/manager.py:93`
  - `src/elspeth/engine/orchestrator.py:132`
  - `src/elspeth/engine/orchestrator.py:885`
  - `src/elspeth/engine/executors.py:1337`
- Minimal repro input (attach or link): Multi-sink pipeline with forced sink failure

## Impact

- User-facing impact: Resume can finish without emitting outputs for some sinks, even though the run reports completion
- Data integrity / security impact: Audit trail implies sink outputs were produced, but artifacts are missing for routed rows; violates auditability guarantees
- Performance or cost impact: Operators may rerun or manually backfill, risking duplicate writes and extra compute

## Root Cause Hypothesis

Recovery assumes a single monotonic row_index boundary derived from the latest checkpoint, but checkpoints are ordered by sequence_number (token write order) which is not aligned with row_index across multiple sinks. This causes rows routed to later/failed sinks to be skipped.

## Proposed Fix

- Code changes (modules/files): In `src/elspeth/core/checkpoint/recovery.py`, compute unprocessed rows by identifying tokens lacking a completed sink node_state (join `tokens_table`, `node_states_table`, and `nodes_table` where node_type="sink") and map those tokens to row_ids, instead of using a single row_index boundary from the latest checkpoint. Optionally keep the current boundary as an optimization only when there is a single sink.
- Config or schema changes: None
- Tests to add/update: Add a multi-sink recovery test where one sink fails after another succeeds; verify rows routed to the failed sink are returned by `get_unprocessed_rows`
- Risks or migration steps: May reprocess some rows and create duplicate sink outputs if sinks are not idempotent; document expected behavior or add idempotency keys

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `CLAUDE.md:28` - Sink output is a non-negotiable data storage point
- Observed divergence: Resume can mark runs completed while some sink outputs are missing for routed rows
- Reason (if known): Recovery uses a row_index boundary based on the last checkpoint rather than actual sink completion per token
- Alignment plan or decision needed: Decide whether recovery should be token/sink-state-based (accurate) or enforce per-row sink write ordering/checkpointing

## Acceptance Criteria

- A multi-sink recovery test (with sink failure) returns rows routed to the failed sink
- Resume emits their outputs
- No missing sink artifacts after resume

## Tests

- Suggested tests to run: `tests/core/checkpoint/test_recovery.py`, `tests/integration/test_checkpoint_recovery.py`
- New tests required: Yes, multi-sink recovery scenario with a forced sink failure

## Notes / Links

- Related issues/PRs: Unknown
- Related design docs: `CLAUDE.md:28`

## Verification Status

- [ ] Bug confirmed via reproduction
- [ ] Root cause verified
- [ ] Fix implemented
- [ ] Tests added
- [ ] Fix verified

## Verification Status (2026-01-24)

**Status**: STILL VALID

**Verified by**: Automated verification agent

**Current code state**:
The bug remains unfixed in the current codebase. Analysis confirms:

1. **Recovery logic unchanged**: `RecoveryManager.get_unprocessed_rows()` (lines 233-285 in `/home/john/elspeth-rapid/src/elspeth/core/checkpoint/recovery.py`) still uses the single row_index boundary approach:
   - Line 250: Gets latest checkpoint
   - Lines 257-275: Derives row_index from checkpoint token
   - Lines 277-283: Returns only rows with `row_index > checkpointed_row_index`

2. **No sink-state awareness**: The method does NOT query `node_states_table` or check which tokens have completed sink writes. It purely relies on row_index ordering.

3. **Prior verification confirms bug**: A comprehensive verification report from 2026-01-22 (`docs/bugs/generated/VERIFICATION_P1_recovery_skips_rows.md`) analyzed this bug in detail and concluded:
   - **VERIFIED** - Bug is real and manifests in interleaved sink routing scenarios
   - Example: Row 0 → sink_a, Row 1 → sink_b, Row 2 → sink_a
   - If sink_a writes first (rows 0, 2) and sink_b fails (row 1), recovery skips row 1
   - Latest checkpoint would be row_index=2, so `get_unprocessed_rows` returns empty list
   - Row 1's output is permanently lost

4. **Related fix did NOT address this bug**: Commit b2a3518 (2026-01-23) modified `recovery.py` but only added type fidelity preservation for resume (source schema validation). It did not change the row boundary computation logic.

5. **Related but distinct bug**: Bug P0-2026-01-19-checkpoint-before-sink-write (now closed) addressed checkpoint timing (creating checkpoints AFTER sink writes), which is a prerequisite for this fix but does not solve the multi-sink interleaving problem.

**Trigger conditions**:
- Pipeline with 2+ sinks
- Rows routed to different sinks in non-contiguous order (interleaved)
- One sink fails after another succeeds
- Checkpoint frequency enabled

**Impact**: Critical data loss - rows routed to failed sinks are silently skipped on resume, violating audit integrity guarantees. The audit trail may show row processing succeeded, but sink artifacts are missing.

**Recommendation**: KEEP OPEN

This is a valid P1 bug requiring implementation of the proposed fix: query for tokens lacking completed sink node_states rather than using row_index boundary. A test case for interleaved multi-sink recovery should be added before implementing the fix.

---

## CLOSURE: 2026-01-28

**Status:** FIXED

**Fixed By:** Unknown (discovered during bug audit)

**Resolution:**

The fix was implemented in `src/elspeth/core/checkpoint/recovery.py`. The `get_unprocessed_rows()` method (lines 233-310) was completely rewritten to use token outcomes instead of the row_index boundary approach.

**Key changes:**

1. **New approach:** Query for rows whose tokens do NOT have terminal outcomes (COMPLETED, ROUTED, QUARANTINED, FAILED)
2. **Explicit bug reference:** Code includes comment at line 262-267 citing this bug ID
3. **Multi-sink safe:** The new approach correctly handles interleaved sink routing

**Code excerpt (lines 257-267):**
```python
# Strategy: Find rows whose tokens do NOT have terminal sink outcomes.
#
# A token is "complete" if it has a terminal outcome (is_terminal=1)
# that indicates it reached a sink (COMPLETED or ROUTED).
#
# BUG FIX (P1-2026-01-22-recovery-skips-rows-multi-sink):
# Previous approach used row_index boundary from checkpoint, which
# failed in multi-sink scenarios where rows interleave between sinks.
# Example: Row 0→sink_a (done), Row 1→sink_b (failed), Row 2→sink_a (done)
# Old code: checkpoint at row 2, return rows > 2, miss row 1
# New code: return rows without terminal outcomes, includes row 1
```

**Verification:**

The fix correctly handles the scenario described in the bug:
- Row 0 → sink_a (COMPLETED) - not reprocessed
- Row 1 → sink_b (no terminal outcome) - reprocessed ✓
- Row 2 → sink_a (COMPLETED) - not reprocessed
