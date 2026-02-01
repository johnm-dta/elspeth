# Bug Report: Coalesce timeout/incomplete failures are never recorded in audit

## Summary

- `flush_pending()` returns failure outcomes for `require_all` and `quorum` but does not record any audit state or token outcomes.
- `check_timeouts()` only merges when quorum is met; if quorum is not met at timeout, no failure is recorded and the pending entry persists.

## Severity

- Severity: major
- Priority: P2

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
- Notable tool calls or steps: code inspection of coalesce executor and orchestrator

## Steps To Reproduce

1. Configure a coalesce with `policy: require_all` and a timeout.
2. Send only one branch token, then end the source (or hit timeout for `check_timeouts`).
3. Inspect the audit trail for failure records for the pending coalesce.

## Expected Behavior

- Missing branches should be recorded explicitly; failures should be recorded in audit (and missing branches quarantined if policy dictates).

## Actual Behavior

- Failures are returned to the caller but never recorded; the audit trail has no record of missing branches or timeout failures.

## Evidence

- `flush_pending()` creates failure outcomes without recording them: `src/elspeth/engine/coalesce_executor.py:421`
- `check_timeouts()` ignores quorum-not-met timeouts (no failure recorded): `src/elspeth/engine/coalesce_executor.py:357`
- Orchestrator assumes failures are recorded by executor: `src/elspeth/engine/orchestrator.py:878`
- Design expects missing branches to be recorded/quarantined: `docs/design/subsystems/00-overview.md#L322`

## Impact

- User-facing impact: explain/replay cannot show why a coalesce failed or which branches were missing.
- Data integrity / security impact: audit trail is incomplete; missing branches are not recorded.
- Performance or cost impact: pending entries can linger and grow without resolution.

## Root Cause Hypothesis

- Failure paths return `CoalesceOutcome` only and never call `LandscapeRecorder` to persist failure details.

## Proposed Fix

- Code changes (modules/files):
  - Record explicit failure outcomes for all arrived tokens when coalesce fails (require_all/quorum).
  - Record missing branches in audit metadata and consider quarantining or failure outcomes for missing branches per policy.
- Config or schema changes: none.
- Tests to add/update:
  - Add tests for require_all/quorum timeout failure recording.
- Risks or migration steps:
  - Decide how missing branches are represented in audit (failure vs quarantine) and document the behavior.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `docs/design/subsystems/00-overview.md#L322`
- Observed divergence: missing branches are not recorded; failures are silent.
- Reason (if known): executor returns outcomes but does not persist failure records.
- Alignment plan or decision needed: define failure recording semantics and implement them.

## Acceptance Criteria

- Coalesce failures create audit records indicating missing branches and policy decision.
- No pending coalesce remains after a timeout without a recorded resolution.

## Tests

- Suggested tests to run:
  - `pytest tests/engine/test_coalesce_executor.py -k failure`
- New tests required: yes (timeout failure recording)

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `docs/design/subsystems/00-overview.md`

---

## VERIFICATION: 2026-01-25

**Status:** STILL VALID

**Verified By:** Claude Code P2 verification wave 2

**Current Code Analysis:**

Examined the current implementation in commit `a2698bd` (HEAD of fix/rc1-bug-burndown-session-4 branch):

### 1. Failure Outcomes Are Created But Never Recorded

**Location:** `src/elspeth/engine/coalesce_executor.py:421-434` (quorum_not_met) and lines 436-449 (incomplete_branches)

When `flush_pending()` is called and coalesce policies fail:
- `quorum` policy: If quorum not met, creates `CoalesceOutcome` with `failure_reason="quorum_not_met"`
- `require_all` policy: Creates `CoalesceOutcome` with `failure_reason="incomplete_branches"`

**Critical Gap:** These failure outcomes:
1. Do NOT populate `consumed_tokens` field (left as empty list)
2. Do NOT call `recorder.begin_node_state()` or `recorder.complete_node_state()` for any arrived tokens
3. Only populate `coalesce_metadata` dict with policy info and branches_arrived list

**Contrast with successful merge (lines 236-249):**
```python
# Record node states for consumed tokens
for token in consumed_tokens:
    state = self._recorder.begin_node_state(
        token_id=token.token_id,
        node_id=node_id,
        step_index=step_in_pipeline,
        input_data=token.row_data,
    )
    self._recorder.complete_node_state(
        state_id=state.state_id,
        status="completed",
        output_data={"merged_into": merged_token.token_id},
        duration_ms=0,
    )
```

This recording is completely absent from failure paths.

### 2. Orchestrator False Comment

**Location:** `src/elspeth/engine/orchestrator.py:1048-1053`

```python
elif outcome.failure_reason:
    # Coalesce failed (timeout, missing branches, etc.)
    # Failure is recorded in audit trail by executor.  <-- FALSE
    # Not counted as rows_failed since the individual fork children
    # were already counted when they reached their terminal states.
    pass
```

The comment claims "Failure is recorded in audit trail by executor" but this is demonstrably false. The executor only returns the outcome object; it never calls any `recorder` methods on the failure path.

### 3. check_timeouts() Has Same Gap

**Location:** `src/elspeth/engine/coalesce_executor.py:357-369`

The `check_timeouts()` method only calls `_execute_merge()` if quorum is met (line 360-368). If timeout occurs but quorum is NOT met:
- The pending entry remains in `self._pending` (not cleaned up)
- No failure outcome is created
- No audit record is made

**Related Bug:** Per the already-verified bug P1-2026-01-22-coalesce-timeouts-never-fired.md, `check_timeouts()` is never called by the orchestrator anyway, so this is currently a theoretical issue but would become real once timeouts are wired up.

### 4. Tokens in pending.arrived Are Lost

When a coalesce fails, the tokens that DID arrive are in `pending.arrived` dict (lines 431, 446). These tokens:
- Successfully reached the coalesce point
- Were held waiting for missing branches
- Are deleted when `del self._pending[key]` executes (lines 423, 438)
- Have NO node state recorded
- Have NO terminal outcome recorded

This violates the "no silent drops" principle - these tokens disappear from the audit trail.

**Git History:**

Searched for relevant fixes since bug report date (2026-01-22):
```bash
git log --all --oneline --since="2026-01-22" -- src/elspeth/engine/coalesce_executor.py
git log --all --oneline --grep="coalesce.*failure\|coalesce.*timeout\|coalesce.*record" -i
```

**Result:** No commits addressed failure outcome recording. Recent commits to these files were:
- `935ee6b` - cleanup: delete ExecutionGraph.from_config() method
- `0a9cf2a` - fix(audit): record COMPLETED outcomes with sink_name for disambiguation
- Various observability and phase event additions

None addressed the missing audit recording for coalesce failures.

**Root Cause Confirmed:**

The bug is exactly as described in the original report:

## Verification (2026-02-01)

**Status: FIXED**

- `check_timeouts()` now records FAILED node states and token outcomes when quorum is not met at timeout. (`src/elspeth/engine/coalesce_executor.py:540-579`)
- `flush_pending()` now records FAILED node states and token outcomes for `quorum` and `require_all` failure paths. (`src/elspeth/engine/coalesce_executor.py:640-738`)
- Failure outcomes set `outcomes_recorded=True`, matching orchestrator expectations. (`src/elspeth/engine/coalesce_executor.py:676-738`)

## Closure Report (2026-02-01)

**Status:** CLOSED (FIXED)

### Closure Notes

- Coalesce failure paths now complete pending node states and record terminal outcomes with failure reasons and error hashes.
- Pending entries are removed and marked completed, preventing silent drops after timeouts.

---

## Verification (2026-02-01)

**Status: FIXED**

- Failure paths now record node states and token outcomes for quorum/require_all failures in `flush_pending()`, and quorum timeout failures in `check_timeouts()`. (`src/elspeth/engine/coalesce_executor.py:530-579`, `src/elspeth/engine/coalesce_executor.py:611-735`)

1. Failure outcomes are created with metadata but never persisted to Landscape
2. Tokens that arrived at a failed coalesce have no node states recorded
3. The orchestrator incorrectly assumes failures are recorded (line 1050 comment)
4. Tests verify that failure outcomes are RETURNED but do not verify audit trail recording

**Tests verified this gap:**
- `tests/engine/test_coalesce_executor.py:975` - asserts `failure_reason == "quorum_not_met"` but does NOT query recorder
- `tests/engine/test_coalesce_executor.py:1054` - asserts `failure_reason == "incomplete_branches"` but does NOT query recorder
- No test calls `recorder.get_node_states()` after a failure outcome

**Recommendation:**

Keep open - STILL VALID

**Fix Required:**
1. In `flush_pending()` failure paths (lines 421-434, 436-449), add recording similar to successful merge:
   - Loop through `pending.arrived.values()` tokens
   - Call `recorder.begin_node_state()` and `recorder.complete_node_state()` with status indicating failure
   - Populate `consumed_tokens` field in the returned outcome
2. Update orchestrator.py line 1050 comment to reflect actual behavior
3. Add test that queries audit trail after coalesce failure to verify node states exist

**Audit Impact:**
Without this fix, the audit trail cannot answer:
- "Which tokens arrived at coalesce X before it failed?"
- "What data did those tokens contain?"
- "Why did row Y never reach the output sink?" (if it was stuck in failed coalesce)
