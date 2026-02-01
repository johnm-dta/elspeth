# Bug Report: RetryManager on_retry attempt numbers are 1-based, while audit uses 0-based

## Summary

- `RetryManager.execute_with_retry()` passes Tenacity's 1-based `attempt_number` to `on_retry`, but engine audit attempt numbering is 0-based (first attempt = 0), so callback consumers will record misaligned attempt indices.

## Severity

- Severity: minor
- Priority: P3

## Reporter

- Name or handle: Codex
- Date: 2026-01-21
- Related run/issue ID: N/A

## Environment

- Commit/branch: ae2c0e6f088f467276582fa8016f91b4d3bb26c7 (fix/rc1-bug-burndown-session-2)
- OS: Linux nyx.foundryside.dev 6.8.0-90-generic
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: Deep dive into src/elspeth/engine/retry.py for bugs.
- Model/version: Codex (GPT-5)
- Tooling and permissions (sandbox/approvals): Codex CLI, workspace-write sandbox
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: Manual code inspection only

## Steps To Reproduce

1. Use `RetryManager.execute_with_retry()` with an `on_retry` callback to record attempt numbers.
2. Compare recorded attempt numbers to node state attempts (`attempt=0` for first attempt).
3. Observe mismatch: on_retry reports 1 for the first failure, audit uses 0 for first attempt.

## Expected Behavior

- Retry callbacks should use the same attempt indexing as audit records (0-based), or explicitly document a different convention.

## Actual Behavior

- `on_retry` receives Tenacity's 1-based attempt numbers.

## Evidence

- `src/elspeth/engine/retry.py` sets `attempt = attempt_state.retry_state.attempt_number` and passes it to `on_retry`.
- Audit recorder docs: `src/elspeth/core/landscape/recorder.py` notes “attempt number (0 for first attempt)”.

## Impact

- User-facing impact: Retry audit hooks can produce off-by-one attempt indices.
- Data integrity / security impact: Potential mismatch or conflicts if attempt numbers are used in unique keys or lineage.
- Performance or cost impact: None.

## Root Cause Hypothesis

- RetryManager uses tenacity's attempt numbering without normalizing to engine conventions.

## Proposed Fix

- Code changes (modules/files):
  - Normalize attempt number before calling `on_retry` (e.g., `attempt_number - 1`).
  - Or document that `on_retry` is 1-based and update consumers accordingly.
- Config or schema changes: None.
- Tests to add/update:
  - Add test asserting callback attempt numbering matches audit convention.
- Risks or migration steps: If changing numbering, update any existing consumers.

## Architectural Deviations

- Spec or doc reference: `src/elspeth/core/landscape/recorder.py` (0-based attempts).
- Observed divergence: Retry callback uses 1-based attempts.
- Reason (if known): Direct passthrough from tenacity.
- Alignment plan or decision needed: Standardize attempt indexing.

## Acceptance Criteria

- Retry callback attempt numbers align with audit attempt indexing.

## Tests

- Suggested tests to run: `pytest tests/engine/test_retry.py -k records_attempts`
- New tests required: Yes.

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: N/A

---

## VERIFICATION: 2026-02-01

**Status:** STILL VALID

- RetryManager still passes tenacity’s 1-based `attempt_number` to `on_retry` without normalization. (`src/elspeth/engine/retry.py:114-121`)

## VERIFICATION: 2026-01-25

**Status:** STILL VALID (but latent/not currently exploited)

**Verified By:** Claude Code P3 verification wave 5

**Current Code Analysis:**

Examined `/home/john/elspeth-rapid/src/elspeth/engine/retry.py` at lines 164-171:
- Line 164: `attempt = attempt_state.retry_state.attempt_number` (tenacity's 1-based number)
- Line 171: `on_retry(attempt, e)` (passes 1-based number directly)

Examined `/home/john/elspeth-rapid/src/elspeth/core/landscape/recorder.py`:
- Documentation at line 1038: "attempt: Attempt number (0 for first attempt)"
- Documentation at line 1340: "attempt: Attempt number (0 for first attempt)"
- Landscape audit system expects 0-based attempt numbering

Examined `/home/john/elspeth-rapid/src/elspeth/engine/processor.py` at lines 448-470:
- RowProcessor does NOT use the `on_retry` callback parameter
- Instead implements its own 0-based `attempt_tracker` workaround (lines 449-453)
- Passes 0-based attempt to `execute_transform()` at line 459
- This workaround avoids the bug but leaves the API contract broken

Examined `/home/john/elspeth-rapid/tests/engine/test_retry.py` at lines 62-86:
- Test `test_records_attempts` documents the WRONG behavior
- Line 85 asserts `attempts[0][0] == 1`, expecting 1-based numbering from first retry
- This test validates the bug rather than the correct behavior

**Git History:**

- Bug filed 2026-01-21 at commit ae2c0e6
- Checked commits since 2026-01-21: no fixes applied to retry.py for this issue
- The `attempt_tracker` workaround existed from RC1 (commit c786410, 2026-01-22)
- No commits mention "1-based", "0-based", or attempt indexing normalization

**Root Cause Confirmed:**

YES - the bug is still present in the code:

1. **API Contract Violation:** `RetryManager.execute_with_retry()` accepts an `on_retry` callback but passes 1-based attempt numbers from tenacity, violating the system-wide 0-based convention
2. **Latent Bug:** Currently not exploited because RowProcessor doesn't use `on_retry` - it implements a workaround
3. **Test Documents Wrong Behavior:** The test validates that `on_retry` receives 1-based numbers, which is the bug
4. **Future Risk:** If any code starts using the `on_retry` callback parameter, it will receive misaligned attempt numbers

**Recommendation:**

**Keep open** - this is a valid API contract violation that should be fixed:

**Fix Strategy:**
1. Normalize attempt number in retry.py line 171: `on_retry(attempt - 1, e)`
2. Update test at line 85 to expect 0-based: `assert attempts[0][0] == 0`
3. Update docstring for `on_retry` parameter to document 0-based convention
4. Consider adding a test that verifies `on_retry` attempt numbers match what would be passed to `recorder.record_node_state()`

**Rationale:** While not currently causing failures, this violates the principle of least surprise and creates a landmine for future developers who might use `on_retry` expecting system-standard 0-based numbering.

---

## RESOLUTION: 2026-02-02

**Status:** FIXED

**Fixed By:** Claude Code (Opus 4.5)

**Fix Summary:**

Moved `on_retry` callback invocation from manual call inside the attempt block to tenacity's `before_sleep` hook, which:

1. **Normalizes to 0-based:** `on_retry(retry_state.attempt_number - 1, exc)` converts tenacity's 1-based to audit convention
2. **Fires only when retry scheduled:** `before_sleep` is only called when tenacity will actually sleep before another attempt
3. **Documents convention:** Updated docstring to explicitly document 0-based attempt numbering

**Files Changed:**
- `src/elspeth/engine/retry.py` - Added `before_sleep_handler`, updated docstring
- `tests/engine/test_retry.py` - Renamed test and fixed assertion to expect 0-based

**Tests Added:**
- `test_on_retry_uses_zero_based_attempts` - Verifies 0-based numbering
- `test_on_retry_not_called_on_final_attempt` - Verifies no callback with max_attempts=1
- `test_on_retry_not_called_on_exhausted_retries` - Verifies callback count with max_attempts=3
