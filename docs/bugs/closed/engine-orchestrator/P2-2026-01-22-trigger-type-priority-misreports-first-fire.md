# Bug Report: Trigger type misreported when multiple triggers are true

## Summary

- `TriggerEvaluator.should_trigger()` checks count, then timeout, then condition. When multiple triggers are true at evaluation time, it always reports the first in that fixed order, even if another trigger would have fired earlier. This violates the "first one to fire wins" contract and mislabels `trigger_type` in the audit trail.

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
- Notable tool calls or steps: inspected trigger ordering and audit usage

## Steps To Reproduce

1. Configure an aggregation trigger with `count: 100` and `timeout_seconds: 1`.
2. Accept 99 rows quickly (<1s), then wait >1s to exceed the timeout.
3. Accept the 100th row; `should_trigger()` returns `True` with `which_triggered() == "count"`.

## Expected Behavior

- `TriggerType.TIMEOUT` should be recorded because the timeout elapsed before the count threshold was reached.

## Actual Behavior

- `TriggerType.COUNT` is recorded due to fixed ordering in `should_trigger()`.

## Evidence

- Fixed trigger order: `src/elspeth/engine/triggers.py:95-116`
- "First one to fire wins" contract: `docs/contracts/plugin-protocol.md:1213`

## Impact

- User-facing impact: misleading trigger type in batch audit metadata.
- Data integrity / security impact: audit trail attributes batches to the wrong trigger.
- Performance or cost impact: none directly, but mislabels can obscure tuning.

## Root Cause Hypothesis

- Trigger evaluator does not track when each condition first becomes true; it short-circuits on a fixed priority order.

## Proposed Fix

- Code changes (modules/files):
  - Track the timestamp when each trigger first becomes true.
  - When multiple triggers are satisfied, select the one with the earliest fire time.
- Config or schema changes: none.
- Tests to add/update:
  - Add a test where timeout elapses before count threshold; assert trigger_type is TIMEOUT.
- Risks or migration steps:
  - Ensure the chosen trigger type remains stable across re-evaluations.

## Architectural Deviations

- Spec or doc reference: `docs/contracts/plugin-protocol.md:1213`
- Observed divergence: fixed priority order contradicts "first to fire wins."
- Reason (if known): order-based short-circuiting for simplicity.
- Alignment plan or decision needed: define tie-breaking rules for simultaneous triggers.

## Acceptance Criteria

- When multiple triggers are true, the recorded trigger type reflects the earliest trigger to fire.

## Tests

- Suggested tests to run: `pytest tests/engine/test_triggers.py -k combined`
- New tests required: yes

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `docs/contracts/plugin-protocol.md`

## Verification (2026-01-25)

**Status: STILL VALID**

The bug is confirmed to still exist in the current codebase on branch `fix/rc1-bug-burndown-session-4`.

### Verification Test

Executed the exact reproduction scenario:

```python
from elspeth.core.config import TriggerConfig
from elspeth.engine.triggers import TriggerEvaluator

config = TriggerConfig(count=100, timeout_seconds=1.0)
evaluator = TriggerEvaluator(config)

# Accept 99 rows quickly
for _ in range(99):
    evaluator.record_accept()

# Wait for timeout to elapse (1.1 seconds)
time.sleep(1.1)

# Accept 100th row - now BOTH conditions are true
evaluator.record_accept()

# Result:
# Triggered: True
# Which: count  # <-- BUG: should be 'timeout'
# Batch count: 100
# Batch age: 1.10s
```

### Analysis

1. **Code Location**: The issue is in `src/elspeth/engine/triggers.py:95-118` in the `should_trigger()` method.

2. **Root Cause Confirmed**: The method uses short-circuit evaluation with a fixed priority order:
   - First checks count (line 96)
   - Then checks timeout (line 101)
   - Finally checks condition (line 106)

   When multiple triggers are simultaneously true at evaluation time, it always returns the first one in this priority order, regardless of which trigger became true first.

3. **Contract Violation**: This violates the documented contract in `docs/contracts/plugin-protocol.md:1213`: "Multiple triggers can be combined (first one to fire wins)."

4. **Git History**: No fixes found. The file was added in commit `c786410` (RC1) on 2026-01-22 and has not been modified since to address this issue.

5. **Test Coverage**: The existing tests in `tests/engine/test_triggers.py` do test combined triggers but only verify scenarios where one trigger is clearly dominant:
   - `test_combined_count_and_timeout_count_wins`: Count reaches threshold before timeout
   - `test_combined_count_and_timeout_timeout_wins`: Timeout elapses before count threshold

   However, there is NO test for the edge case where both conditions become true simultaneously (e.g., timeout elapsed, then count reached on the same evaluation).

### Impact Confirmation

- **Audit Trail Accuracy**: The trigger_type metadata in the batch audit trail will be incorrect when multiple triggers are true simultaneously.
- **Behavioral Impact**: If timeout elapsed at 1.0s but count threshold reached at 1.1s, the audit trail would incorrectly attribute the batch flush to COUNT rather than TIMEOUT.
- **Diagnostic Impact**: This makes it difficult to tune trigger configurations, as operators can't trust which trigger actually fired first.

### Missing Test Case

The bug report correctly identifies that a test is needed for: "timeout elapses before count threshold; assert trigger_type is TIMEOUT"

This test does NOT currently exist in the test suite.

## Verification (2026-02-01)

**Status: STILL VALID**

- `TriggerEvaluator.should_trigger()` still checks count then timeout then condition with fixed ordering, so it reports the first match instead of earliest trigger. (`src/elspeth/engine/triggers.py:115-136`)

## Resolution (2026-02-02)

**Status: FIXED**

### Implementation

Modified `TriggerEvaluator` to track when each trigger first fires and select the earliest:

1. **New state fields** (`triggers.py:67-68`):
   - `_count_fire_time: float | None` - tracks when count threshold was first reached
   - `_condition_fire_time: float | None` - tracks when condition first became true

2. **Fire time tracking in `record_accept()`** (`triggers.py:114-129`):
   - Records timestamp when count threshold is first reached
   - Records timestamp when condition first becomes true
   - Only sets fire time once (first occurrence)

3. **Fire-time-based selection in `should_trigger()`** (`triggers.py:131-175`):
   - Collects all triggered candidates with their fire times
   - For timeout: fire time is deterministic (`first_accept_time + timeout_seconds`)
   - For count: fire time tracked in `record_accept()`
   - For condition: re-evaluates in `should_trigger()` to handle time-dependent conditions
   - Sorts by fire time and selects earliest ("first to fire wins")

4. **Reset clears fire times** (`triggers.py:210-211`)

### Key Design Decision: Lazy Condition Evaluation

Conditions involving `batch_age_seconds` can become true due to time passing, not just row accepts. The fix re-evaluates conditions in `should_trigger()` when they haven't fired yet, using current time as a conservative fire time estimate when first detected. This ensures time-dependent conditions work correctly.

### Tests Added

Added `TestTriggerFirstToFireWins` class with 4 tests (`tests/engine/test_triggers.py:240-398`):
- `test_timeout_fires_before_count_reports_timeout` - timeout at t=1.0s, count at t=1.1s → TIMEOUT wins
- `test_count_fires_before_timeout_reports_count` - count at t=0.1s, timeout at t=5.0s → COUNT wins
- `test_condition_fires_before_timeout_reports_condition` - condition at t=0.5s, timeout at t=2.0s → CONDITION wins
- `test_timeout_fires_before_condition_reports_timeout` - timeout at t=1.0s, condition at t=1.5s → TIMEOUT wins

### Verification

```bash
.venv/bin/python -m pytest tests/engine/test_triggers.py -v
# 19 passed
```
