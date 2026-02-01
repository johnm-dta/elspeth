# Bug Report: PoolConfig allows invalid min/max dispatch delay combinations

## Summary

- PoolConfig does not validate `min_dispatch_delay_ms <= max_dispatch_delay_ms`. If `min_dispatch_delay_ms` exceeds `max_dispatch_delay_ms`, AIMDThrottle can set `current_delay_ms` above the configured maximum, breaking the "ceiling" guarantee and producing inconsistent throttling.

## Severity

- Severity: minor
- Priority: P3

## Reporter

- Name or handle: Codex
- Date: 2026-01-21
- Related run/issue ID: N/A

## Environment

- Commit/branch: ae2c0e6 / fix/rc1-bug-burndown-session-2
- OS: Linux
- Python version: Python 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: Deep dive src/elspeth/plugins/pooling for bugs; create bug reports.
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, no escalations
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: Code inspection only

## Steps To Reproduce

1. Create `PoolConfig(min_dispatch_delay_ms=1000, max_dispatch_delay_ms=100)`.
2. Instantiate `AIMDThrottle` with `pool_config.to_throttle_config()`.
3. Call `on_capacity_error()` then `on_success()`.

## Expected Behavior

- Invalid configs should be rejected, or delays should never exceed `max_dispatch_delay_ms`.

## Actual Behavior

- `on_success()` floors delay to `min_dispatch_delay_ms`, which can exceed the configured max; delay can oscillate above the ceiling.

## Evidence

- Config allows the invalid relationship: `src/elspeth/plugins/pooling/config.py:25-30`.
- Throttle floors to min after success: `src/elspeth/plugins/pooling/throttle.py:111-116`.
- Throttle caps to max on capacity error: `src/elspeth/plugins/pooling/throttle.py:98-99`.

## Impact

- User-facing impact: Misconfigured throttling can cause excessive delays or confusing behavior.
- Data integrity / security impact: None.
- Performance or cost impact: Increased latency and under-utilized throughput.

## Root Cause Hypothesis

- Missing cross-field validation in PoolConfig for min/max delay invariants.

## Proposed Fix

- Code changes (modules/files): `src/elspeth/plugins/pooling/config.py`
- Config or schema changes: Add a model validator to enforce `min_dispatch_delay_ms <= max_dispatch_delay_ms` (and optionally `recovery_step_ms <= max_dispatch_delay_ms`).
- Tests to add/update: Add validation tests in `tests/plugins/llm/test_pool_config.py`.
- Risks or migration steps: Invalid configs should fail fast with a clear error.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): N/A
- Observed divergence: "Ceiling" semantics are not enforced for invalid config combinations.
- Reason (if known): Cross-field validation not implemented.
- Alignment plan or decision needed: Enforce invariant in config model.

## Acceptance Criteria

- Configs with `min_dispatch_delay_ms > max_dispatch_delay_ms` raise validation errors.
- Throttle delay never exceeds `max_dispatch_delay_ms`.

## Tests

- Suggested tests to run: `pytest tests/plugins/llm/test_pool_config.py -k validation`
- New tests required: Yes (min/max invariant).

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: N/A

## Verification (2026-01-25)

**Status: STILL VALID**

### Verification Steps

1. Examined current `/home/john/elspeth-rapid/src/elspeth/plugins/pooling/config.py` (lines 25-30)
2. Checked git history for fixes since 2026-01-21 bug report date
3. Tested invalid configuration acceptance
4. Confirmed throttle behavior exceeds max delay with invalid config
5. Verified no validation tests exist for min/max invariant

### Findings

**Bug is fully reproducible on current HEAD:**

```python
# Invalid config is accepted without error
config = PoolConfig(
    pool_size=10,
    min_dispatch_delay_ms=1000,
    max_dispatch_delay_ms=100
)
# Config created: min=1000, max=100 ✓ (no validation error)

# Throttle behavior confirms the bug
throttle = AIMDThrottle(config.to_throttle_config())
throttle.on_capacity_error()  # Sets delay to 100 (capped at max)
throttle.on_success()         # Floors delay to 1000 (min)
# Result: delay (1000.0) exceeds configured max (100) ✓
```

**Code Analysis:**

- `/home/john/elspeth-rapid/src/elspeth/plugins/pooling/config.py` has NO Pydantic model validators
- No `@model_validator` or `@field_validator` decorators present
- Individual field constraints exist (`ge=0`, `gt=1.0`) but no cross-field validation
- PoolConfig is a Pydantic BaseModel with `extra="forbid"` but lacks invariant enforcement

**Git History:**

- Only 2 commits touched pooling files since 2026-01-21:
  - `c786410` - ELSPETH - Release Candidate 1 (no validation changes)
  - `a5196c1` - refactor: use PEP 695 type parameter syntax for generics (no validation changes)
- Original code from `0b1cf47` (refactor: extract pooling infrastructure) had no validation
- No validation was added in subsequent commits

**Test Coverage:**

- `/home/john/elspeth-rapid/tests/plugins/llm/test_pool_config.py` has validation tests for:
  - `pool_size >= 1` ✓
  - `backoff_multiplier > 1` ✓
  - `max_capacity_retry_seconds > 0` ✓
- **Missing:** No test for `min_dispatch_delay_ms <= max_dispatch_delay_ms` invariant

**Impact Confirmed:**

The bug allows configurations that violate the "ceiling" guarantee. When `min > max`:
- `on_capacity_error()` caps delay at max (100ms in test)
- `on_success()` floors delay at min (1000ms in test)
- Delay oscillates between 100ms and 1000ms, never respecting the 100ms "ceiling"
- This breaks expected throttling behavior and can cause performance issues

**Recommendation:**

Bug should remain **P3** (minor severity) as it requires misconfiguration to trigger, but should be fixed to prevent confusing production behavior. The fix is straightforward: add a Pydantic `@model_validator` to enforce the invariant and reject invalid configs at construction time.
