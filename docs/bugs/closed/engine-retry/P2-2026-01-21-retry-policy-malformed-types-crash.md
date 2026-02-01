# Bug Report: RetryConfig.from_policy crashes on malformed types despite "graceful" contract

**STATUS: RESOLVED** (2026-02-02)

## Summary

- `RetryConfig.from_policy()` claims to handle malformed policy values gracefully, but non-numeric values (e.g., strings or None) raise `TypeError` when passed through `max()`.

## Severity

- Severity: major
- Priority: P2

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

1. Call `RetryConfig.from_policy({"max_attempts": "3", "base_delay": "1"})`.
2. Observe a `TypeError` (e.g., comparing `int` and `str`) instead of graceful clamping.

## Expected Behavior

- Malformed policy values are sanitized or rejected with a clear, typed error at the boundary (no raw `TypeError`).

## Actual Behavior

- `max()` is called on potentially non-numeric values, raising `TypeError`.

## Evidence

- `src/elspeth/engine/retry.py`: `max(1, policy.get("max_attempts", 3))` and similar for delays.
- Docstring explicitly states the method handles malformed policy gracefully.

## Impact

- User-facing impact: Mis-typed config crashes initialization instead of producing a clear validation error.
- Data integrity / security impact: None.
- Performance or cost impact: None.

## Root Cause Hypothesis

- `from_policy` assumes numeric types and applies `max()` without validation or coercion.

## Proposed Fix

- Code changes (modules/files):
  - Validate types and either coerce or raise `PluginConfigError` with actionable message.
  - Treat non-numeric values as missing and fall back to defaults.
- Config or schema changes: None.
- Tests to add/update:
  - Add tests covering string/None values in policy dict.
- Risks or migration steps: None.

## Architectural Deviations

- Spec or doc reference: `src/elspeth/engine/retry.py` docstring (graceful handling at trust boundary).
- Observed divergence: Crashes on malformed types.
- Reason (if known): Missing validation.
- Alignment plan or decision needed: Define boundary behavior for malformed retry policy values.

## Acceptance Criteria

- Malformed policy values do not raise `TypeError`; they are sanitized or yield a clear validation error.

## Tests

- Suggested tests to run: `pytest tests/engine/test_retry.py -k from_policy`
- New tests required: Yes.

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: N/A

---

## VERIFICATION: 2026-02-01

**Status:** STILL VALID

- `RuntimeRetryConfig.from_policy()` still casts policy values with `int()`/`float()` without guarding non-numeric types, so malformed values still raise. (`src/elspeth/contracts/config/runtime.py:163-180`)

## VERIFICATION: 2026-01-25

**Status:** STILL VALID

**Verified By:** Claude Code P2 verification wave 5

**Current Code Analysis:**

The bug is **confirmed present** in the current codebase. The issue exists at lines 79-84 of `/home/john/elspeth-rapid/src/elspeth/engine/retry.py`:

```python
@classmethod
def from_policy(cls, policy: RetryPolicy | None) -> "RetryConfig":
    """Factory from plugin policy dict with safe defaults.

    Handles missing/malformed policy gracefully.
    This is a trust boundary - external config may have invalid values.
    """
    if policy is None:
        return cls.no_retry()

    return cls(
        max_attempts=max(1, policy.get("max_attempts", 3)),
        base_delay=max(0.01, policy.get("base_delay", 1.0)),
        max_delay=max(0.1, policy.get("max_delay", 60.0)),
        jitter=max(0.0, policy.get("jitter", 1.0)),
    )
```

The docstring claims the method "handles missing/malformed policy gracefully" and notes "This is a trust boundary - external config may have invalid values." However, the implementation does NOT handle non-numeric types.

**Test Results:**

Empirical testing confirms the bug:
- String values: `{'max_attempts': '3'}` → `TypeError: '>' not supported between instances of 'str' and 'int'`
- None values: `{'max_attempts': None}` → `TypeError: '>' not supported between instances of 'NoneType' and 'int'`
- Negative numeric values: `{'max_attempts': -5}` → Works correctly (clamped to 1)

The existing test in `tests/engine/test_retry.py::test_from_policy_handles_malformed()` only covers negative numeric values, not non-numeric types.

**Git History:**

The code has been stable since initial implementation:
- `f2f3e2b` - feat(engine): implement RetryManager with tenacity integration (original)
- `db0d187` - feat(contracts): add RetryPolicy TypedDict
- `443114a` - feat(retry): add RetryConfig.from_settings() factory
- `c786410` - ELSPETH - Release Candidate 1

No commits have addressed this type validation issue.

**Root Cause Confirmed:**

YES. The `from_policy()` method uses `max()` directly on `policy.get()` results without type validation or coercion. When plugin configuration contains non-numeric values (strings, None, etc.), the comparison in `max()` raises `TypeError`.

This violates the Three-Tier Trust Model stated in CLAUDE.md:
- Plugin configuration is a **trust boundary** (similar to external data)
- The method's docstring explicitly claims graceful handling
- The crash on malformed types contradicts both the docstring contract and the trust boundary principle

**Recommendation:**

**Keep open** - This is a valid P2 bug that should be fixed. The fix should:

1. Add type validation/coercion in `from_policy()` before calling `max()`
2. Handle string-numeric coercion (e.g., `"3"` → `3`)
3. Treat non-coercible values as missing (use defaults)
4. Add test cases for: string values, None values, non-numeric strings, mixed types
5. Consider whether to raise `PluginConfigError` with clear message vs. silent fallback to defaults

The bug has moderate impact (crashes on misconfigured plugins) but is unlikely to occur in production with validated configuration. However, it represents a gap between contract (docstring) and implementation.

---

## RESOLUTION: 2026-02-02

**Status:** FIXED

**Fixed By:** Claude Code (Opus 4.5)

**Root Cause:**

The `from_policy()` method in `RuntimeRetryConfig` used `int()` and `float()` conversions directly on policy values without type validation. When policy dict values were `None` or non-numeric strings, these conversions raised raw `TypeError` or `ValueError` with unhelpful messages.

The underlying issue was the merge operation `{**POLICY_DEFAULTS, **policy}` — explicit `None` values in policy override the numeric defaults, so `int(None)` was being called.

**Fix Applied:**

1. Added `_validate_int_field()` and `_validate_float_field()` helper functions that:
   - Reject `None` with clear error: `"Invalid retry policy: {field} must be numeric, got None"`
   - Reject non-numeric strings with clear error: `"Invalid retry policy: {field} must be numeric, got 'abc'"`
   - Reject non-numeric types (list, dict) with clear error: `"Invalid retry policy: {field} must be numeric, got list"`
   - Accept and coerce numeric strings (`"3"` → `3`, `"2.5"` → `2.5`)

2. Updated `from_policy()` to use these validators before clamping values

3. Updated docstring to clarify behavior:
   - Missing fields use defaults (unchanged)
   - Malformed fields raise `ValueError` with actionable message (explicit rejection, not silent fallback)

**Files Changed:**

- `src/elspeth/contracts/config/runtime.py` — Added validation helpers, updated `from_policy()`
- `tests/contracts/config/test_runtime_retry.py` — Added `TestFromPolicyTypeValidation` class with 8 test cases

**Test Coverage:**

New tests verify:
- `None` values raise `ValueError` with field name and "None" in message
- Non-numeric strings raise `ValueError` with field name and value in message
- List/dict values raise `ValueError` with field name and type name
- Numeric strings (`"3"`, `"2.5"`) are correctly coerced
- Multiple invalid fields report at least one clearly

**Verification:**

```bash
.venv/bin/python -m pytest tests/contracts/config/test_runtime_retry.py -v  # 23 passed
.venv/bin/python -m pytest tests/engine/test_retry_policy.py -v              # 13 passed
.venv/bin/python -m pytest tests/engine/test_retry.py -v                     # 9 passed
.venv/bin/python -m mypy src/elspeth/contracts/config/runtime.py             # clean
.venv/bin/python -m ruff check src/elspeth/contracts/config/runtime.py       # clean
```
