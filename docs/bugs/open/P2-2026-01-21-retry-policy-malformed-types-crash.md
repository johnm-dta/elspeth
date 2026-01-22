# Bug Report: RetryConfig.from_policy crashes on malformed types despite "graceful" contract

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
