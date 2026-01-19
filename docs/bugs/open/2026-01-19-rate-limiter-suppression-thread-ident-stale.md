# Bug Report: Rate limiter suppression set can retain stale thread idents (risk of suppressing unrelated AssertionErrors)

## Summary

- `RateLimiter.close()` registers pyrate-limiter leaker thread `ident`s in `_suppressed_thread_idents` so the custom `threading.excepthook` can suppress a known-benign `AssertionError`.
- The `ident` is removed only if that leaker thread later throws an `AssertionError` and hits the custom excepthook path.
- If the leaker thread exits cleanly (no exception), its `ident` remains in `_suppressed_thread_idents` indefinitely. Thread identifiers can be reused, creating a small but real risk of suppressing unrelated `AssertionError`s in future threads.

## Severity

- Severity: minor
- Priority: P3

## Reporter

- Name or handle: codex
- Date: 2026-01-19
- Related run/issue ID: N/A

## Environment

- Commit/branch: `8cfebea78be241825dd7487fed3773d89f2d7079` (local)
- OS: Linux (Ubuntu kernel 6.8.0-90-generic)
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: deep dive into subsystem 3 (core infrastructure) and create bug tickets
- Model/version: GPT-5.2 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: static inspection of `src/elspeth/core/rate_limit/limiter.py`

## Steps To Reproduce

This is easiest to observe via inspection/debugging rather than a deterministic unit test:

1. Create a `RateLimiter(...)` and call `close()` under conditions where pyrate-limiter’s leaker thread does *not* raise an `AssertionError`.
2. Inspect `elspeth.core.rate_limit.limiter._suppressed_thread_idents`.
3. Observe that the leaker `ident` can remain in the set after the thread is no longer alive.

## Expected Behavior

- Suppression registration should be strictly scoped to the specific leaker thread lifecycle and cleaned up deterministically after `close()` (even if the leaker exits without error).

## Actual Behavior

- `_suppressed_thread_idents` cleanup depends on the leaker thread throwing an `AssertionError`; clean exits can leave stale idents registered.

## Evidence

- Registration of thread idents for suppression:
  - `src/elspeth/core/rate_limit/limiter.py:263-279`
- Removal happens only when suppression triggers (one-time removal):
  - `src/elspeth/core/rate_limit/limiter.py:57-65`
- `close()` joins leakers but does not remove idents after join:
  - `src/elspeth/core/rate_limit/limiter.py:285-292`

## Impact

- User-facing impact: rare and hard-to-debug cases where unrelated thread `AssertionError`s are suppressed if a stale ident is reused.
- Data integrity / security impact: low (observability/debuggability risk).
- Performance or cost impact: negligible.

## Root Cause Hypothesis

- Suppression set is treated as “one-shot” and only cleaned when suppression fires, but the cleanup path doesn’t handle the (likely common) case where no exception occurs.

## Proposed Fix

- Code changes (modules/files):
  - `src/elspeth/core/rate_limit/limiter.py`:
    - After joining each leaker thread in `close()`, remove its ident from `_suppressed_thread_idents` unconditionally (if thread is dead).
    - Alternatively (more robust): in `_custom_excepthook`, if `thread_ident` is in the set but `args.thread` is not alive (or ident is stale), discard it and delegate to `_original_excepthook`.
- Tests to add/update:
  - Add a unit test that simulates registration + clean exit and asserts idents are removed from `_suppressed_thread_idents` after `close()`.
- Risks or migration steps: none.

## Architectural Deviations

- Spec or doc reference: N/A (observability/workaround behavior)
- Observed divergence: suppression scope can outlive the intended thread lifecycle.
- Reason (if known): suppression cleanup is currently tied only to the exception path.
- Alignment plan or decision needed: none.

## Acceptance Criteria

- After `RateLimiter.close()`, no dead thread ident remains in `_suppressed_thread_idents`.
- Suppression cannot accidentally match unrelated future threads via ident reuse.

## Tests

- Suggested tests to run: `.venv/bin/python -m pytest tests/core/rate_limit/test_limiter.py`
- New tests required: yes (stale-ident cleanup)

## Notes / Links

- Related issues/PRs: `docs/bugs/closed/2026-01-19-rate-limiter-global-excepthook-suppression.md` (previously fixed broader issue)
