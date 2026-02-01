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

1. Create a `RateLimiter(...)` and call `close()` under conditions where pyrate-limiter's leaker thread does *not* raise an `AssertionError`.
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

- Suppression set is treated as "one-shot" and only cleaned when suppression fires, but the cleanup path doesn't handle the (likely common) case where no exception occurs.

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

---

## VERIFICATION: 2026-01-25

**Status:** STILL VALID

**Verified By:** Claude Code P2 verification wave 6c (final)

**Current Code Analysis:**

Examined `/home/john/elspeth-rapid/src/elspeth/core/rate_limit/limiter.py` at current HEAD (branch: fix/rc1-bug-burndown-session-4).

The bug is **confirmed to still exist**:

1. **Thread ident registration** (lines 277-288): When `close()` is called, it collects alive leaker threads and adds their idents to `_suppressed_thread_idents`:
   ```python
   for limiter in self._limiters:
       leaker = limiter.bucket_factory._leaker
       if leaker is not None and leaker.is_alive() and leaker.ident is not None:
           leakers.append(leaker)
           with _suppressed_lock:
               _suppressed_thread_idents.add(leaker.ident)
   ```

2. **Removal only via exception path** (lines 58-61 in `_custom_excepthook`): Thread idents are removed from the suppression set ONLY when the excepthook fires:
   ```python
   if thread_ident is not None and thread_ident in _suppressed_thread_idents and args.exc_type is AssertionError:
       _suppressed_thread_idents.discard(thread_ident)
   ```

3. **No cleanup after join** (lines 299-301): After `close()` joins the leaker threads, there is NO cleanup of thread idents from the suppression set:
   ```python
   for leaker in leakers:
       leaker.join(timeout=0.05)
   # Missing: cleanup of _suppressed_thread_idents here
   ```

**Git History:**

Examined commits since 2026-01-19:
- `f819d19`: "fix(rate-limiter): make acquire() thread-safe and atomic across rate windows" - addressed different issue (thread safety of acquire, not cleanup)
- `c786410`: "ELSPETH - Release Candidate 1" - code was already in this state at RC-1
- No commits have addressed the stale thread ident cleanup issue

The related closed bug `docs/bugs/closed/2026-01-19-rate-limiter-global-excepthook-suppression.md` fixed the broader issue of using thread *names* (which could collide) and suppressing *all exception types*. The fix changed to using thread *idents* and only suppressing *AssertionError*. However, this left the cleanup issue unaddressed.

**Root Cause Confirmed:**

YES. The suppression mechanism has two paths:
1. **Happy path**: Leaker thread raises AssertionError → `_custom_excepthook` suppresses it and removes ident from set
2. **Clean exit path**: Leaker thread exits without exception → ident stays in set indefinitely

Because thread IDs can be reused by the OS (especially on systems with limited thread ID pools), a stale ident could accidentally suppress a legitimate AssertionError from a different thread that reuses the same ID.

**Test Coverage:**

Current tests in `tests/core/rate_limit/test_limiter.py` verify:
- Suppression works for registered threads with AssertionError (test_suppression_works_for_registered_assertion_error)
- Suppression is scoped to AssertionError only (test_suppression_only_for_assertion_error)
- Unregistered threads are not suppressed (test_suppression_only_for_registered_threads)

However, **no test verifies that thread idents are cleaned up after close() when threads exit cleanly**.

**Recommendation:**

**Keep open** - This is a valid P2 bug that should be fixed. While the risk is low (thread ID reuse is uncommon and would only suppress AssertionErrors), it violates the principle of deterministic cleanup and could cause hard-to-debug issues in long-running processes.

The proposed fix in the bug report is sound: after `leaker.join()` completes, unconditionally remove the leaker's ident from `_suppressed_thread_idents`. This ensures cleanup happens regardless of whether the thread raised an exception.

---

## CLOSURE: 2026-01-29

**Status:** FIXED

**Fixed By:** Commit `c5fb53e` - "fix(core): address three audit-related bugs"

**Resolution:**

The fix added cleanup after `join()` completes in `src/elspeth/core/rate_limit/limiter.py:301-308`:

```python
# Wait for leaker threads to exit
for leaker_thread, leaker_ident in leakers_with_idents:
    # Wait up to 50ms for thread to exit
    leaker_thread.join(timeout=0.05)
    # Clean up suppression registration after join completes.
    # If the thread raised AssertionError, the hook already removed it (discard is safe).
    # If the thread exited cleanly, we remove it here to prevent stale idents.
    with _suppressed_lock:
        _suppressed_thread_idents.discard(leaker_ident)
```

**Changes Made:**

1. Changed leaker collection to capture `(thread, ident)` tuples before disposal (ident may become `None` after thread exits)
2. Added unconditional `discard()` after each `join()` completes
3. Uses `discard()` (not `remove()`) since the excepthook may have already removed the ident

**Test Coverage Added:**

New test in `tests/core/rate_limit/test_limiter.py` verifies that thread idents are cleaned up from `_suppressed_thread_idents` after `close()` completes, regardless of whether the thread raised an exception.

**Verified By:** Claude Opus 4.5 (2026-01-29)
