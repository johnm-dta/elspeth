# Bug Report: RateLimiter.acquire() not locked/atomic across multi-rate limiters

## Summary

- `RateLimiter.acquire()` iterates limiters without `self._lock`, so concurrent calls or later limiter failures can occur after earlier token consumption, leading to inconsistent multi-window rate limiting and lost capacity.

## Severity

- Severity: major
- Priority: P2

## Reporter

- Name or handle: codex
- Date: 2026-01-22
- Related run/issue ID: Unknown

## Environment

- Commit/branch: fix/rc1-bug-burndown-session-2 @ 81a0925d7d6de0d0e16fdd2d535f63d096a7d052
- OS: Linux 6.8.0-90-generic x86_64
- Python version: 3.13.1
- Config profile / env vars: Unknown
- Data set or fixture: Unknown

## Agent Context (if relevant)

- Goal or task prompt: Static analysis bug audit of `src/elspeth/core/rate_limit/limiter.py`
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): read-only sandbox, approvals never, network restricted
- Determinism details (seed, run ID): Unknown
- Notable tool calls or steps: inspected rate limiter wrapper and pyrate-limiter internals for locking semantics

## Steps To Reproduce

1. Create a `RateLimiter` with both `requests_per_second` and `requests_per_minute`.
2. Monkeypatch the minute limiter’s `try_acquire` to raise `BucketFullException` after the per-second limiter succeeds (or call `acquire()` concurrently from two threads).
3. Call `RateLimiter.acquire()` and observe that the first limiter consumed tokens despite overall acquisition failing.

## Expected Behavior

- `acquire()` is serialized per instance and does not partially consume tokens across multi-rate windows when the overall acquisition cannot complete.

## Actual Behavior

- `acquire()` loops over limiters without `self._lock`; partial consumption and interleaving are possible under concurrency or when a later limiter fails.

## Evidence

- Logs or stack traces: Unknown; code references `src/elspeth/core/rate_limit/limiter.py:191`, `src/elspeth/core/rate_limit/limiter.py:229`, `src/elspeth/core/rate_limit/limiter.py:147`
- Artifacts (paths, IDs, screenshots): Unknown
- Minimal repro input (attach or link): Unknown

## Impact

- User-facing impact: inconsistent rate limiting under concurrency; tokens can be “lost,” increasing throttling
- Data integrity / security impact: low
- Performance or cost impact: elevated wait times and reduced throughput under load

## Root Cause Hypothesis

- `acquire()` does not use `self._lock` or any atomic multi-window acquisition strategy, unlike `try_acquire()`.

## Proposed Fix

- Code changes (modules/files): add `with self._lock:` around `acquire()` in `src/elspeth/core/rate_limit/limiter.py`, and consider mirroring `try_acquire()`’s all-or-nothing semantics for the blocking path
- Config or schema changes: Unknown
- Tests to add/update: add a unit test that forces failure in the second limiter and asserts no partial consumption; add a concurrency test for `acquire()`
- Risks or migration steps: minimal; behavior becomes more deterministic under concurrency

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): Unknown
- Observed divergence: `acquire()` lacks the atomicity guarantees implied by `try_acquire()`
- Reason (if known): Unknown
- Alignment plan or decision needed: define whether `acquire()` must guarantee per-instance atomicity across multi-rate windows

## Acceptance Criteria

- `acquire()` is thread-safe per instance and does not leave partial token consumption when a later limiter fails.

## Tests

- Suggested tests to run: `.venv/bin/python -m pytest tests/core/rate_limit/test_limiter.py`
- New tests required: yes, for `acquire()` atomicity and concurrency

## Notes / Links

- Related issues/PRs: `docs/bugs/open/P2-2026-01-19-rate-limiter-acquire-not-thread-safe-or-atomic.md`
- Related design docs: Unknown
---
# Bug Report: Rate limiter suppression set retains stale thread idents

## Summary

- `RateLimiter.close()` registers leaker thread idents for exception suppression but only removes them when suppression fires; clean exits leave stale idents that can suppress unrelated `AssertionError`s if thread IDs are reused.

## Severity

- Severity: minor
- Priority: P3

## Reporter

- Name or handle: codex
- Date: 2026-01-22
- Related run/issue ID: Unknown

## Environment

- Commit/branch: fix/rc1-bug-burndown-session-2 @ 81a0925d7d6de0d0e16fdd2d535f63d096a7d052
- OS: Linux 6.8.0-90-generic x86_64
- Python version: 3.13.1
- Config profile / env vars: Unknown
- Data set or fixture: Unknown

## Agent Context (if relevant)

- Goal or task prompt: Static analysis bug audit of `src/elspeth/core/rate_limit/limiter.py`
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): read-only sandbox, approvals never, network restricted
- Determinism details (seed, run ID): Unknown
- Notable tool calls or steps: inspected rate limiter cleanup/excepthook logic

## Steps To Reproduce

1. Create a `RateLimiter` and call `close()` under conditions where the leaker thread exits cleanly.
2. Inspect `elspeth.core.rate_limit.limiter._suppressed_thread_idents` after close.
3. Observe the leaker thread ident remains in the set despite the thread no longer being alive.

## Expected Behavior

- Suppression should be scoped to the leaker thread lifetime and cleaned up after `close()` regardless of whether an exception occurred.

## Actual Behavior

- `_suppressed_thread_idents` is only cleaned when suppression triggers; clean exits leave stale idents registered.

## Evidence

- Logs or stack traces: Unknown; code references `src/elspeth/core/rate_limit/limiter.py:32`, `src/elspeth/core/rate_limit/limiter.py:57`, `src/elspeth/core/rate_limit/limiter.py:256`, `src/elspeth/core/rate_limit/limiter.py:277`
- Artifacts (paths, IDs, screenshots): Unknown
- Minimal repro input (attach or link): Unknown

## Impact

- User-facing impact: rare suppression of unrelated `AssertionError`s, making real failures harder to detect
- Data integrity / security impact: low
- Performance or cost impact: negligible

## Root Cause Hypothesis

- Suppression cleanup is tied only to the exception path; `close()` does not remove idents when threads exit cleanly.

## Proposed Fix

- Code changes (modules/files): in `src/elspeth/core/rate_limit/limiter.py`, discard each leaker ident after `join()` when the thread is no longer alive, or track thread objects in a `WeakSet` and check `args.thread` directly
- Config or schema changes: Unknown
- Tests to add/update: add a unit test that calls `close()` and asserts no dead thread ident remains in `_suppressed_thread_idents`
- Risks or migration steps: minimal; only affects cleanup bookkeeping

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): Unknown
- Observed divergence: suppression scope can outlive the intended leaker thread lifecycle
- Reason (if known): Unknown
- Alignment plan or decision needed: Unknown

## Acceptance Criteria

- After `close()`, no dead leaker thread ident remains in `_suppressed_thread_idents`.

## Tests

- Suggested tests to run: `.venv/bin/python -m pytest tests/core/rate_limit/test_limiter.py`
- New tests required: yes, stale-ident cleanup

## Notes / Links

- Related issues/PRs: `docs/bugs/open/P2-2026-01-19-rate-limiter-suppression-thread-ident-stale.md`
- Related design docs: Unknown
---
# Bug Report: try_acquire uses stale bucket counts and over-throttles

## Summary

- `_would_all_buckets_accept()` relies on `bucket.count()` (total items), which ignores the active rate window and depends on the leaker’s 10s cleanup cadence; `try_acquire()` can therefore return False even after the rate window has cleared, throttling below configured limits.

## Severity

- Severity: major
- Priority: P2

## Reporter

- Name or handle: codex
- Date: 2026-01-22
- Related run/issue ID: Unknown

## Environment

- Commit/branch: fix/rc1-bug-burndown-session-2 @ 81a0925d7d6de0d0e16fdd2d535f63d096a7d052
- OS: Linux 6.8.0-90-generic x86_64
- Python version: 3.13.1
- Config profile / env vars: Unknown
- Data set or fixture: Unknown

## Agent Context (if relevant)

- Goal or task prompt: Static analysis bug audit of `src/elspeth/core/rate_limit/limiter.py`
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): read-only sandbox, approvals never, network restricted
- Determinism details (seed, run ID): Unknown
- Notable tool calls or steps: inspected rate limiter wrapper and pyrate-limiter bucket/leaker implementations

## Steps To Reproduce

1. Create `RateLimiter("svc", requests_per_second=10)` with no persistence (in-memory bucket).
2. Call `try_acquire()` 10 times quickly (all True).
3. Wait ~1–2 seconds (less than the leaker’s 10s interval) and call `try_acquire()` again.
4. Observe `try_acquire()` returns False until the leaker runs, despite the 1-second rate window having expired.

## Expected Behavior

- `try_acquire()` returns True once the configured rate window has cleared, matching the effective behavior of the underlying bucket.

## Actual Behavior

- `try_acquire()` returns False based on stale total counts until the leaker cleans the bucket, effectively enforcing a much longer window.

## Evidence

- Logs or stack traces: Unknown; code references `src/elspeth/core/rate_limit/limiter.py:201`, `src/elspeth/core/rate_limit/limiter.py:155`, `/home/john/elspeth-rapid/.venv/lib/python3.13/site-packages/pyrate_limiter/buckets/in_memory_bucket.py:85`, `/home/john/elspeth-rapid/.venv/lib/python3.13/site-packages/pyrate_limiter/abstracts/bucket.py:118`
- Artifacts (paths, IDs, screenshots): Unknown
- Minimal repro input (attach or link): Unknown

## Impact

- User-facing impact: non-blocking checks deny calls that should be allowed, reducing throughput
- Data integrity / security impact: low
- Performance or cost impact: significant underutilization of allowed rate (e.g., per-second behaves like per-10s)

## Root Cause Hypothesis

- `_would_all_buckets_accept()` uses `bucket.count()` without first purging expired items or calculating window-scoped counts, and the leaker’s default 10s interval leaves expired items in the bucket for long periods.

## Proposed Fix

- Code changes (modules/files): in `src/elspeth/core/rate_limit/limiter.py`, compute window-accurate counts by calling `bucket.leak(now_ms)` before `count()` (using the same clock units as pyrate-limiter), or replace the peek logic with a time-windowed count derived from bucket internals; optionally set leaker interval to <= min rate interval when creating buckets
- Config or schema changes: Unknown
- Tests to add/update: add a unit test that verifies `try_acquire()` returns True after >1s but <10s for a per-second limiter
- Risks or migration steps: calling `leak()` on each `try_acquire()` may add overhead; assess performance for SQLite buckets

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): Unknown
- Observed divergence: `try_acquire()` does not reflect the intended rate window semantics under default leaker cadence
- Reason (if known): Unknown
- Alignment plan or decision needed: decide whether `try_acquire()` must be strictly accurate or intentionally conservative; current docstring implies accuracy

## Acceptance Criteria

- `try_acquire()` returns True once the configured rate window has elapsed, without waiting for leaker cleanup.

## Tests

- Suggested tests to run: `.venv/bin/python -m pytest tests/core/rate_limit/test_limiter.py`
- New tests required: yes, `try_acquire()` window-accuracy test

## Notes / Links

- Related issues/PRs: Unknown
- Related design docs: Unknown
