# Bug Report: Rate limiter globally overrides `threading.excepthook` and suppresses thread exceptions by name

## Summary

- Importing `elspeth.core.rate_limit.limiter` replaces `threading.excepthook` process-wide. The custom hook suppresses exceptions based on thread *name* (not identity) and suppresses *all exception types* for matching names, which can hide real failures and is a surprising global side effect for embedders.

## Severity

- Severity: major
- Priority: P2

## Reporter

- Name or handle: codex
- Date: 2026-01-19
- Related run/issue ID: N/A

## Environment

- Commit/branch: `main` @ `25468ac9550b481a55b81a05d84bbf2592e6430c`
- OS: Linux (Ubuntu 6.8.0-90-generic)
- Python version: 3.13.1
- Config profile / env vars: N/A (static analysis)
- Data set or fixture: N/A (static analysis)

## Agent Context (if relevant)

- Goal or task prompt: deep dive into subsystems, identify hotspots, write bug reports
- Model/version: GPT-5.2 (Codex CLI)
- Tooling and permissions (sandbox/approvals): sandbox read-only, network restricted
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: inspected rate limiter wrapper + cleanup logic around pyrate-limiter leaker thread

## Steps To Reproduce

1. Import `elspeth.core.rate_limit.limiter`.
2. Observe `threading.excepthook` has been replaced for the entire process.
3. Create/trigger a scenario where a thread name is added to `_suppressed_threads` (e.g., by calling `RateLimiter.close()` which tracks pyrate-limiter leaker thread names).
4. Raise an exception from a thread with a suppressed name and observe it is silently suppressed (no delegation to the original hook).

## Expected Behavior

- Importing ELSPETH should not permanently change process-wide exception handling.
- If a workaround is needed, suppression should be narrowly scoped: specific threads by identity, and specific known-benign exception types/context, with observability.

## Actual Behavior

- A custom `threading.excepthook` is installed at import time and uses thread names to decide suppression, without checking exception type/context.

## Evidence

- Captures original hook and installs custom hook at import time: `src/elspeth/core/rate_limit/limiter.py:26-56`
- Suppression keyed by thread name and suppresses all exception types for matching names: `src/elspeth/core/rate_limit/limiter.py:35-52`
- Uses pyrate-limiter private attribute `_leaker` to locate cleanup threads and then adds names to suppression map: `src/elspeth/core/rate_limit/limiter.py:241-270`

## Impact

- User-facing impact: thread exceptions may be silently hidden, complicating debugging and test reliability.
- Data integrity / security impact: suppressed exceptions can hide real errors in shared processes.
- Performance or cost impact: potential hidden retries/loops if failures are masked.

## Root Cause Hypothesis

- A process-global hook was chosen to work around an upstream cleanup race, and thread names were used as a proxy for identifying “safe to suppress” threads.

## Proposed Fix

- Code changes (modules/files):
  - Option A (recommended): narrow suppression to the specific benign failure mode:
    - identify threads by identity (e.g., `thread.ident`) rather than name
    - suppress only the known benign exception type(s) and (ideally) only when stack/context matches pyrate-limiter cleanup
    - emit a debug log (or counter) when suppression occurs
  - Option B: install/uninstall the custom hook dynamically (install on first `RateLimiter` creation; restore original on last close).
  - Option C: avoid overriding `threading.excepthook` (upstream fix, vendor workaround, or configurable opt-out).
- Config or schema changes:
  - Consider an env var/setting to disable suppression in embedded contexts.
- Tests to add/update:
  - Add a unit test ensuring importing the module does not permanently override `threading.excepthook` (or that it restores on shutdown).
  - Add a test ensuring only the intended exception type is suppressed.
- Risks or migration steps:
  - Tight coupling to pyrate-limiter internals (`_leaker`) may break on upstream changes; consider feature detection/fallback.

## Architectural Deviations

- Spec or doc reference: N/A (general library embedding expectations)
- Observed divergence: a leaf utility module introduces process-global side effects on import.
- Reason (if known): workaround for pyrate-limiter leaker cleanup noise in tests.
- Alignment plan or decision needed: decide whether ELSPETH prioritizes “quiet tests” over safe embeddability; document the decision either way.

## Acceptance Criteria

- Importing rate-limiting code does not permanently change process-wide exception handling, or the change is narrowly scoped and reversible.
- Only the intended pyrate-limiter cleanup race is suppressed; other exceptions remain visible.
- Suppression cannot accidentally match unrelated threads by name.

## Tests

- Suggested tests to run: `pytest tests/`
- New tests required: yes (excepthook side effects + suppression narrowing)

## Notes / Links

- Upstream dependency: `pyrate-limiter` (workaround targets its leaker thread cleanup race).
