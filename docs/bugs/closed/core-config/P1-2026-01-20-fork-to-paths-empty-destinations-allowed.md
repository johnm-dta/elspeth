# Bug Report: RoutingAction.fork_to_paths allows empty destinations (can silently drop tokens)

## Summary

- `RoutingAction.fork_to_paths(paths)` accepts an empty list and produces an action with `destinations=()`.
- `GateExecutor` treats `RoutingKind.FORK_TO_PATHS` as a fork regardless of destination count, but:
  - `_record_routing()` records **no routing_events** when there are zero destinations
  - `TokenManager.fork_token()` / `LandscapeRecorder.fork_token()` create **no child tokens** when branches is empty
  - `RowProcessor` returns `RowOutcome.FORKED` for the parent token even when there are no children
- Net result: a token can be marked “forked” in runtime logic but have no continuation and no corresponding audit routing events, violating the “no silent drops” invariant.

## Severity

- Severity: critical
- Priority: P1

## Reporter

- Name or handle: codex
- Date: 2026-01-20
- Related run/issue ID: N/A

## Environment

- Commit/branch: `8cfebea78be241825dd7487fed3773d89f2d7079` (main)
- OS: Linux (Ubuntu kernel 6.8.0-90-generic)
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Steps To Reproduce

1. Create a `RoutingAction` with no fork destinations:

```python
from elspeth.contracts.routing import RoutingAction

action = RoutingAction.fork_to_paths([])
assert action.destinations == ()
```

2. Have a gate return that action (buggy gate implementation).
3. Observe:
   - No routing events recorded for the gate state
   - No child tokens created
   - Parent token yields `RowOutcome.FORKED` in the row processor

## Expected Behavior

- `fork_to_paths` requires at least one destination. Empty forks should crash immediately (system-owned plugin bug) to prevent audit integrity violations.

## Actual Behavior

- Empty forks are allowed and can result in a token effectively disappearing without a recorded terminal path or routing events.

## Evidence

- Factory method accepts empty lists:
  - `src/elspeth/contracts/routing.py:85-97` (`RoutingAction.fork_to_paths()` sets `destinations=tuple(paths)` with no validation)
- Gate executor treats FORK_TO_PATHS and passes `branches=list(action.destinations)`:
  - `src/elspeth/engine/executors.py:412-430`
- Recorder fork implementation inserts children in a loop over `branches` without validating non-empty:
  - `src/elspeth/core/landscape/recorder.py:785-847`
- RowProcessor returns FORKED regardless of child count:
  - `src/elspeth/engine/processor.py:573-593`
- Multi-route recording silently does nothing for empty `routes`:
  - `src/elspeth/core/landscape/recorder.py:1196-1249` (`record_routing_events()` returns [] when routes is empty)

## Impact

- User-facing impact: data can be silently dropped in fork scenarios if a gate misbehaves.
- Data integrity / security impact: high. Missing routing events and missing child tokens break lineage/audit invariants.
- Performance or cost impact: N/A

## Root Cause Hypothesis

- `fork_to_paths` is missing a simple validation guard, and downstream code paths assume non-empty destinations without asserting it.

## Proposed Fix

- Code changes (modules/files):
  - `src/elspeth/contracts/routing.py`: in `RoutingAction.fork_to_paths()`, raise `ValueError` if `paths` is empty.
  - Defense-in-depth:
    - `src/elspeth/engine/executors.py`: assert `action.destinations` is non-empty when `action.kind == FORK_TO_PATHS`.
    - `src/elspeth/core/landscape/recorder.py`: raise if `branches` is empty in `fork_token()`.
- Tests to add/update:
  - Unit test for `RoutingAction.fork_to_paths([])` raising.
  - Integration test ensuring fork produces at least one routing_event and at least one child token.

## Architectural Deviations

- Spec or doc reference: CLAUDE.md “No silent drops / every token terminal”
- Observed divergence: empty fork leads to no routing events and no child tokens
- Alignment plan or decision needed: none; this should fail fast

## Acceptance Criteria

- `RoutingAction.fork_to_paths([])` raises `ValueError`.
- Runtime paths cannot record a fork outcome without at least one routing_event and at least one child token.

## Tests

- Suggested tests to run: `.venv/bin/python -m pytest tests/engine -k fork`
- New tests required: yes

## Resolution

**Status:** FIXED (2026-01-21)
**Fixed by:** Claude Opus 4.5

**Investigation findings:**
- Primary validation (`RoutingAction.fork_to_paths([])` raising `ValueError`) already existed at lines 112-113 of routing.py
- Bug report was partially outdated - validation was already in place
- Missing: test coverage for existing validation + defense-in-depth at recorder layer

**Changes made:**
1. Added test `test_fork_to_paths_rejects_empty_list` - verifies existing validation
2. Added test `test_fork_to_paths_rejects_duplicate_paths` - verifies existing duplicate check
3. Added defense-in-depth validation in `LandscapeRecorder.fork_token()` - raises `ValueError` if branches is empty
4. Added test `test_fork_token_rejects_empty_branches` - verifies defense-in-depth

**Files modified:**
- `tests/contracts/test_routing.py` - Added 2 tests for RoutingAction validation
- `tests/core/landscape/test_recorder.py` - Added 1 test for defense-in-depth
- `src/elspeth/core/landscape/recorder.py` - Added validation at `fork_token()`

**Verification:**
- All 7 fork-related tests pass
- mypy strict: no issues
- ruff: all checks passed

**Defense-in-depth achieved:**
- Layer 1: `RoutingAction.fork_to_paths()` validates at contract level
- Layer 2: `LandscapeRecorder.fork_token()` validates at recorder level
