# Bug Report: ROUTE accepts string mode, allowing COPY semantics without fork

## RESOLUTION (2026-01-27)

**Status: NOT A BUG (Invalid Report)**

This bug report was based on incorrect analysis. `RoutingMode` is a **StrEnum** (`class RoutingMode(str, Enum)`), which means string comparison works correctly.

### Evidence:
1. `"copy" == RoutingMode.COPY` evaluates to `True` due to StrEnum inheritance
2. `RoutingAction.route('review', mode="copy")` correctly raises ValueError
3. 24 routing contract tests pass including `test_route_with_copy_raises`

### Verification:
```python
>>> from elspeth.contracts.routing import RoutingAction
>>> RoutingAction.route('review', mode='copy')
# Raises: ValueError: COPY mode not supported for ROUTE kind...
```

The static analysis tool that generated this report didn't understand Python's StrEnum behavior.

---

## Summary (Original Report - INVALID)

- `RoutingAction.__post_init__` only blocks `RoutingMode.COPY` enums, so `mode="copy"` bypasses the ROUTE prohibition and gets coerced later, recording a COPY route even though no fork occurs.

## Severity

- Severity: ~~major~~ **INVALID**
- Priority: ~~P1~~ **CLOSED**

## Reporter

- Name or handle: Codex
- Date: 2026-01-25
- Related run/issue ID: N/A

## Environment

- Commit/branch: Unknown
- OS: Unknown
- Python version: Unknown
- Config profile / env vars: Unknown
- Data set or fixture: Unknown

## Agent Context (if relevant)

- Goal or task prompt: static analysis deep bug audit of /home/john/elspeth-rapid/src/elspeth/contracts/routing.py
- Model/version: Codex (GPT-5)
- Tooling and permissions (sandbox/approvals): read-only sandbox
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code review only

## Steps To Reproduce

1. Implement a gate that returns `RoutingAction.route("review", mode="copy")` (string mode).
2. Run a pipeline through that gate.
3. Inspect routing events recorded for that token.

## Expected Behavior

- `RoutingAction.route(..., mode="copy")` should raise immediately (invalid type and COPY not allowed for ROUTE), preventing any COPY routing record without an actual fork.

## Actual Behavior

- The action is accepted, and `record_routing_event` coerces `"copy"` to `RoutingMode.COPY`, so the audit trail records COPY routing even though the token does not fork/continue.

## Evidence

- `src/elspeth/contracts/routing.py:59` and `src/elspeth/contracts/routing.py:70` only compare `mode` to `RoutingMode.COPY`, so `mode="copy"` bypasses the guard.
- `src/elspeth/contracts/routing.py:43`–`src/elspeth/contracts/routing.py:45` explicitly prohibit COPY with ROUTE for audit integrity.
- `src/elspeth/core/landscape/recorder.py:1202`–`src/elspeth/core/landscape/recorder.py:1233` coerces string modes to enums, causing a COPY record to be stored.

## Impact

- User-facing impact: `explain()`/lineage can indicate a token continued and was copied when it actually terminated at a sink.
- Data integrity / security impact: audit trail inconsistency; implies dual terminal states without corresponding token flow.
- Performance or cost impact: None.

## Root Cause Hypothesis

- `RoutingAction.__post_init__` does not enforce `mode` type (`RoutingMode`), so string values bypass COPY checks and are later coerced in the recorder.

## Proposed Fix

- Code changes (modules/files):
  - Add strict type validation in `src/elspeth/contracts/routing.py` to require `mode` is `RoutingMode` (raise on strings).
  - Optionally normalize via `RoutingMode(mode)` if you decide to accept strings, then re-check COPY prohibition for ROUTE.
- Config or schema changes: Unknown
- Tests to add/update:
  - Unit test that `RoutingAction.route("x", mode="copy")` raises.
  - Unit test that any non-`RoutingMode` value for `mode` raises.
- Risks or migration steps:
  - Tightens contract; any code passing string modes must be fixed to pass enums explicitly.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `CLAUDE.md:36`–`CLAUDE.md:44`
- Observed divergence: internal routing contracts allow wrong types to pass and get coerced later, violating “wrong type = crash” for trusted audit data.
- Reason (if known): missing type enforcement in `RoutingAction.__post_init__`.
- Alignment plan or decision needed: enforce `RoutingMode` type strictly in routing contracts.

## Acceptance Criteria

- `RoutingAction.route(..., mode="copy")` raises a clear error.
- ROUTE actions can never record COPY mode.
- All routing actions reject non-`RoutingMode` values at construction.

## Tests

- Suggested tests to run: `./.venv/bin/python -m pytest tests/contracts/test_routing.py`
- New tests required: yes, add coverage for invalid `mode` inputs and COPY-on-ROUTE rejection.

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `CLAUDE.md:36`–`CLAUDE.md:44`
