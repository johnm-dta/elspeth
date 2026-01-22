# Bug Report: get_calls returns raw strings for call enums

## Summary

- `LandscapeRecorder.get_calls()` builds `Call` objects with `call_type` and `status` taken directly from DB strings, violating the enum-only contract and allowing invalid DB values to propagate silently.

## Severity

- Severity: major
- Priority: P2

## Reporter

- Name or handle: Codex
- Date: Unknown
- Related run/issue ID: Unknown

## Environment

- Commit/branch: Unknown
- OS: Unknown
- Python version: Unknown
- Config profile / env vars: Unknown
- Data set or fixture: Unknown

## Agent Context (if relevant)

- Goal or task prompt: Static analysis bug audit for `src/elspeth/core/landscape/recorder.py`
- Model/version: GPT-5 (Codex)
- Tooling and permissions (sandbox/approvals): Read-only sandbox; no approvals; used `rg`/`sed`
- Determinism details (seed, run ID): Unknown
- Notable tool calls or steps: `rg`, `sed`, `nl -ba` for line numbers

## Steps To Reproduce

1. Record an external call via `LandscapeRecorder.record_call()` or run a pipeline that records at least one call.
2. Invoke `LandscapeRecorder.get_calls(state_id)` for the recorded state.
3. Observe that `call.call_type` and `call.status` are `str` rather than `CallType`/`CallStatus` (e.g., `call.call_type.value` raises).

## Expected Behavior

- `get_calls()` returns `Call` objects with `call_type` and `status` coerced to `CallType`/`CallStatus`, and invalid DB values raise immediately.

## Actual Behavior

- `get_calls()` returns raw DB strings for `call_type` and `status`, bypassing enum coercion and strict contract checks.

## Evidence

- Logs or stack traces: Code reference in `src/elspeth/core/landscape/recorder.py:1944`, `src/elspeth/core/landscape/recorder.py:1948`, `src/elspeth/core/landscape/recorder.py:1949`; contract requires enums in `src/elspeth/contracts/audit.py:201`, `src/elspeth/contracts/audit.py:211`, `src/elspeth/contracts/audit.py:212`
- Artifacts (paths, IDs, screenshots): Unknown
- Minimal repro input (attach or link): Unknown

## Impact

- User-facing impact: Downstream code comparing to `CallType`/`CallStatus` may misbehave or raise unexpectedly.
- Data integrity / security impact: Violates Tier-1 “strict enum” contract; invalid DB values would not be detected.
- Performance or cost impact: Unknown

## Root Cause Hypothesis

- `get_calls()` fails to coerce DB string fields to enums, unlike other recorder getters and the CallRepository.

## Proposed Fix

- Code changes (modules/files): Convert `call_type` and `status` via `CallType(r.call_type)` and `CallStatus(r.status)` in `src/elspeth/core/landscape/recorder.py`
- Config or schema changes: None
- Tests to add/update: Add unit test ensuring `get_calls()` returns enums and rejects invalid values
- Risks or migration steps: Low risk; aligns with existing strict contract behavior

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `src/elspeth/contracts/audit.py:205` (strict enum contract)
- Observed divergence: `get_calls()` returns strings instead of enums
- Reason (if known): Inconsistent enum coercion in recorder read path
- Alignment plan or decision needed: Align `get_calls()` with contract and repository conversion

## Acceptance Criteria

- `get_calls()` returns `CallType`/`CallStatus` enums for all records.
- Invalid `call_type`/`status` values in DB raise `ValueError` on read.
- Test coverage exists for enum coercion in `get_calls()`.

## Tests

- Suggested tests to run: `pytest tests/core/landscape/test_recorder_calls.py::test_get_calls_coerces_enums`
- New tests required: Yes

## Notes / Links

- Related issues/PRs: Unknown
- Related design docs: Unknown
