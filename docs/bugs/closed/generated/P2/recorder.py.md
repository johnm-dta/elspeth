# Bug Report: get_calls returns raw strings instead of CallType/CallStatus enums

## Summary

- LandscapeRecorder.get_calls returns Call objects with call_type/status as raw DB strings, violating the strict enum contract.

## Severity

- Severity: minor
- Priority: P2

## Reporter

- Name or handle: Codex
- Date: 2026-01-25
- Related run/issue ID: N/A

## Environment

- Commit/branch: 86357898ee109a1dbb8d60f3dc687983fa22c1f0 (fix/rc1-bug-burndown-session-4)
- OS: unknown
- Python version: unknown
- Config profile / env vars: N/A
- Data set or fixture: Any run with external call records

## Agent Context (if relevant)

- Goal or task prompt: Static analysis bug audit of `/home/john/elspeth-rapid/src/elspeth/core/landscape/recorder.py`
- Model/version: Codex (GPT-5)
- Tooling and permissions (sandbox/approvals): read-only sandbox
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code review only

## Steps To Reproduce

1. Record a call via `record_call` using `CallType.LLM` and `CallStatus.SUCCESS`.
2. Call `LandscapeRecorder.get_calls(state_id)`.
3. Inspect `call.call_type`/`call.status` (e.g., compare to `CallType.LLM` or access `.value`).

## Expected Behavior

- `call.call_type` and `call.status` are `CallType`/`CallStatus` enums.

## Actual Behavior

- `call.call_type` and `call.status` are plain strings from the DB.

## Evidence

- `src/elspeth/core/landscape/recorder.py:1990` returns `Call(... call_type=r.call_type, status=r.status ...)` without enum coercion.
- `src/elspeth/contracts/audit.py:228` defines `Call.call_type`/`Call.status` as strict enums.

## Impact

- User-facing impact: Call filtering/comparisons may be wrong or raise `AttributeError` on `.value`.
- Data integrity / security impact: Contract violation in audit models.
- Performance or cost impact: None.

## Root Cause Hypothesis

- `get_calls` omits coercion from DB strings to `CallType`/`CallStatus`.

## Proposed Fix

- Code changes (modules/files): `src/elspeth/core/landscape/recorder.py` coerce `r.call_type` with `CallType(...)` and `r.status` with `CallStatus(...)` and let invalid values raise.
- Config or schema changes: None.
- Tests to add/update: Add a recorder test that asserts enum types and raises on invalid DB strings.
- Risks or migration steps: May surface previously hidden invalid rows.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `src/elspeth/contracts/audit.py:228`
- Observed divergence: `get_calls` returns strings instead of enums.
- Reason (if known): Unknown.
- Alignment plan or decision needed: Enforce enum coercion on retrieval.

## Acceptance Criteria

- `get_calls` returns `Call` objects with `call_type` and `status` as enums, and invalid DB values raise.

## Tests

- Suggested tests to run: `.venv/bin/python -m pytest tests/`
- New tests required: yes, unit test for enum coercion in `get_calls`.

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `CLAUDE.md`
---
# Bug Report: record_call can omit request/response payloads when payload_store is missing

## Summary

- record_call records only hashes when payload_store is None and no refs are provided, leaving external call payloads unrecorded.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: Codex
- Date: 2026-01-25
- Related run/issue ID: N/A

## Environment

- Commit/branch: 86357898ee109a1dbb8d60f3dc687983fa22c1f0 (fix/rc1-bug-burndown-session-4)
- OS: unknown
- Python version: unknown
- Config profile / env vars: N/A
- Data set or fixture: Any run with external calls and payload_store unset

## Agent Context (if relevant)

- Goal or task prompt: Static analysis bug audit of `/home/john/elspeth-rapid/src/elspeth/core/landscape/recorder.py`
- Model/version: Codex (GPT-5)
- Tooling and permissions (sandbox/approvals): read-only sandbox
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code review only

## Steps To Reproduce

1. Instantiate `LandscapeRecorder(db, payload_store=None)`.
2. Call `record_call(..., request_ref=None, response_ref=None)` with a request and response.
3. Inspect the stored call row: `request_ref`/`response_ref` are NULL and payloads are unrecoverable.

## Expected Behavior

- External calls persist full request/response payloads (or fail fast if payload storage is unavailable).

## Actual Behavior

- Calls are recorded with hashes only; payload references can remain NULL.

## Evidence

- `src/elspeth/core/landscape/recorder.py:2067` only stores payloads when `self._payload_store` is not None.
- `src/elspeth/core/landscape/recorder.py:2088` persists `request_ref`/`response_ref` even if they remain None.
- `CLAUDE.md:23` mandates full request/response recording for external calls.

## Impact

- User-facing impact: Replay/verify cannot reconstruct requests or responses.
- Data integrity / security impact: Audit trail violates non-negotiable storage requirements.
- Performance or cost impact: None.

## Root Cause Hypothesis

- record_call lacks enforcement that payloads must be persisted when recording external calls.

## Proposed Fix

- Code changes (modules/files): `src/elspeth/core/landscape/recorder.py` should raise if `payload_store` is None and `request_ref` is None, and if `response_data` is provided but `response_ref` is None.
- Config or schema changes: None (unless choosing to add inline payload storage columns instead of enforcing payload_store).
- Tests to add/update: Add unit test asserting record_call fails without payload storage or refs.
- Risks or migration steps: Enforcing this may require configuring payload_store in environments that currently omit it.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `CLAUDE.md:23`
- Observed divergence: External calls can be recorded without payload persistence.
- Reason (if known): Optional payload_store and no validation.
- Alignment plan or decision needed: Require payload_store or request/response refs for all calls.

## Acceptance Criteria

- record_call always persists request/response payload refs (unless no response_data) or raises if storage is unavailable.

## Tests

- Suggested tests to run: `.venv/bin/python -m pytest tests/`
- New tests required: yes, coverage for payload enforcement in record_call.

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `CLAUDE.md`
---
# Bug Report: Token outcome retrieval masks invalid is_terminal values

## Summary

- get_token_outcome and get_token_outcomes_for_row silently coerce invalid is_terminal values to False, masking audit data corruption.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: Codex
- Date: 2026-01-25
- Related run/issue ID: N/A

## Environment

- Commit/branch: 86357898ee109a1dbb8d60f3dc687983fa22c1f0 (fix/rc1-bug-burndown-session-4)
- OS: unknown
- Python version: unknown
- Config profile / env vars: N/A
- Data set or fixture: Any run where token_outcomes.is_terminal is corrupted

## Agent Context (if relevant)

- Goal or task prompt: Static analysis bug audit of `/home/john/elspeth-rapid/src/elspeth/core/landscape/recorder.py`
- Model/version: Codex (GPT-5)
- Tooling and permissions (sandbox/approvals): read-only sandbox
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code review only

## Steps To Reproduce

1. Insert a token_outcomes row with `is_terminal = 2` (or NULL).
2. Call `get_token_outcome(token_id)` or `get_token_outcomes_for_row(run_id, row_id)`.
3. Observe `is_terminal` returned as False instead of an error.

## Expected Behavior

- Invalid `is_terminal` values trigger a crash/ValueError per Tier 1 trust rules.

## Actual Behavior

- Non-1 values are silently treated as False.

## Evidence

- `src/elspeth/core/landscape/recorder.py:2298` returns `is_terminal=result.is_terminal == 1` without validation.
- `src/elspeth/core/landscape/recorder.py:2360` repeats the same coercion in the row-level query.
- `CLAUDE.md:38` requires crashes on audit DB anomalies.

## Impact

- User-facing impact: Lineage can misreport terminal vs non-terminal outcomes.
- Data integrity / security impact: Corrupted audit data can be silently accepted.
- Performance or cost impact: None.

## Root Cause Hypothesis

- Missing validation for `is_terminal` values when constructing TokenOutcome.

## Proposed Fix

- Code changes (modules/files): `src/elspeth/core/landscape/recorder.py` should validate `is_terminal` is exactly 0 or 1 and raise otherwise before boolean conversion.
- Config or schema changes: None.
- Tests to add/update: Add tests that inject invalid `is_terminal` values and assert a ValueError.
- Risks or migration steps: May surface existing bad rows.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `CLAUDE.md:38`
- Observed divergence: Invalid Tier 1 data is coerced instead of crashing.
- Reason (if known): Missing validation logic.
- Alignment plan or decision needed: Enforce strict validation for audit DB fields.

## Acceptance Criteria

- Invalid `is_terminal` values raise errors; valid 0/1 values map correctly.

## Tests

- Suggested tests to run: `.venv/bin/python -m pytest tests/`
- New tests required: yes, tests for `is_terminal` validation.

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `CLAUDE.md`
