# Bug Report: `_row_to_node_state()` does not enforce `completed_at` for terminal node states (integrity check gap)

## Summary

- `_row_to_node_state()` is responsible for enforcing “Tier 1” invariants when reading `node_states` from the audit DB.
- For `COMPLETED` and `FAILED` states, it validates `output_hash` and/or `duration_ms`, but does not validate `completed_at` is non-NULL.
- This allows audit DB corruption (or partial writes) to produce terminal `NodeStateCompleted/Failed` objects with `completed_at=None`, silently weakening the audit record and violating the implied invariants in docstrings.

## Severity

- Severity: minor
- Priority: P3

## Reporter

- Name or handle: codex
- Date: 2026-01-19
- Related run/issue ID: N/A

## Environment

- Commit/branch: `main` @ `8ca061c9293db459c9a900f2f74b19b59a364a42`
- OS: Linux (Ubuntu kernel 6.8.0-90-generic)
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: deep dive subsystem 4 (Landscape) and create bug tickets
- Model/version: GPT-5.2 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code inspection

## Steps To Reproduce

1. Create a `node_states` row with `status='completed'` or `status='failed'` but `completed_at=NULL` (e.g., via a manual DB edit, partial transaction, or corruption).
2. Call `LandscapeRecorder.get_node_state(state_id)` / `get_node_states_for_token(token_id)`.
3. Observe no explicit crash for missing `completed_at` (it is passed through as `None`).

## Expected Behavior

- For terminal states:
  - `completed_at` must be non-NULL; otherwise raise `ValueError` as an audit integrity violation.

## Actual Behavior

- `completed_at` is not validated for `COMPLETED` or `FAILED` rows.

## Evidence

- `_row_to_node_state()` missing `completed_at` checks:
  - `src/elspeth/core/landscape/recorder.py:131-179` (`completed_at=row.completed_at` without NULL validation)
- Docstrings state terminal invariants:
  - `src/elspeth/contracts/audit.py` node state contracts and invariants

## Impact

- User-facing impact: explain/export may show incomplete timing info without crashing, which can mislead investigations.
- Data integrity / security impact: moderate. Tier 1 DB corruption should crash immediately.
- Performance or cost impact: N/A

## Root Cause Hypothesis

- Integrity checks were added incrementally (output_hash/duration_ms) and `completed_at` was overlooked.

## Proposed Fix

- Code changes (modules/files):
  - `src/elspeth/core/landscape/recorder.py`:
    - Add `if row.completed_at is None: raise ValueError(...)` for `COMPLETED` and `FAILED`.
- Config or schema changes: none.
- Tests to add/update:
  - Add tests that simulate missing `completed_at` and assert the read crashes (Tier 1 policy).
- Risks or migration steps:
  - None (this tightens validation).

## Architectural Deviations

- Spec or doc reference: `CLAUDE.md` (Tier 1 “crash immediately” policy)
- Observed divergence: missing validation allows incomplete terminal state records to pass.
- Reason (if known): missing check.
- Alignment plan or decision needed: none.

## Acceptance Criteria

- Terminal `node_states` rows with `completed_at=NULL` crash on read with a clear integrity violation error.

## Tests

- Suggested tests to run: `pytest tests/core/landscape/test_recorder.py`
- New tests required: yes (integrity validation for completed_at)

## Notes / Links

- Related issues/PRs: N/A
