# Bug Report: Source row payloads never persisted, making row data unavailable

## Summary

- Although the target file is `src/elspeth/core/landscape/row_data.py`, a P0 issue prevents it from ever returning `AVAILABLE`: source row payloads are not persisted during normal runs, so `rows.source_data_ref` stays NULL and `get_row_data` returns `NEVER_STORED`, violating the non-negotiable audit requirement to store raw source data.

## Severity

- Severity: critical
- Priority: P0

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

- Goal or task prompt: Static analysis deep bug audit of `src/elspeth/core/landscape/row_data.py`.
- Model/version: GPT-5 (Codex CLI).
- Tooling and permissions (sandbox/approvals): Read-only filesystem sandbox; approval_policy=never.
- Determinism details (seed, run ID): Unknown
- Notable tool calls or steps: Reviewed row data types and traced payload storage paths in recorder, tokens, and recovery code.

## Steps To Reproduce

1. Configure a pipeline with a payload store and run any source to create rows.
2. Call `LandscapeRecorder.get_row_data(row_id)` for a row created by the run (or attempt resume via checkpoint recovery).
3. Observe `RowDataState.NEVER_STORED` (or a resume ValueError about missing `source_data_ref`).

## Expected Behavior

- With a payload store configured, source row payloads are stored and `get_row_data` returns `RowDataState.AVAILABLE` with the payload; resume should succeed.

## Actual Behavior

- Rows created by the normal pipeline have no `source_data_ref`, so `get_row_data` returns `RowDataState.NEVER_STORED`, and resume fails with missing-payload errors.

## Evidence

- Logs or stack traces: Unknown
- Artifacts (paths, IDs, screenshots): `src/elspeth/engine/tokens.py:73`, `src/elspeth/core/landscape/recorder.py:694`, `src/elspeth/core/landscape/recorder.py:1825`, `src/elspeth/core/checkpoint/recovery.py:192`, `CLAUDE.md:23`
- Minimal repro input (attach or link): Unknown

## Impact

- User-facing impact: `get_row_data` cannot return payloads; explain/resume tooling cannot recover raw source rows.
- Data integrity / security impact: Audit trail violates the requirement to store raw source entries, undermining lineage verification.
- Performance or cost impact: Unknown

## Root Cause Hypothesis

- P0 root cause is outside the target file: `TokenManager.create_initial_token` never persists row payloads or passes a `payload_ref`, and `LandscapeRecorder.create_row` does not auto-store rows, leaving `rows.source_data_ref` NULL and forcing `get_row_data` to return `NEVER_STORED`.

## Proposed Fix

- Code changes (modules/files): Add automatic row payload persistence and `payload_ref` assignment in `src/elspeth/core/landscape/recorder.py` (or in `src/elspeth/engine/tokens.py` before `create_row`) when a payload store is configured.
- Config or schema changes: None.
- Tests to add/update: Add an integration test that runs a source with a payload store and asserts `get_row_data` returns AVAILABLE and `rows.source_data_ref` is set.
- Risks or migration steps: Increased payload storage usage; ensure retention policy handles growth.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `CLAUDE.md:23`
- Observed divergence: Source entry payloads are not persisted in normal runs, so raw data is missing from the audit trail.
- Reason (if known): Unknown
- Alignment plan or decision needed: Implement automatic source payload storage at row creation and document retention behavior.

## Acceptance Criteria

- Running a pipeline with a payload store configured results in `rows.source_data_ref` populated and `get_row_data` returning AVAILABLE; resume does not error due to missing payload.

## Tests

- Suggested tests to run: `tests/core/landscape/test_recorder_row_data.py`
- New tests required: Yes, an engine-level integration test asserting source payloads are stored.

## Notes / Links

- Related issues/PRs: Unknown
- Related design docs: `CLAUDE.md:23`
