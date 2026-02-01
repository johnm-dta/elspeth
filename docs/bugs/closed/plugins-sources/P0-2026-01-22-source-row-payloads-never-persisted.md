# Bug Report: Source row payloads never persisted, making row data unavailable

## Summary

Source row payloads are not persisted during normal runs, so `rows.source_data_ref` stays NULL and `get_row_data` returns `NEVER_STORED`, violating the non-negotiable audit requirement to store raw source data at the source entry point.

## Severity

- Severity: critical
- Priority: P0

## Reporter

- Name or handle: Codex (static analysis agent)
- Date: 2026-01-22
- Related run/issue ID: Unknown

## Environment

- Commit/branch: main (d8df733)
- OS: Linux
- Python version: 3.12+
- Config profile / env vars: Any configuration with a payload store
- Data set or fixture: Any source producing rows

## Agent Context (if relevant)

- Goal or task prompt: Static analysis deep bug audit of `src/elspeth/core/landscape/row_data.py`
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): Read-only filesystem sandbox; approval_policy=never
- Determinism details (seed, run ID): Unknown
- Notable tool calls or steps: Reviewed row data types and traced payload storage paths in recorder, tokens, and recovery code

## Steps To Reproduce

1. Configure a pipeline with a payload store and run any source to create rows
2. Call `LandscapeRecorder.get_row_data(row_id)` for a row created by the run (or attempt resume via checkpoint recovery)
3. Observe `RowDataState.NEVER_STORED` (or a resume ValueError about missing `source_data_ref`)

## Expected Behavior

- With a payload store configured, source row payloads are stored and `get_row_data` returns `RowDataState.AVAILABLE` with the payload
- Resume should succeed with access to original row data

## Actual Behavior

- Rows created by the normal pipeline have no `source_data_ref`, so `get_row_data` returns `RowDataState.NEVER_STORED`
- Resume fails with missing-payload errors

## Evidence

- Logs or stack traces: Unknown
- Artifacts (paths, IDs, screenshots):
  - `src/elspeth/engine/tokens.py:73` - TokenManager.create_initial_token never persists row payloads
  - `src/elspeth/core/landscape/recorder.py:694` - create_row does not auto-store rows
  - `src/elspeth/core/landscape/recorder.py:1825` - get_row_data expects source_data_ref
  - `src/elspeth/core/checkpoint/recovery.py:192` - recovery expects payload availability
- Minimal repro input (attach or link): Any pipeline run with payload store configured

## Impact

- User-facing impact: `get_row_data` cannot return payloads; explain/resume tooling cannot recover raw source rows
- Data integrity / security impact: Audit trail violates the requirement to store raw source entries, undermining lineage verification. This is a non-negotiable audit requirement per CLAUDE.md:23
- Performance or cost impact: Unknown

## Root Cause Hypothesis

P0 root cause is outside the target file: `TokenManager.create_initial_token` never persists row payloads or passes a `payload_ref`, and `LandscapeRecorder.create_row` does not auto-store rows, leaving `rows.source_data_ref` NULL and forcing `get_row_data` to return `NEVER_STORED`.

## Proposed Fix

- Code changes (modules/files): Add automatic row payload persistence and `payload_ref` assignment in `src/elspeth/core/landscape/recorder.py` (or in `src/elspeth/engine/tokens.py` before `create_row`) when a payload store is configured
- Config or schema changes: None
- Tests to add/update: Add an integration test that runs a source with a payload store and asserts `get_row_data` returns AVAILABLE and `rows.source_data_ref` is set
- Risks or migration steps: Increased payload storage usage; ensure retention policy handles growth

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `CLAUDE.md:23` - "Source entry - Raw data stored before any processing" is a non-negotiable data storage point
- Observed divergence: Source entry payloads are not persisted in normal runs, so raw data is missing from the audit trail
- Reason (if known): Unknown - implementation gap
- Alignment plan or decision needed: Implement automatic source payload storage at row creation and document retention behavior

## Acceptance Criteria

- Running a pipeline with a payload store configured results in `rows.source_data_ref` populated
- `get_row_data` returns `RowDataState.AVAILABLE` with correct payload
- Resume does not error due to missing payload
- All existing tests pass

## Tests

- Suggested tests to run: `tests/core/landscape/test_recorder_row_data.py`
- New tests required: Yes, an engine-level integration test asserting source payloads are stored

## Notes / Links

- Related issues/PRs: Unknown
- Related design docs: `CLAUDE.md:23` - Data storage points (non-negotiable)

## Verification Status

- [x] Bug confirmed via reproduction
- [x] Root cause verified
- [x] Fix implemented
- [x] Tests added
- [x] Fix verified

## Resolution

**Status:** FIXED
**Fixed in commit:** 3399faf
**Fixed on:** 2026-01-22

**Solution implemented:**
- Added `payload_store` parameter to `Orchestrator.run()`, `_execute_run()`, and `_process_resumed_rows()`
- Wired `payload_store` through `RowProcessor.__init__()` to `TokenManager`
- Implemented payload storage in `TokenManager.create_initial_token()` before calling `recorder.create_row()`
- Added integration test `tests/integration/test_source_payload_storage.py` verifying payloads are stored and retrievable

**Files changed:**
- `src/elspeth/engine/orchestrator.py` - Added payload_store parameter and wiring
- `src/elspeth/engine/processor.py` - Passed payload_store to TokenManager
- `src/elspeth/engine/tokens.py` - Implemented payload storage at row creation
- `tests/integration/test_source_payload_storage.py` - New integration test
- `config/cicd/no_bug_hiding.yaml` - Updated line numbers after changes
