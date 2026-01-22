# Bug Report: CSVSource crashes on malformed CSV rows instead of quarantining them

## Summary

- `CSVSource` uses `pandas.read_csv` without handling parser errors, so a single malformed line (extra columns, bad quoting) aborts the run.
- External data is Tier 3; malformed rows should be quarantined and recorded, not crash the pipeline.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: codex
- Date: 2026-01-21
- Related run/issue ID: N/A (static analysis)

## Environment

- Commit/branch: `main` @ `ae2c0e6f088f467276582fa8016f91b4d3bb26c7`
- OS: Linux 6.8.0-90-generic
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: CSV containing at least one malformed row

## Agent Context (if relevant)

- Goal or task prompt: deep dive into `src/elspeth/plugins/sources`, identify bugs, create tickets
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code inspection (no runtime execution)

## Steps To Reproduce

1. Create a CSV file with inconsistent columns or malformed quotes, e.g.:
   - `id,name`
   - `1,Alice`
   - `2,Bob,extra`
   - `3,Carol`
2. Configure `CSVSource` with `schema: { fields: dynamic }` and `on_validation_failure: quarantine`.
3. Run the pipeline or call `CSVSource.load()` with a `PluginContext`.

## Expected Behavior

- The malformed row is recorded as a validation error and quarantined (or discarded if configured).
- Valid rows before/after the malformed line still load and process.

## Actual Behavior

- `pandas.errors.ParserError` escapes `CSVSource.load()` and crashes the run.

## Evidence

- `CSVSource.load()` calls `pd.read_csv(...)` without handling `ParserError`: `src/elspeth/plugins/sources/csv_source.py:94-106`
- Pandas defaults `on_bad_lines="error"`, so malformed rows raise before per-row validation.

## Impact

- User-facing impact: a single corrupt line can halt ingestion of an entire dataset.
- Data integrity / security impact: external garbage causes a hard crash instead of quarantine; audit trail lacks the bad row record.
- Performance or cost impact: reruns and manual data cleanup required.

## Root Cause Hypothesis

- CSV parsing errors occur before row-level schema validation and are not treated as row-level quarantine events.

## Proposed Fix

- Code changes (modules/files):
  - `src/elspeth/plugins/sources/csv_source.py`: use `on_bad_lines` (callable) to capture malformed lines and continue, or switch to a row-wise CSV parser that can catch and quarantine bad rows.
  - Record validation errors with raw line content and line number for auditability.
- Config or schema changes: none.
- Tests to add/update:
  - Add a test that includes a malformed CSV row and asserts: valid rows load, malformed row is quarantined or discarded per config.
- Risks or migration steps:
  - Ensure raw line recording is size-limited to avoid payload bloat in audit storage.

## Architectural Deviations

- Spec or doc reference: `CLAUDE.md` Tier 3 external data handling; `docs/plans/completed/2026-01-16-contracts-subsystem.md` ("CSV with malformed rows")
- Observed divergence: malformed CSV row crashes the run instead of being quarantined.
- Reason (if known): parse errors are raised by pandas before row-level validation.
- Alignment plan or decision needed: define CSV malformed-row quarantine strategy (row-level vs file-level) and ensure audit record is captured.

## Acceptance Criteria

- Malformed CSV rows are quarantined or discarded without aborting the run.
- Valid rows before/after a malformed line still process.
- Validation errors include enough context (line number, raw line) for audit.

## Tests

- Suggested tests to run: `pytest tests/plugins/sources/test_csv_source.py`
- New tests required: yes

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `CLAUDE.md`
