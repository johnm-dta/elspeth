# Bug Report: AzureBlobSource CSV parse errors abort instead of quarantine/audit

## Summary

- Malformed CSV content in Azure Blob can raise a parse exception that aborts the source load without recording a validation error or yielding a quarantined row.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: Codex
- Date: 2026-01-25
- Related run/issue ID: N/A

## Environment

- Commit/branch: 8635789
- OS: unknown
- Python version: unknown
- Config profile / env vars: N/A
- Data set or fixture: Azure Blob CSV with a malformed row (e.g., unmatched quote or wrong column count)

## Agent Context (if relevant)

- Goal or task prompt: Static analysis deep bug audit of `src/elspeth/plugins/azure/blob_source.py`
- Model/version: Codex (GPT-5)
- Tooling and permissions (sandbox/approvals): read-only sandbox
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code review only

## Steps To Reproduce

1. Upload a CSV blob containing a malformed row (e.g., unbalanced quotes or invalid field count).
2. Configure AzureBlobSource with `format: csv` and `on_validation_failure: quarantine`.
3. Run a pipeline or iterate `list(source.load(ctx))`.

## Expected Behavior

- CSV parse errors at the external-data boundary are recorded via `ctx.record_validation_error` and yielded as `SourceRow.quarantined` (or a file-level quarantined row) so the run does not crash.

## Actual Behavior

- `pandas.read_csv` raises a parse exception that propagates and aborts source loading; no validation error record or quarantine row is emitted.

## Evidence

- `src/elspeth/plugins/azure/blob_source.py:363`: `pd.read_csv(...)` is called without a try/except or quarantine handling.
- `src/elspeth/plugins/sources/csv_source.py:115`: CSV parse errors are quarantined and recorded at the source boundary in the file-based source for comparison.

## Impact

- User-facing impact: Pipeline can crash on a single malformed CSV row in a blob.
- Data integrity / security impact: Parse failures are not recorded in the audit trail, violating the “record/quarantine external data” requirement.
- Performance or cost impact: N/A

## Root Cause Hypothesis

- `_load_csv` lacks any parse-error handling and does not emit `ctx.record_validation_error`/`SourceRow.quarantined` for malformed CSV content.

## Proposed Fix

- Code changes (modules/files):
  - Add parse-error handling in `src/elspeth/plugins/azure/blob_source.py` to quarantine and record CSV parse failures (either by switching to `csv.reader` like CSVSource or catching `pandas.errors.ParserError` and emitting a file-level quarantine).
- Config or schema changes: None
- Tests to add/update:
  - Add tests in `tests/plugins/azure/test_blob_source.py` that simulate malformed CSV content and assert quarantine behavior (including discard mode).
- Risks or migration steps:
  - Behavior change: parse errors will quarantine instead of raising; update any tests or expectations that currently assume exceptions.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `CLAUDE.md:80`
- Observed divergence: External data parse failures are not quarantined and recorded, and instead abort the run.
- Reason (if known): Missing parse-error handling in `_load_csv`.
- Alignment plan or decision needed: Implement quarantine/recording for CSV parse errors in the Azure blob source.

## Acceptance Criteria

- Malformed CSV content yields a quarantined SourceRow (or file-level quarantine) and records a validation error without crashing the pipeline.

## Tests

- Suggested tests to run: `.venv/bin/python -m pytest tests/plugins/azure/test_blob_source.py -k csv`
- New tests required: yes, malformed CSV quarantine and discard-mode tests

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `CLAUDE.md`
---
# Bug Report: AzureBlobSource JSON array parse/shape errors crash without quarantine/audit

## Summary

- Invalid JSON arrays, missing `data_key`, or non-array JSON payloads raise `ValueError` and abort the source load without recording a validation error or yielding a quarantined row.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: Codex
- Date: 2026-01-25
- Related run/issue ID: N/A

## Environment

- Commit/branch: 8635789
- OS: unknown
- Python version: unknown
- Config profile / env vars: N/A
- Data set or fixture: Azure Blob JSON with invalid JSON or object-not-array payload

## Agent Context (if relevant)

- Goal or task prompt: Static analysis deep bug audit of `src/elspeth/plugins/azure/blob_source.py`
- Model/version: Codex (GPT-5)
- Tooling and permissions (sandbox/approvals): read-only sandbox
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code review only

## Steps To Reproduce

1. Upload a blob with invalid JSON (e.g., missing closing brackets) or a JSON object when an array is expected.
2. Configure AzureBlobSource with `format: json` and `on_validation_failure: quarantine`.
3. Run a pipeline or iterate `list(source.load(ctx))`.

## Expected Behavior

- JSON parse/shape errors at the external-data boundary are recorded via `ctx.record_validation_error` (schema_mode `parse`) and yield a quarantined SourceRow (or file-level quarantine) without crashing.

## Actual Behavior

- `_load_json_array` raises `ValueError` for invalid JSON, missing `data_key`, or non-array JSON, aborting the run and skipping audit/quarantine recording.

## Evidence

- `src/elspeth/plugins/azure/blob_source.py:396`: JSON decoding errors are converted into `ValueError` without quarantine.
- `src/elspeth/plugins/azure/blob_source.py:405`: Missing `data_key` raises `ValueError` without audit recording.
- `src/elspeth/plugins/azure/blob_source.py:410`: Non-array JSON raises `ValueError`.
- `tests/plugins/azure/test_blob_source.py:304`: Tests currently expect a `ValueError` for non-array JSON.
- `src/elspeth/plugins/sources/json_source.py:149`: File-based JSON source records parse errors and yields quarantined rows for invalid JSON.

## Impact

- User-facing impact: Pipelines crash on malformed JSON blobs instead of quarantining and continuing.
- Data integrity / security impact: Parse failures are not recorded in the audit trail, violating the auditability standard.
- Performance or cost impact: N/A

## Root Cause Hypothesis

- `_load_json_array` handles parse/shape errors by raising exceptions rather than recording validation failures and quarantining, contrary to the trust model.

## Proposed Fix

- Code changes (modules/files):
  - Update `src/elspeth/plugins/azure/blob_source.py` to mirror JSONSource: record JSONDecodeError and shape failures via `ctx.record_validation_error` and yield a quarantined row (or file-level quarantine) instead of raising.
- Config or schema changes: None
- Tests to add/update:
  - Update `tests/plugins/azure/test_blob_source.py` to expect quarantine instead of `ValueError` for invalid JSON and non-array payloads; add a test for missing `data_key` quarantine.
- Risks or migration steps:
  - Behavior change: parse/shape errors will no longer raise; adjust existing tests and any consumers expecting exceptions.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `CLAUDE.md:80`
- Observed divergence: External data parse failures are not quarantined or recorded; they abort the run.
- Reason (if known): `_load_json_array` raises `ValueError` instead of recording validation errors.
- Alignment plan or decision needed: Implement quarantine/recording for JSON array parse/shape failures in AzureBlobSource.

## Acceptance Criteria

- Invalid JSON or unexpected JSON shape yields a quarantined SourceRow (or file-level quarantine) and records a validation error without crashing.

## Tests

- Suggested tests to run: `.venv/bin/python -m pytest tests/plugins/azure/test_blob_source.py -k json`
- New tests required: yes, parse/shape quarantine coverage for JSON array input

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `CLAUDE.md`
