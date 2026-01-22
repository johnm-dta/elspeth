# Bug Report: AzureBlobSource `has_header=false` does not map columns to schema fields

## Summary

- When `csv_options.has_header=false`, AzureBlobSource reads CSV with `header=None`, producing numeric column names (`0`, `1`, ...).
- The source never maps these columns to schema field names, so valid headerless CSVs will fail schema validation and be quarantined or dropped.

## Severity

- Severity: major
- Priority: P2

## Reporter

- Name or handle: codex
- Date: 2026-01-21
- Related run/issue ID: N/A

## Environment

- Commit/branch: `ae2c0e6f088f467276582fa8016f91b4d3bb26c7` (main)
- OS: Linux 6.8.0-90-generic
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: deep dive into `src/elspeth/plugins/azure` for bugs and create reports
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code inspection of `src/elspeth/plugins/azure/blob_source.py`

## Steps To Reproduce

1. Configure `AzureBlobSource` with `format: csv`, `csv_options.has_header: false`, and a schema with named fields (e.g., `id`, `name`).
2. Upload a headerless CSV blob with two columns matching that schema.
3. Run the pipeline.

## Expected Behavior

- Column names are mapped to schema field names (or configuration rejects headerless CSV without explicit column mapping).

## Actual Behavior

- Pandas assigns numeric column names, producing rows like `{ "0": "...", "1": "..." }`, which do not match schema field names and fail validation.

## Evidence

- `has_header=false` is passed to pandas as `header=None` with no schema-based names:
  - `src/elspeth/plugins/azure/blob_source.py:350`
  - `src/elspeth/plugins/azure/blob_source.py:362`
  - `src/elspeth/plugins/azure/blob_source.py:371`

## Impact

- User-facing impact: headerless CSV inputs are effectively unusable with named schemas.
- Data integrity / security impact: valid source rows are quarantined due to mismatched column names.
- Performance or cost impact: wasted runs and unnecessary quarantine volume.

## Root Cause Hypothesis

- The implementation does not translate column positions to schema field names when `has_header` is disabled.

## Proposed Fix

- Code changes (modules/files):
  - `src/elspeth/plugins/azure/blob_source.py`: when `has_header=false` and schema is explicit, pass `names=[field_def.name ...]` to `pd.read_csv` (and optionally error if schema is dynamic).
- Config or schema changes: none.
- Tests to add/update:
  - Add a test for headerless CSV with explicit schema to confirm columns map correctly.
- Risks or migration steps: none.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `src/elspeth/plugins/config_base.py` (schema-required data plugins).
- Observed divergence: headerless CSV option exists but does not honor schema field names.
- Reason (if known): missing mapping logic when `has_header=false`.
- Alignment plan or decision needed: define expected behavior for headerless CSV and enforce it.

## Acceptance Criteria

- Headerless CSV inputs with explicit schemas validate successfully.
- If schema is dynamic, behavior is explicit (either allow numeric column names or reject configuration).

## Tests

- Suggested tests to run:
  - `pytest tests/plugins/azure/test_blob_source.py -k csv`
- New tests required: yes (headerless CSV column mapping)

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: N/A
