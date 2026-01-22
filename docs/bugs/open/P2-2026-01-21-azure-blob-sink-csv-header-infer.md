# Bug Report: AzureBlobSink CSV headers are inferred from first row, ignoring schema

## Summary

- `AzureBlobSink` derives CSV fieldnames from the first row only, rather than from the configured schema.
- Later rows with additional valid fields can crash (`csv.DictWriter` extrasaction=raise), and optional schema fields missing from the first row are silently omitted from the header.

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
- Notable tool calls or steps: code inspection of `src/elspeth/plugins/azure/blob_sink.py`

## Steps To Reproduce

1. Configure `AzureBlobSink` with an explicit schema that includes an optional field (e.g., `id`, `name`, `email`).
2. Write a batch where the first row omits `email` but later rows include it.
3. Run the pipeline.

## Expected Behavior

- CSV headers are derived from the schema (or a deterministic union), so optional fields are present and extra fields do not crash the sink.

## Actual Behavior

- Fieldnames are derived from the first row only, so later rows with extra fields raise `ValueError`, or optional fields are dropped from the output header.

## Evidence

- Fieldnames are taken from `rows[0].keys()` with no schema awareness:
  - `src/elspeth/plugins/azure/blob_sink.py:334`
  - `src/elspeth/plugins/azure/blob_sink.py:338`
  - `src/elspeth/plugins/azure/blob_sink.py:350`

## Impact

- User-facing impact: valid runs can crash mid-batch on later rows.
- Data integrity / security impact: optional fields may be silently omitted from CSV output.
- Performance or cost impact: failed runs and retried uploads.

## Root Cause Hypothesis

- CSV serialization was implemented without reusing CSVSink's schema-aware header selection.

## Proposed Fix

- Code changes (modules/files):
  - `src/elspeth/plugins/azure/blob_sink.py`: derive fieldnames from schema when explicit (mirroring `CSVSink._get_fieldnames_from_schema_or_row()`), and only fall back to row keys for dynamic schemas.
- Config or schema changes: none.
- Tests to add/update:
  - Add a test that writes rows with optional fields missing from the first row and ensure no crash and headers include schema fields.
- Risks or migration steps: none.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `src/elspeth/plugins/sinks/csv_sink.py` header selection behavior.
- Observed divergence: Azure CSV sink does not respect schema-configured headers.
- Reason (if known): Azure sink reimplemented CSV serialization without schema logic.
- Alignment plan or decision needed: align Azure CSV sink with CSVSink schema handling.

## Acceptance Criteria

- Explicit schemas drive CSV headers for AzureBlobSink.
- Runs with optional or late-appearing fields no longer crash or drop columns.

## Tests

- Suggested tests to run:
  - `pytest tests/plugins/azure/test_blob_sink.py -k csv`
- New tests required: yes (schema-aware header selection)

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: N/A
