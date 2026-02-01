# Bug Report: AzureBlobSource CSV Parse Errors Abort Instead of Quarantine

## Summary

- AzureBlobSource CSV parsing raises exceptions on malformed rows instead of quarantining them, violating CLAUDE.md Three-Tier Trust Model requirement to validate external data at boundary.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: Branch Bug Scan (fix/rc1-bug-burndown-session-4)
- Date: 2026-01-25
- Related run/issue ID: BUG-BLOB-01

## Environment

- Commit/branch: fix/rc1-bug-burndown-session-4
- OS: unknown
- Python version: unknown
- Config profile / env vars: N/A
- Data set or fixture: Azure Blob CSV with malformed rows

## Agent Context (if relevant)

- Goal or task prompt: Static analysis agent doing a deep bug audit of blob_source.py
- Model/version: Codex (GPT-5)
- Tooling and permissions (sandbox/approvals): read-only sandbox
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code review only

## Steps To Reproduce

1. Upload CSV to Azure Blob with malformed row (wrong column count, encoding error, etc.).
2. Run pipeline with AzureBlobSource.
3. Observe pipeline crash on parse error.

## Expected Behavior

- Parse errors should quarantine row with reason.
- Pipeline continues processing valid rows.
- Quarantined rows recorded in audit trail.

## Actual Behavior

- Exception raised, pipeline aborts.
- No rows processed.
- No quarantine record.

## Evidence

```python
# Current code (blob_source.py)
for row in csv.DictReader(blob_data):
    yield row  # If parse fails → exception → pipeline abort
```

Per CLAUDE.md Three-Tier Trust Model:
- External data (Tier 3) = zero trust
- Must validate at boundary
- Parse failures should quarantine, not crash

## Impact

- User-facing impact: Single malformed row crashes entire pipeline.
- Data integrity / security impact: No audit record of rejected rows.
- Performance or cost impact: Full pipeline re-run required after fixing data.

## Root Cause Hypothesis

- Source treats external CSV data as trusted instead of validating at boundary.

## Proposed Fix

```python
for row_num, row in enumerate(csv.DictReader(blob_data), start=1):
    try:
        validated = self.validate_row(row)
        yield validated
    except ValidationError as e:
        yield {
            "_quarantined": True,
            "_quarantine_reason": f"CSV parse error at row {row_num}: {e}",
            "_raw_data": str(row),
            "_row_number": row_num,
        }
```

- Config or schema changes: None.
- Tests to add/update:
  - `test_csv_parse_error_quarantines_row()` - Verify quarantine behavior
  - `test_csv_valid_rows_processed_despite_errors()` - Verify pipeline continues

- Risks or migration steps: None (bug fix only).

## Architectural Deviations

- Spec or doc reference: `CLAUDE.md` - Three-Tier Trust Model
- Observed divergence: Source crashes on external data errors instead of quarantining.
- Reason (if known): Missing boundary validation.
- Alignment plan or decision needed: Add validation and quarantine logic to all sources.

## Acceptance Criteria

- Malformed CSV rows quarantined, not crashed.
- Valid rows processed successfully.
- Quarantine reason recorded in audit trail.

## Tests

- Suggested tests to run: `pytest tests/plugins/azure/test_blob_source.py`
- New tests required: yes (2 tests above)

## Notes / Links

- Related issues/PRs: N/A
- Related design docs:
  - `docs/bugs/BRANCH_BUG_TRIAGE_2026-01-25.md`
  - `CLAUDE.md` - Three-Tier Trust Model
