# Bug Report: RunRepository masks invalid export_status values

## Summary

- RunRepository.load treats falsy export_status values as None, so invalid values like "" bypass ExportStatus coercion and do not crash, masking Tier 1 data corruption and misreporting export status.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: Unknown
- Date: Unknown
- Related run/issue ID: Unknown

## Environment

- Commit/branch: Unknown
- OS: Unknown
- Python version: Unknown
- Config profile / env vars: Unknown
- Data set or fixture: Unknown

## Agent Context (if relevant)

- Goal or task prompt: Static analysis bug audit of repositories.py
- Model/version: Codex (GPT-5)
- Tooling and permissions (sandbox/approvals): sandbox_mode=read-only, approval_policy=never
- Determinism details (seed, run ID): Unknown
- Notable tool calls or steps: Reviewed repository, contracts, schema, and recorder mappings

## Steps To Reproduce

1. Create a mock row with export_status="" (or another falsy non-None value) and valid required fields.
2. Call RunRepository.load(mock_row).
3. Inspect Run.export_status.

## Expected Behavior

- Invalid export_status values should raise ValueError (or otherwise crash) during ExportStatus coercion; only None should map to None.

## Actual Behavior

- export_status is returned as None with no exception, masking the invalid value.

## Evidence

- Logs or stack traces: Unknown
- Artifacts (paths, IDs, screenshots): `src/elspeth/core/landscape/repositories.py:55`; `src/elspeth/core/landscape/recorder.py:338`; `src/elspeth/contracts/audit.py:43`
- Minimal repro input (attach or link): Mock row with export_status=""

## Impact

- User-facing impact: Export status can appear "unset" when the DB contains an invalid value.
- Data integrity / security impact: Violates Tier 1 crash-on-anomaly rule by silently accepting corrupted audit data.
- Performance or cost impact: Unknown

## Root Cause Hypothesis

- export_status is gated by a truthiness check instead of an explicit None check, so falsy invalid values bypass enum validation.

## Proposed Fix

- Code changes (modules/files): Update RunRepository.load in `src/elspeth/core/landscape/repositories.py:55` to use `row.export_status is not None` before coercion.
- Config or schema changes: None
- Tests to add/update: Add a test for empty-string export_status in `tests/core/landscape/test_repositories.py`.
- Risks or migration steps: None

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `CLAUDE.md:40`
- Observed divergence: Invalid audit DB values can be silently coerced to None instead of crashing.
- Reason (if known): Unknown
- Alignment plan or decision needed: Enforce explicit None checks and allow invalid values to raise.

## Acceptance Criteria

- RunRepository.load raises ValueError for export_status="" (or other non-None invalid values) while preserving None as None and valid strings as ExportStatus enums.

## Tests

- Suggested tests to run: `pytest tests/core/landscape/test_repositories.py::TestRunRepository`
- New tests required: Add a case asserting empty-string export_status raises ValueError.

## Notes / Links

- Related issues/PRs: Unknown
- Related design docs: `CLAUDE.md:32`
---
# Bug Report: NodeRepository drops schema_mode and schema_fields

## Summary

- NodeRepository.load ignores schema_mode and schema_fields_json, returning Node objects without schema metadata even when the DB stores it, which makes audit lineage incomplete.

## Severity

- Severity: major
- Priority: P2

## Reporter

- Name or handle: Unknown
- Date: Unknown
- Related run/issue ID: Unknown

## Environment

- Commit/branch: Unknown
- OS: Unknown
- Python version: Unknown
- Config profile / env vars: Unknown
- Data set or fixture: Unknown

## Agent Context (if relevant)

- Goal or task prompt: Static analysis bug audit of repositories.py
- Model/version: Codex (GPT-5)
- Tooling and permissions (sandbox/approvals): sandbox_mode=read-only, approval_policy=never
- Determinism details (seed, run ID): Unknown
- Notable tool calls or steps: Reviewed repository, contracts, schema, and recorder mappings

## Steps To Reproduce

1. Create a mock node row with schema_mode="strict" and schema_fields_json='[{"name":"field"}]'.
2. Call NodeRepository.load(mock_row).
3. Inspect Node.schema_mode and Node.schema_fields.

## Expected Behavior

- schema_mode should be populated from the DB, and schema_fields should be parsed from schema_fields_json.

## Actual Behavior

- Node.schema_mode and Node.schema_fields are None even when the row has values.

## Evidence

- Logs or stack traces: Unknown
- Artifacts (paths, IDs, screenshots): `src/elspeth/core/landscape/repositories.py:74`; `src/elspeth/contracts/audit.py:69`; `src/elspeth/core/landscape/schema.py:64`; `src/elspeth/core/landscape/recorder.py:603`
- Minimal repro input (attach or link): Mock row with schema_mode set and schema_fields_json containing JSON

## Impact

- User-facing impact: Explain/export output omits schema configuration for nodes.
- Data integrity / security impact: Audit metadata recorded in the DB is silently dropped in repository reads.
- Performance or cost impact: Unknown

## Root Cause Hypothesis

- NodeRepository.load was not updated to map schema_mode and parse schema_fields_json after schema tracking was added.

## Proposed Fix

- Code changes (modules/files): In `src/elspeth/core/landscape/repositories.py:74`, parse row.schema_fields_json with json.loads and pass schema_mode and schema_fields into Node.
- Config or schema changes: None
- Tests to add/update: Add tests covering schema_mode and schema_fields loading in `tests/core/landscape/test_repositories.py`.
- Risks or migration steps: None

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `CLAUDE.md:17`
- Observed divergence: Recorded schema metadata is not returned by the repository layer.
- Reason (if known): Unknown
- Alignment plan or decision needed: Map all schema metadata fields and parse schema_fields_json to maintain complete audit lineage.

## Acceptance Criteria

- NodeRepository.load returns schema_mode and parsed schema_fields for non-NULL DB values and raises on invalid schema_fields_json.

## Tests

- Suggested tests to run: `pytest tests/core/landscape/test_repositories.py::TestNodeRepository`
- New tests required: Add a case asserting schema_mode/schema_fields round-trip and invalid JSON raises.

## Notes / Links

- Related issues/PRs: Unknown
- Related design docs: `CLAUDE.md:17`
---
# Bug Report: BatchRepository drops trigger_type

## Summary

- BatchRepository.load never assigns trigger_type from the DB, so Batch objects lose aggregation trigger metadata that is stored and expected by the contract.

## Severity

- Severity: minor
- Priority: P3

## Reporter

- Name or handle: Unknown
- Date: Unknown
- Related run/issue ID: Unknown

## Environment

- Commit/branch: Unknown
- OS: Unknown
- Python version: Unknown
- Config profile / env vars: Unknown
- Data set or fixture: Unknown

## Agent Context (if relevant)

- Goal or task prompt: Static analysis bug audit of repositories.py
- Model/version: Codex (GPT-5)
- Tooling and permissions (sandbox/approvals): sandbox_mode=read-only, approval_policy=never
- Determinism details (seed, run ID): Unknown
- Notable tool calls or steps: Reviewed repository, contracts, schema, and recorder mappings

## Steps To Reproduce

1. Create a mock batch row with trigger_type="timeout" (or any valid trigger type).
2. Call BatchRepository.load(mock_row).
3. Inspect Batch.trigger_type.

## Expected Behavior

- trigger_type should be populated from the DB row.

## Actual Behavior

- Batch.trigger_type is None regardless of the DB value.

## Evidence

- Logs or stack traces: Unknown
- Artifacts (paths, IDs, screenshots): `src/elspeth/core/landscape/repositories.py:224`; `src/elspeth/contracts/audit.py:270`; `src/elspeth/core/landscape/schema.py:255`; `src/elspeth/core/landscape/recorder.py:1435`
- Minimal repro input (attach or link): Mock row with trigger_type="timeout"

## Impact

- User-facing impact: Batch reads omit the trigger reason type in explain/export views.
- Data integrity / security impact: Audit metadata is incomplete when using repository reads.
- Performance or cost impact: Unknown

## Root Cause Hypothesis

- BatchRepository.load was not updated to include the trigger_type field when the batches table and contract added it.

## Proposed Fix

- Code changes (modules/files): Add trigger_type=row.trigger_type in BatchRepository.load in `src/elspeth/core/landscape/repositories.py:224`.
- Config or schema changes: None
- Tests to add/update: Extend batch repository tests in `tests/core/landscape/test_repositories.py` to include trigger_type.
- Risks or migration steps: None

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `CLAUDE.md:17`
- Observed divergence: Stored batch trigger metadata is not surfaced by the repository layer.
- Reason (if known): Unknown
- Alignment plan or decision needed: Map trigger_type on read to keep audit views complete.

## Acceptance Criteria

- BatchRepository.load returns trigger_type from the DB for all batches with non-NULL trigger_type.

## Tests

- Suggested tests to run: `pytest tests/core/landscape/test_repositories.py::TestBatchRepository`
- New tests required: Add a case asserting trigger_type is preserved.

## Notes / Links

- Related issues/PRs: Unknown
- Related design docs: `CLAUDE.md:17`
