# Bug Report: BatchReplicate output_schema omits copy_index when include_copy_index=True

## Summary

- BatchReplicate adds a copy_index field to output rows by default, but output_schema is set to the input schema, so strict schemas and downstream validators do not reflect the actual output shape.

## Severity

- Severity: major
- Priority: P2

## Reporter

- Name or handle: codex
- Date: 2026-01-21
- Related run/issue ID: N/A

## Environment

- Commit/branch: not checked
- OS: not checked (workspace sandbox)
- Python version: not checked
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: deep dive into src/elspeth/plugins/transforms for bugs
- Model/version: GPT-5 Codex
- Tooling and permissions (sandbox/approvals): workspace-write sandbox, no escalations
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: reviewed BatchReplicate implementation

## Steps To Reproduce

1. Configure batch_replicate with a strict schema that does not include copy_index and leave include_copy_index at its default (True).
2. Run an aggregation using batch_replicate with output_mode: transform.
3. Observe output rows include copy_index, but output_schema remains the strict input schema.

## Expected Behavior

- output_schema reflects the actual output shape (dynamic or includes optional copy_index) so schema compatibility and downstream validation are accurate.

## Actual Behavior

- output_schema is identical to input_schema even when output rows include copy_index.

## Evidence

- Output schema set to input schema: src/elspeth/plugins/transforms/batch_replicate.py:92-99
- copy_index added to output rows: src/elspeth/plugins/transforms/batch_replicate.py:133-136

## Impact

- User-facing impact: strict downstream schemas or sinks can reject rows unexpectedly.
- Data integrity / security impact: schema contracts are inaccurate, undermining validation guarantees.
- Performance or cost impact: potential pipeline failures and retries.

## Root Cause Hypothesis

- BatchReplicate assumes output shape matches input, but include_copy_index introduces new fields that are not represented in output_schema.

## Proposed Fix

- Code changes (modules/files): src/elspeth/plugins/transforms/batch_replicate.py
- Config or schema changes: set output_schema to dynamic when include_copy_index is True or extend schema with optional copy_index.
- Tests to add/update: add transform tests for output_schema behavior with include_copy_index True.
- Risks or migration steps: if users rely on strict schema, document the added field or require include_copy_index False.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): docs/contracts/plugin-protocol.md:334-339 (output_schema describes outgoing rows)
- Observed divergence: output_schema does not include copy_index even though output rows add it.
- Reason (if known): output_schema reused from input schema.
- Alignment plan or decision needed: make output_schema dynamic or explicitly include copy_index.

## Acceptance Criteria

- When include_copy_index is True, output_schema allows copy_index (dynamic or explicit optional field).
- Schema validation no longer rejects output rows that include copy_index.

## Tests

- Suggested tests to run: pytest tests/plugins/transforms/test_batch_replicate.py
- New tests required: yes, output_schema expectations for include_copy_index True/False.

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: docs/contracts/plugin-protocol.md
