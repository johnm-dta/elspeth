# Bug Report: Azure content safety and prompt shield add error fields not declared in output_schema

## Summary

- In batch mode, AzureContentSafety and AzurePromptShield embed _content_safety_error or _prompt_shield_error in output rows, but output_schema is identical to input_schema, so strict schemas and downstream validators do not match the actual output shape.

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
- Notable tool calls or steps: reviewed Azure content_safety and prompt_shield implementations

## Steps To Reproduce

1. Configure a strict schema without error fields and set pool_size > 1 (batch mode).
2. Provide input rows that trigger a content safety or prompt shield violation.
3. Observe output rows include _content_safety_error or _prompt_shield_error even though output_schema is unchanged.

## Expected Behavior

- output_schema reflects the actual output shape (dynamic or explicit optional error fields).

## Actual Behavior

- output_schema remains identical to input_schema while output rows add error fields in batch mode.

## Evidence

- output_schema set to input schema: src/elspeth/plugins/transforms/azure/content_safety.py:160-162; src/elspeth/plugins/transforms/azure/prompt_shield.py:126-133
- error fields added in batch mode: src/elspeth/plugins/transforms/azure/content_safety.py:418-425; src/elspeth/plugins/transforms/azure/prompt_shield.py:387-395
- Transform output_schema should describe outgoing rows: docs/contracts/plugin-protocol.md:334-339

## Impact

- User-facing impact: strict sink validation can fail on unexpected error fields.
- Data integrity / security impact: schema contracts are inaccurate, complicating validation and auditing.
- Performance or cost impact: potential pipeline failures and retries.

## Root Cause Hypothesis

- Batch error embedding was added without updating output_schema to include the new fields.

## Proposed Fix

- Code changes (modules/files): src/elspeth/plugins/transforms/azure/content_safety.py, src/elspeth/plugins/transforms/azure/prompt_shield.py
- Config or schema changes: use dynamic output_schema in batch mode or extend schema with optional error fields.
- Tests to add/update: add tests asserting output_schema allows error fields when batch mode is enabled.
- Risks or migration steps: document behavior change for strict schemas.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): docs/contracts/plugin-protocol.md:334-339
- Observed divergence: output rows include new fields not represented in output_schema.
- Reason (if known): output_schema reused from input schema.
- Alignment plan or decision needed: align output_schema with actual output fields.

## Acceptance Criteria

- Batch mode outputs with error fields validate against output_schema.
- Schema compatibility checks reflect the presence of error fields.

## Tests

- Suggested tests to run: pytest tests/plugins/transforms/azure/test_content_safety.py pytest tests/plugins/transforms/azure/test_prompt_shield.py
- New tests required: yes, output_schema checks for batch error fields.

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: docs/contracts/plugin-protocol.md
