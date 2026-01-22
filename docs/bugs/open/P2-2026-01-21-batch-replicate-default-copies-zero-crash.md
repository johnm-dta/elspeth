# Bug Report: BatchReplicate default_copies can be 0, causing success_multi([]) crash

## Summary

- BatchReplicate allows default_copies <= 0; when configured, the transform can emit zero output rows and then raises ValueError because TransformResult.success_multi forbids empty output.

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
- Notable tool calls or steps: reviewed BatchReplicate process logic

## Steps To Reproduce

1. Configure batch_replicate with default_copies: 0 (or a negative value).
2. Run aggregation with batch_replicate on any non-empty input batch.
3. Observe TransformResult.success_multi raises ValueError because output_rows is empty.

## Expected Behavior

- Configuration should reject default_copies < 1, or the transform should return a proper error result instead of crashing.

## Actual Behavior

- The transform builds zero output rows and then raises ValueError from TransformResult.success_multi.

## Evidence

- default_copies is not validated for minimum value: src/elspeth/plugins/transforms/batch_replicate.py:25-36
- copies loop uses range(copies) and can produce zero outputs: src/elspeth/plugins/transforms/batch_replicate.py:122-137
- success_multi([]) is invalid per protocol: docs/contracts/plugin-protocol.md:365-369

## Impact

- User-facing impact: pipeline crashes on valid-looking configuration.
- Data integrity / security impact: none (crash), but prevents processing.
- Performance or cost impact: wasted runs and retries.

## Root Cause Hypothesis

- BatchReplicateConfig does not constrain default_copies, and process() assumes at least one output row per input.

## Proposed Fix

- Code changes (modules/files): src/elspeth/plugins/transforms/batch_replicate.py
- Config or schema changes: validate default_copies >= 1 (Field with ge=1) and/or raise TransformResult.error when copies < 1.
- Tests to add/update: add config validation test and process test for invalid default_copies.
- Risks or migration steps: existing configs with default_copies <= 0 will now fail fast (intended).

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): docs/contracts/plugin-protocol.md:365-369 (success_multi([]) invalid)
- Observed divergence: transform can generate empty output_rows but still calls success_multi.
- Reason (if known): default_copies lacks validation.
- Alignment plan or decision needed: enforce minimum copies or return a proper error.

## Acceptance Criteria

- default_copies < 1 is rejected at config validation or converted into a TransformResult.error.
- BatchReplicate never calls success_multi([]).

## Tests

- Suggested tests to run: pytest tests/plugins/transforms/test_batch_replicate.py
- New tests required: yes, config validation and empty-output guard.

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: docs/contracts/plugin-protocol.md
