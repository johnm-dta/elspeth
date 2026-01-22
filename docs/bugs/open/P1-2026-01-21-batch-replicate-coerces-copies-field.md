# Bug Report: BatchReplicate coerces copies_field and masks upstream type bugs

## Summary

- BatchReplicate uses int() coercion and defaults for copies_field values, which violates the transform contract (no coercion at transform boundary) and can silently produce incorrect replication counts.

## Severity

- Severity: major
- Priority: P1

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

1. Configure batch_replicate with copies_field: "copies" and default_copies: 1.
2. Provide a row where copies is "3" (string) or "abc" or None.
3. Observe the transform coerces or defaults instead of failing.

## Expected Behavior

- Wrong types in pipeline data should crash or surface as transform errors (upstream bug), not be coerced or defaulted.

## Actual Behavior

- Non-int values are coerced via int() or replaced by default_copies, silently changing behavior.

## Evidence

- Coercion and defaulting logic: src/elspeth/plugins/transforms/batch_replicate.py:122-130
- Contract prohibits coercion at transform boundary: docs/contracts/plugin-protocol.md:151-176

## Impact

- User-facing impact: replication counts are silently wrong on bad data.
- Data integrity / security impact: audit trail records incorrect outputs without surfacing upstream bug.
- Performance or cost impact: extra or missing rows change downstream workload.

## Root Cause Hypothesis

- BatchReplicate treats copies_field as untrusted and attempts to coerce, contradicting pipeline trust model for transforms.

## Proposed Fix

- Code changes (modules/files): src/elspeth/plugins/transforms/batch_replicate.py
- Config or schema changes: require copies_field type in schema and access directly; remove int() coercion and default fallback for type errors.
- Tests to add/update: add tests asserting wrong-type copies_field raises or returns TransformResult.error.
- Risks or migration steps: pipelines with invalid types will now fail fast (intended per trust model).

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): docs/contracts/plugin-protocol.md:151-176 (Transforms MUST NOT coerce types)
- Observed divergence: transform converts/normalizes wrong types instead of failing.
- Reason (if known): defensive handling added for convenience.
- Alignment plan or decision needed: enforce strict typing and fail on wrong types.

## Acceptance Criteria

- copies_field type violations cause a crash or TransformResult.error (no coercion/defaulting).
- Valid int values replicate exactly; invalid types surface upstream bugs.

## Tests

- Suggested tests to run: pytest tests/plugins/transforms/test_batch_replicate.py
- New tests required: yes, wrong-type copies_field behavior.

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: docs/contracts/plugin-protocol.md
