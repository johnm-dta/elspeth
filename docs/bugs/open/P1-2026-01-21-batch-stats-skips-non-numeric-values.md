# Bug Report: BatchStats skips non-numeric values instead of surfacing upstream type bugs

## Summary

- BatchStats converts values with float() and silently skips non-convertible inputs, which violates the transform contract (no coercion at transform boundary) and yields incorrect aggregates without surfacing upstream schema violations.

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
- Notable tool calls or steps: reviewed BatchStats process logic

## Steps To Reproduce

1. Configure batch_stats with value_field: "amount" and a strict schema expecting numeric types.
2. Provide a batch with one row where amount is "not_a_number".
3. Observe the transform silently skips the row and returns count/sum based only on numeric values.

## Expected Behavior

- Wrong types in pipeline data should crash or surface as transform errors (upstream bug), not be skipped.

## Actual Behavior

- Non-numeric values are ignored, producing aggregates that do not reflect the actual batch and masking upstream bugs.

## Evidence

- Coercion and skip logic: src/elspeth/plugins/transforms/batch_stats.py:100-132
- Contract prohibits coercion at transform boundary: docs/contracts/plugin-protocol.md:151-176

## Impact

- User-facing impact: aggregates are silently wrong when data types are invalid.
- Data integrity / security impact: audit trail records incorrect stats without surfacing upstream bug.
- Performance or cost impact: downstream decisions based on incorrect aggregates.

## Root Cause Hypothesis

- BatchStats treats value_field as untrusted and attempts to coerce/skip, contradicting the pipeline trust model for transforms.

## Proposed Fix

- Code changes (modules/files): src/elspeth/plugins/transforms/batch_stats.py
- Config or schema changes: validate input schema and fail on non-numeric types; remove float() coercion or treat conversion failures as TransformResult.error.
- Tests to add/update: update tests to assert failures on wrong types (currently expect skipping).
- Risks or migration steps: existing pipelines with invalid types will now fail fast (intended per trust model).

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): docs/contracts/plugin-protocol.md:151-176 (Transforms MUST NOT coerce types)
- Observed divergence: transform converts/ignores wrong types instead of failing.
- Reason (if known): defensive handling added for convenience.
- Alignment plan or decision needed: enforce strict typing and fail on wrong types.

## Acceptance Criteria

- Non-numeric value_field inputs cause a crash or TransformResult.error (no silent skipping).
- Aggregates reflect the actual batch or the row is quarantined via on_error.

## Tests

- Suggested tests to run: pytest tests/plugins/transforms/test_batch_stats.py
- New tests required: yes, wrong-type handling and error routing.

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: docs/contracts/plugin-protocol.md
