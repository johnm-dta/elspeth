# Bug Report: Pre-run validation allows mixed dynamic/explicit schemas on pass-through nodes

## Summary

- Pre-run DAG validation treats dynamic schemas as compatible with explicit schemas for pass-through nodes (gate/coalesce), allowing mixed branches to pass validation. The effective schema then defaults to the first branch, which can be explicit, masking the dynamic branch. Validation should detect this mismatch pre-run and raise an error.

## Severity

- Severity: major
- Priority: P2

## Reporter

- Name or handle: Codex
- Date: 2026-02-01
- Related run/issue ID: N/A

## Environment

- Commit/branch: RC1-bugs-final
- OS: Linux
- Python version: Python 3.12+
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: File bug for mixed dynamic/explicit branch schema validation.
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, no escalations
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: Code inspection only

## Steps To Reproduce

1. Define a gate or coalesce with two incoming branches.
2. Ensure one branch produces a dynamic schema (`output_schema` is `None` or a dynamic schema) and the other produces an explicit `PluginSchema`.
3. Route the pass-through node into a downstream transform or sink with a strict input schema.
4. Run configuration validation (pre-run DAG validation).

## Expected Behavior

- Pre-run validation detects the mixed dynamic/explicit branch schemas and raises a `ValueError` indicating the mismatch and the offending branches.

## Actual Behavior

- Validation passes because dynamic schemas are treated as compatible with explicit schemas. The effective schema is taken from the first incoming edge, which can be explicit, hiding the dynamic branch. Downstream strict schema validation then fails at runtime when rows from the dynamic branch don’t conform.

## Evidence

- `src/elspeth/core/dag.py:1036-1056` selects `first_schema` as the effective schema for pass-through nodes, even when other branches are dynamic.
- `src/elspeth/core/dag.py:1072-1083` treats `None` (dynamic) as compatible with explicit schemas, so mixed branches do not raise.
- `src/elspeth/core/dag.py:1127-1145` coalesce validation reuses the same compatibility check, allowing mixed dynamic/explicit branches.

## Impact

- User-facing impact: Pipelines validate but fail at runtime on rows from dynamic branches.
- Data integrity / security impact: Pre-run validation gives a false sense of safety.
- Performance or cost impact: Wasted runs and retries due to preventable runtime failures.

## Root Cause Hypothesis

- Schema compatibility logic considers dynamic schemas universally compatible and pass-through schema selection always returns the first branch’s schema, masking mixed branch types.

## Proposed Fix

- Code changes (modules/files): `src/elspeth/core/dag.py`
  - Treat dynamic/explicit branch mixes as incompatible for pass-through nodes.
  - Raise a `ValueError` during validation when any incoming branch is dynamic and any other is explicit.
- Config or schema changes: None.
- Tests to add/update:
  - Add a validation test that constructs a gate/coalesce with one dynamic and one explicit branch and asserts a pre-run validation error.
- Risks or migration steps:
  - Some existing configs may start failing validation; they should be updated to make all branches explicit or all dynamic.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): N/A
- Observed divergence: Pre-run validation allows mismatched branch schemas, contrary to the expectation that schema validation prevents runtime mismatches.
- Reason (if known): Compatibility logic treats dynamic as universally compatible.
- Alignment plan or decision needed: Decide on a strict rule: mixed dynamic/explicit branches should be invalid at validation time.

## Acceptance Criteria

- Validation raises for mixed dynamic/explicit schemas on pass-through nodes with clear error messaging.
- Tests cover gate and coalesce cases.

## Tests

- Suggested tests to run: `pytest tests/core/test_dag_validation.py -k "dynamic"`.
- New tests required: Yes (mixed dynamic/explicit branch validation failure).

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: N/A
