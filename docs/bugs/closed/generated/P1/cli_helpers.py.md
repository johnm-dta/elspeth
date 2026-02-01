# Bug Report: Aggregation accepts non-batch-aware transform and silently skips batching

## Summary

- Aggregation plugins are instantiated without verifying `is_batch_aware`, so a non-batch-aware transform runs per-row and aggregation triggers are silently ignored.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: Codex
- Date: 2026-01-25
- Related run/issue ID: N/A

## Environment

- Commit/branch: fix/rc1-bug-burndown-session-4 @ 86357898ee109a1dbb8d60f3dc687983fa22c1f0
- OS: Unknown
- Python version: Unknown
- Config profile / env vars: N/A
- Data set or fixture: Minimal aggregation config using a non-batch-aware transform (e.g., `passthrough`).

## Agent Context (if relevant)

- Goal or task prompt: You are a static analysis agent doing a deep bug audit. Target file: /home/john/elspeth-rapid/src/elspeth/cli_helpers.py
- Model/version: Codex (GPT-5)
- Tooling and permissions (sandbox/approvals): read-only sandbox
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code review only

## Steps To Reproduce

1. Create a settings YAML with an aggregation using a non-batch-aware transform (e.g., `plugin: passthrough`) and a trigger (`count: 2`).
2. Run `elspeth validate -s settings.yaml` or `elspeth run -s settings.yaml --execute`.
3. Observe that validation/execution proceeds without error and rows are processed per-row (no batching).

## Expected Behavior

- Configuration should fail early with a clear error that aggregation transforms must be `is_batch_aware=True`.

## Actual Behavior

- Aggregation config is accepted, but `RowProcessor` only buffers when `is_batch_aware` is true, so batching never occurs and aggregation settings are silently ignored.

## Evidence

- `src/elspeth/cli_helpers.py:42` instantiates aggregation transforms without checking `is_batch_aware`.
- `src/elspeth/engine/processor.py:724` only treats aggregation nodes as batch processing when `transform.is_batch_aware` is true.
- `docs/contracts/plugin-protocol.md:1151` specifies that aggregation requires a batch-aware transform.

## Impact

- User-facing impact: Aggregations configured with non-batch-aware transforms run per-row without warning, producing incorrect pipeline semantics.
- Data integrity / security impact: Audit trail misses expected batch records for configured aggregation nodes, undermining traceability expectations.
- Performance or cost impact: Potentially higher cost or latency if batching was intended but never used.

## Root Cause Hypothesis

- `instantiate_plugins_from_config` does not validate that aggregation plugins are batch-aware, allowing incompatible transforms to pass validation.

## Proposed Fix

- Code changes (modules/files):
  - Add a validation check in `src/elspeth/cli_helpers.py` that raises a `ValueError` when `agg_config.plugin` resolves to a transform with `is_batch_aware=False`.
- Config or schema changes: None.
- Tests to add/update:
  - Add a test in `tests/cli/test_cli_helpers.py` that asserts `instantiate_plugins_from_config` raises when an aggregation uses a non-batch-aware transform.
- Risks or migration steps:
  - Existing misconfigured pipelines will fail early; this is intended to prevent silent misbehavior.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `docs/contracts/plugin-protocol.md:1151`
- Observed divergence: Aggregation configs are not enforced to use batch-aware transforms.
- Reason (if known): Missing validation in CLI helper before graph construction.
- Alignment plan or decision needed: Enforce batch-aware requirement during plugin instantiation for aggregations.

## Acceptance Criteria

- A pipeline config with aggregation referencing a non-batch-aware transform fails validation with a clear error message.
- Batch-aware aggregation transforms continue to validate and run unchanged.
- New test covering this validation passes.

## Tests

- Suggested tests to run: `pytest tests/cli/test_cli_helpers.py`
- New tests required: yes, add coverage for non-batch-aware aggregation plugin rejection.

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `docs/contracts/plugin-protocol.md`
