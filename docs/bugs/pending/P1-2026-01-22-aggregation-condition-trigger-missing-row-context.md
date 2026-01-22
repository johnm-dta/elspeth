# Bug Report: Condition triggers cannot access row data

## Summary

- Aggregation condition triggers are evaluated against a context containing only `batch_count` and `batch_age_seconds`. Expressions that reference actual row fields (e.g., `row['type']`) raise `KeyError` and crash trigger evaluation, despite docs/config examples showing row-based conditions.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: codex
- Date: 2026-01-22
- Related run/issue ID: N/A

## Environment

- Commit/branch: `fix/rc1-bug-burndown-session-2` @ `ae2c0e6f088f467276582fa8016f91b4d3bb26c7`
- OS: Linux 6.8.0-90-generic
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: deep dive into `src/elspeth/engine/triggers.py` and file bugs
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code inspection of trigger evaluation and config docs

## Steps To Reproduce

1. Configure an aggregation trigger with `condition: "row['type'] == 'flush_signal'"` (as shown in docs/config examples).
2. Run the pipeline with a row that includes `{"type": "flush_signal"}`.
3. Observe a `KeyError: 'type'` from trigger evaluation, causing the aggregation to crash or never flush.

## Expected Behavior

- Condition triggers should evaluate against the current row (and any batch metadata) and fire when the row matches the expression.

## Actual Behavior

- Condition evaluation is executed against a context containing only `batch_count` and `batch_age_seconds`, so row field access raises `KeyError`.

## Evidence

- Trigger context lacks row data: `src/elspeth/engine/triggers.py:106-113`
- Doc example uses row fields: `docs/contracts/plugin-protocol.md:1169`
- Config tests allow row-based conditions: `tests/core/test_config_aggregation.py:50-60`

## Impact

- User-facing impact: condition-based batching using row signals is unusable.
- Data integrity / security impact: aggregation batches may never flush or crash mid-run.
- Performance or cost impact: buffered rows can accumulate indefinitely or crash runs.

## Root Cause Hypothesis

- `TriggerEvaluator` does not receive row data, and `should_trigger()` builds a context with only batch stats.

## Proposed Fix

- Code changes (modules/files):
  - Extend `TriggerEvaluator` to accept row context (e.g., `record_accept(row)` or `should_trigger(row)`).
  - Build evaluation context as merged row data + reserved batch keys.
  - Update `AggregationExecutor.buffer_row()` to pass row context.
- Config or schema changes:
  - Document reserved keys (`batch_count`, `batch_age_seconds`) and collision behavior.
- Tests to add/update:
  - Add trigger evaluation test with a row-based condition (e.g., `row['type'] == 'flush_signal'`).
- Risks or migration steps:
  - Potential collisions if row data already has `batch_count` keys; define precedence.

## Architectural Deviations

- Spec or doc reference: `docs/contracts/plugin-protocol.md:1209-1211`
- Observed divergence: condition trigger is documented as "Row matches expression" but row data is not provided.
- Reason (if known): evaluator was implemented with batch-only context.
- Alignment plan or decision needed: confirm whether condition triggers are row-scoped or batch-scoped.

## Acceptance Criteria

- Condition triggers can access row fields and fire without raising `KeyError`.

## Tests

- Suggested tests to run: `pytest tests/engine/test_triggers.py -k condition`
- New tests required: yes

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `docs/contracts/plugin-protocol.md`
