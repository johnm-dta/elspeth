# Bug Report: Aggregation output_mode=single terminates pipeline early

## Summary

- Aggregation flush in `output_mode="single"` returns a COMPLETED result immediately, so the aggregated row never traverses downstream transforms or config gates. This breaks pipeline ordering and drops intended processing for aggregated outputs.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: Codex
- Date: 2026-01-21
- Related run/issue ID: N/A

## Environment

- Commit/branch: ae2c0e6 / fix/rc1-bug-burndown-session-2
- OS: Linux
- Python version: Python 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: Deep dive src/elspeth/engine/processor.py for bugs; create reports.
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, no escalations
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: Code inspection only

## Steps To Reproduce

1. Configure a batch-aware transform with aggregation `output_mode: single`.
2. Add another transform after it (or config gates) that should process the aggregated row.
3. Run a pipeline where the aggregation flushes.

## Expected Behavior

- The aggregated output row continues through remaining transforms and config gates.

## Actual Behavior

- The aggregated output is marked COMPLETED and returned immediately; downstream transforms/config gates are never executed for that row.

## Evidence

- `src/elspeth/engine/processor.py:224-246` returns a COMPLETED RowResult for `output_mode == "single"` with no work item for downstream processing.
- Pipeline order specifies transforms then config gates (e.g., `docs/design/subsystems/00-overview.md`).

## Impact

- User-facing impact: Missing downstream processing for aggregated outputs.
- Data integrity / security impact: Audit trail shows completed outcomes without required transform/gate node states.
- Performance or cost impact: Silent logic omission rather than explicit failure.

## Root Cause Hypothesis

- `_process_batch_aggregation_node()` treats `output_mode="single"` as terminal and never enqueues the aggregated token for remaining steps.

## Proposed Fix

- Code changes (modules/files): `src/elspeth/engine/processor.py`
- Config or schema changes: None
- Tests to add/update: Add a test asserting aggregated single output flows through downstream transforms/config gates.
- Risks or migration steps: None; behavior should align with pipeline order.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `docs/design/subsystems/00-overview.md` (pipeline order); `docs/plans/completed/plugin-refactor/2026-01-18-wp06-aggregation-triggers.md` (output_mode semantics).
- Observed divergence: Aggregated row is treated as terminal instead of continuing.
- Reason (if known): Missing continuation logic for single output mode.
- Alignment plan or decision needed: Decide whether single output should continue (expected) or enforce terminal-only aggregation nodes.

## Acceptance Criteria

- Aggregation `output_mode=single` outputs are processed by downstream transforms and config gates.
- Tests confirm correct node_state sequence after a flush.

## Tests

- Suggested tests to run: `pytest tests/engine/test_processor.py -k aggregation_single`
- New tests required: Yes (single output continuation).

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `docs/design/subsystems/00-overview.md`
