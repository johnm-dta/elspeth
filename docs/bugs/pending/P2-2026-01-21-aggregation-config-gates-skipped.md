# Bug Report: Aggregation outputs skip config gates when aggregation is last transform

## Summary

- When an aggregation node is the last transform, passthrough/transform flush results are marked COMPLETED and returned immediately. Config gates (which should run after transforms) are never executed for those outputs.

## Severity

- Severity: major
- Priority: P2

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

1. Configure a batch-aware transform as the last transform in the pipeline with `output_mode: passthrough` or `transform`.
2. Define one or more config-driven gates (`config.gates`).
3. Trigger a batch flush.

## Expected Behavior

- Outputs from the aggregation should run through config gates (pipeline order: transforms → config gates → sinks).

## Actual Behavior

- Aggregation outputs are marked COMPLETED and sent to sinks directly; config gate node_states are missing.

## Evidence

- Passthrough flush returns COMPLETED outputs when `more_transforms` is false: `src/elspeth/engine/processor.py:267-310`.
- Transform-mode flush returns COMPLETED outputs when `more_transforms` is false: `src/elspeth/engine/processor.py:349-379`.
- Config gates are only processed later in `_process_single_token`: `src/elspeth/engine/processor.py:874-954`. These are bypassed because `_process_batch_aggregation_node()` returns early.

## Impact

- User-facing impact: Config gate logic is silently skipped for aggregated outputs.
- Data integrity / security impact: Missing node_state records for config gates; routing decisions not applied.
- Performance or cost impact: Incorrect routing or sink selection.

## Root Cause Hypothesis

- `_process_batch_aggregation_node()` decides continuation based solely on remaining transforms (`more_transforms`) and doesn’t account for config gates.

## Proposed Fix

- Code changes (modules/files): `src/elspeth/engine/processor.py`
- Config or schema changes: None
- Tests to add/update: Add tests ensuring config gates run after aggregation flushes when aggregation is last transform.
- Risks or migration steps: None.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): Pipeline order described in `docs/design/subsystems/00-overview.md`.
- Observed divergence: Config gates are skipped for certain aggregation outputs.
- Reason (if known): `more_transforms` gate doesn’t consider config gates.
- Alignment plan or decision needed: Account for config gates when deciding whether to enqueue aggregated outputs.

## Acceptance Criteria

- Aggregation outputs run through config gates even when aggregation is the last transform.

## Tests

- Suggested tests to run: `pytest tests/engine/test_processor.py -k aggregation_config_gates`
- New tests required: Yes (aggregation + config gate integration).

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `docs/design/subsystems/00-overview.md`
