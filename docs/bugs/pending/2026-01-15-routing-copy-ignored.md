# Bug Report: Routing mode copy is ignored (route stops pipeline)

## Summary

- `RoutingAction.route(..., mode=copy)` is treated like `move`, so routed rows do not continue down the pipeline, violating routing semantics.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: codex
- Date: 2026-01-15
- Related run/issue ID: N/A

## Environment

- Commit/branch: not checked
- OS: not checked
- Python version: not checked
- Config profile / env vars: not checked
- Data set or fixture: not checked

## Agent Context (if relevant)

- Goal or task prompt: identify another bug and document it
- Model/version: GPT-5 (Codex)
- Tooling and permissions (sandbox/approvals): sandbox read-only, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: inspected `src/elspeth/engine/executors.py`, `src/elspeth/engine/processor.py`, `docs/design/architecture.md`

## Steps To Reproduce

1. Create a gate that returns `RoutingAction.route("flagged", mode=RoutingMode.COPY)`.
2. Configure a pipeline with that gate, a sink for `flagged`, and a default output sink.
3. Run the pipeline and check both sinks.

## Expected Behavior

- The row is written to the `flagged` sink and continues to downstream transforms/default sink (copy semantics).

## Actual Behavior

- The row is treated as routed and terminates early; downstream transforms/default sink do not receive it.

## Evidence

- `src/elspeth/engine/executors.py:324-347` sets `sink_name` for all route actions without checking `action.mode`.
- `src/elspeth/engine/processor.py:145-152` returns `outcome="routed"` as soon as `sink_name` is set, halting further processing.
- Architecture defines `copy` as “route and continue” (`docs/design/architecture.md:138-143`).

## Impact

- User-facing impact: workflows relying on “route + continue” lose data in the main path.
- Data integrity / security impact: audit trail implies routing decisions but skips expected downstream transforms.
- Performance or cost impact: retries/reprocessing may be needed to recover missing outputs.

## Root Cause Hypothesis

- Gate execution does not propagate routing mode; `RoutingMode.COPY` is effectively treated as `MOVE` because the executor only returns a terminal `sink_name` and the processor always stops on any routed outcome.

## Proposed Fix

- Code changes (modules/files):
  - Extend gate outcome to signal “route + continue” when `mode == COPY`.
  - Update `RowProcessor` to keep processing after recording a copy route, while still enqueueing/writing to the routed sink.
- Config or schema changes: none.
- Tests to add/update:
  - Add orchestrator-level test verifying that `RoutingMode.COPY` sends a row to the routed sink and continues to the output sink.
- Risks or migration steps:
  - Requires adjusting routing counters/checkpoint semantics to handle dual destinations.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `docs/design/architecture.md#L138`.
- Observed divergence: `copy` behaves like `move`, terminating the current path.
- Reason (if known): routing mode not propagated through gate outcome/processor logic.
- Alignment plan or decision needed: define how routing events are recorded for copy vs move and ensure both path outputs are produced.

## Acceptance Criteria

- When a gate returns `RoutingAction.route(..., mode=copy)`, the row is written to the routed sink and continues down the pipeline.
- Audit trail records routing events while downstream node states are still present.

## Tests

- Suggested tests to run: `.venv/bin/python -m pytest tests/engine/test_orchestrator.py`
- New tests required: yes, for routing copy semantics.

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `docs/design/architecture.md`

## Triage Note (2026-01-19)

**Status:** Kept in pending - needs investigation.

`RoutingMode.COPY` exists in the codebase (`dag.py:369,377`, `routing.py:95`) but it's unclear if the processor implements "route AND continue" semantics correctly.

**Needs investigation:**
1. Trace `RoutingMode.COPY` through gate execution to see if dual outputs are produced
2. Check if processor handles COPY differently from MOVE
3. Write test that verifies COPY results in output to both routed sink AND downstream processing
