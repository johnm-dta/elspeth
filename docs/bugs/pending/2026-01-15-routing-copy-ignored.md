# Bug Report: Routing mode copy is ignored (route stops pipeline)

## Summary

- “Route + continue” semantics (COPY routing to a sink without terminating the main path) are not supported in config gates and may be unreachable in config-only pipelines. If COPY-to-sink routing is added as a config feature, the current processing model must ensure COPY does not stop downstream execution.

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

1. Configure a pipeline that can express “route to sink AND continue” for the same token (COPY semantics). (Config gates currently always use MOVE for sink routes.)
2. Run the pipeline and check both the routed sink and the default output sink for the same row/token.

## Expected Behavior

- The row is written to the `flagged` sink and continues to downstream transforms/default sink (copy semantics).

## Actual Behavior

- In config-gate-only pipelines, COPY-to-sink routing is not expressible; sink routing is MOVE/terminal by construction.
- If a COPY-to-sink feature is introduced, ensure it does not terminate the main path (i.e., avoid treating COPY like MOVE).

## Evidence

- Config gate sink routing is always MOVE (no COPY option): `src/elspeth/engine/executors.py:588-600`
- Processor treats any sink route as terminal for that token: `src/elspeth/engine/processor.py:460-470`
- Architecture defines COPY as “route and continue”: `docs/design/architecture.md:138-143`

## Impact

- User-facing impact: workflows relying on “route + continue” lose data in the main path.
- Data integrity / security impact: audit trail implies routing decisions but skips expected downstream transforms.
- Performance or cost impact: retries/reprocessing may be needed to recover missing outputs.

## Root Cause Hypothesis

- The config gate model does not provide a way to express COPY-to-sink routing, and the token-processing loop treats sink routing as terminal for that token. Adding COPY-to-sink requires explicit support in both executor outcomes and the processor/orchestrator sink buffering.

## Proposed Fix

- Code changes (modules/files):
  - Decide whether COPY-to-sink routing is a supported requirement for config gates. If yes:
    - Extend config gate routing config to express MOVE vs COPY for sink routes.
    - Extend gate outcome to signal “route + continue” and update `RowProcessor`/`Orchestrator` to enqueue sink output while continuing downstream processing.
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

- If COPY-to-sink routing is supported: the row is written to the routed sink and continues down the pipeline.
- If config-only gates are the only gate mechanism: explicitly document that sink routing is terminal (MOVE) and close this report as not applicable.
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
