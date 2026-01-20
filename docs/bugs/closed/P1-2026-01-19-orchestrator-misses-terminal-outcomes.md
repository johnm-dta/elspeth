# Bug Report: Orchestrator ignores RowOutcome.EXPANDED/BUFFERED/COALESCED (silent drops and incorrect counters)

## Summary

- `RowProcessor` can return additional outcomes (`EXPANDED`, `BUFFERED`, `COALESCED`) but `Orchestrator`’s result loop does not handle them and does not fail fast for unknown outcomes.
- This creates two failure modes:
  - Silent outcome drops (no counters, no routing, no checkpoints) when an unhandled terminal outcome occurs.
  - Incorrect bookkeeping and recovery semantics when non-terminal outcomes (BUFFERED) are ignored without explicit handling.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: codex
- Date: 2026-01-19
- Related run/issue ID: N/A

## Environment

- Commit/branch: `8cfebea78be241825dd7487fed3773d89f2d7079` (main)
- OS: Linux (kernel 6.8.0-90-generic)
- Python version: 3.13.1
- Config profile / env vars: pipelines using expansion, aggregation passthrough, or coalesce
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: deep dive into system 5 (engine) and look for bugs
- Model/version: GPT-5.2 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code inspection of `RowOutcome`, `RowProcessor`, `Orchestrator`

## Steps To Reproduce

1. Use a transform that returns `TransformResult.success_multi(...)` (deaggregation) so the parent token gets `RowOutcome.EXPANDED`.
2. Use passthrough aggregation mode to produce `RowOutcome.BUFFERED` for non-flushing rows.
3. Enable coalesce integration (or otherwise cause `RowOutcome.COALESCED` to be returned).
4. Run a pipeline and inspect orchestrator counters, checkpoint behavior, and sink outputs.

## Expected Behavior

- `Orchestrator` should explicitly handle every `RowOutcome`:
  - terminal outcomes are accounted for and never silently ignored
  - non-terminal outcomes are handled intentionally (e.g., BUFFERED is neither success nor failure)
  - unknown outcomes hard-fail to prevent silent data loss

## Actual Behavior

- `Orchestrator` handles only a subset of `RowOutcome` values and silently ignores others.

## Evidence

- Outcomes include `COALESCED`, `EXPANDED`, and `BUFFERED`:
  - `src/elspeth/contracts/enums.py:148`
  - `src/elspeth/contracts/enums.py:169`
  - `src/elspeth/contracts/enums.py:173`
- `RowProcessor` can emit:
  - `BUFFERED`: `src/elspeth/engine/processor.py:355`
  - `EXPANDED`: `src/elspeth/engine/processor.py:697`
  - `COALESCED`: `src/elspeth/engine/processor.py:802`
- `Orchestrator` result handling omits these and has no `else: raise`:
  - `src/elspeth/engine/orchestrator.py:637`
  - `src/elspeth/engine/orchestrator.py:672`

## Impact

- User-facing impact: runs may report misleading counters; certain pipeline features can appear to “do nothing” if outcomes are ignored.
- Data integrity / security impact: violates “no silent drops” expectations; unhandled terminal outcomes can disappear from orchestration accounting.
- Performance or cost impact: could trigger unnecessary reruns due to missing outputs or confusing telemetry.

## Root Cause Hypothesis

- Orchestrator bookkeeping was implemented around a limited set of outcomes and not updated as new outcome types (expand/coalesce/buffer) were added.

## Proposed Fix

- Code changes (modules/files):
  - Add explicit handling for `RowOutcome.EXPANDED` (non-sink terminal for parent token; should affect counters/checkpoints deterministically).
  - Add explicit handling for `RowOutcome.BUFFERED` (non-terminal; should not be counted as success, but may need tracking for end-of-source flush/recovery).
  - Handle `RowOutcome.COALESCED` by enqueuing the merged token to the appropriate sink and ensuring consumed tokens are not double-written.
  - Add a final `else: raise RuntimeError(...)` for any unknown `RowOutcome` to prevent silent drops.
- Config or schema changes: none.
- Tests to add/update:
  - Add an orchestrator integration test that exercises expansion and verifies it does not silently ignore `EXPANDED`.
  - Add a test that intentionally returns an unknown outcome and asserts orchestrator fails fast.
- Risks or migration steps:
  - Ensure counters are clearly defined (rows vs tokens) and stable across forks/expansion/coalesce.

## Architectural Deviations

- Spec or doc reference: `CLAUDE.md` (“No inference - if it’s not recorded, it didn’t happen”)
- Observed divergence: terminal outcomes can be ignored silently at orchestration level.
- Alignment plan or decision needed: define orchestrator’s responsibilities for each `RowOutcome` and encode them as exhaustive handling + tests.

## Acceptance Criteria

- Orchestrator handles all `RowOutcome` values explicitly and fails fast on unknown values.
- No token terminal outcome is silently dropped from orchestration accounting.

## Tests

- Suggested tests to run:
  - `pytest tests/engine/test_orchestrator.py`
  - `pytest tests/engine/test_processor.py -k deaggregation`
- New tests required: yes

## Notes / Links

- Related issues/PRs: `docs/bugs/open/2026-01-19-coalesce-config-ignored.md` (coalesce integration gaps)
- Related design docs: `docs/design/architecture.md`
