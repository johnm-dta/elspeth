# Bug Report: Aggregation buffers are never flushed at end-of-source (end_of_source trigger not implemented)

## Summary

- Aggregation triggers document an implicit `end_of_source` behavior (“engine handles at source exhaustion”), but the engine never flushes aggregation buffers when the source iterator ends.
- Any aggregation configured with a count/timeout/condition trigger can leave a partially filled buffer at end-of-run; without an end-of-source flush, those rows never produce aggregation output, causing silent data loss.

## Severity

- Severity: critical
- Priority: P0

## Reporter

- Name or handle: codex
- Date: 2026-01-19
- Related run/issue ID: N/A

## Environment

- Commit/branch: `8cfebea78be241825dd7487fed3773d89f2d7079` (main)
- OS: Linux (kernel 6.8.0-90-generic)
- Python version: 3.13.1
- Config profile / env vars: aggregation configured (e.g., `trigger.count`)
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: deep dive into system 5 (engine) and look for bugs
- Model/version: GPT-5.2 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code inspection of aggregation trigger and processor/orchestrator flow

## Steps To Reproduce

1. Configure an aggregation with `trigger.count: 3` and a batch-aware transform.
2. Provide a source with only 1–2 rows (or any count not divisible by the trigger count).
3. Run the pipeline.

## Expected Behavior

- When the source is exhausted, the engine should flush any non-empty aggregation buffers and emit final outputs (or fail fast if partial batches are disallowed).
- The audit trail should record that the flush was triggered by end-of-source.

## Actual Behavior

- There is no end-of-source flush in `Orchestrator`; buffered rows remain buffered forever and never produce sink-bound outputs.

## Evidence

- Trigger docs explicitly mention end-of-source flush responsibility:
  - `src/elspeth/engine/triggers.py:14` (“end_of_source: Implicit - engine handles at source exhaustion”)
- Aggregation flushing only occurs when `should_flush()` returns True during per-row processing:
  - `src/elspeth/engine/processor.py:183` (checks `should_flush`)
  - `src/elspeth/engine/processor.py:185` (flushes buffer only in that path)
- Orchestrator ends the source loop and proceeds to sink writes without flushing aggregation buffers:
  - `src/elspeth/engine/orchestrator.py:586` (source loop)
  - `src/elspeth/engine/orchestrator.py:676` (sink writes begin after loop)

## Impact

- User-facing impact: aggregation outputs are missing for final partial batches; pipelines silently drop expected results.
- Data integrity / security impact: audit trail is incomplete; buffered tokens never reach a terminal state.
- Performance or cost impact: may require re-running pipelines or adjusting sources to “pad” batches.

## Root Cause Hypothesis

- The aggregation implementation is currently “flush-on-accept-only” and lacks a finalization step at source exhaustion to flush remaining buffers.

## Proposed Fix

- Code changes (modules/files):
  - Add an explicit end-of-source flush step in `Orchestrator._execute_run()` after the source iterator is exhausted and before sink writing:
    - For each aggregation node with non-empty buffer: flush, execute the batch-aware transform, and enqueue resulting tokens for sink writes.
  - Record the trigger reason as `TriggerType.END_OF_SOURCE` (or equivalent) in batch metadata/audit trail.
- Config or schema changes:
  - Decide policy for partial batches at end-of-source (always flush vs require full batch vs configurable).
- Tests to add/update:
  - Add orchestrator integration test: `trigger.count=3` with 2 input rows should still emit an aggregation output due to end-of-source flush.
- Risks or migration steps:
  - Ensure flush ordering is deterministic and consistent with checkpoint semantics and retry/recovery plans.

## Architectural Deviations

- Spec or doc reference: `src/elspeth/engine/triggers.py` end-of-source note; `CLAUDE.md` audit completeness invariants
- Observed divergence: engine does not flush at end-of-source.
- Alignment plan or decision needed: define and implement end-of-source semantics for all aggregation policies.

## Acceptance Criteria

- Any non-empty aggregation buffer is flushed at end-of-source and produces the expected outputs.
- Audit trail includes a clear trigger reason indicating end-of-source.

## Tests

- Suggested tests to run:
  - `pytest tests/engine/test_triggers.py`
  - `pytest tests/engine/test_processor.py -k batch`
  - `pytest tests/engine/test_orchestrator.py`
- New tests required: yes (orchestrator-level end-of-source flush)

## Notes / Links

- Related issues/PRs: `docs/bugs/open/2026-01-15-dag-fork-aggregation-drop.md` (aggregation outputs not propagated downstream)
- Related design docs: `docs/contracts/plugin-protocol.md` (trigger semantics)
