# Bug Report: Aggregation flush bypasses audit executors and never updates batch status (incomplete audit + broken recovery)

## Summary

- Batch-aware aggregation transforms are executed via direct `transform.process(buffered_rows, ctx)` calls, bypassing `TransformExecutor` audit recording (no node_state open/complete, no duration, no exception capture, no output hash recording).
- The engine creates `batches` and `batch_members`, but never transitions batches through `executing → completed/failed` via `LandscapeRecorder.update_batch_status(...)` / `complete_batch(...)`.
- Because batches remain in `draft` and no flush node_state is recorded, crash recovery logic that expects `executing`/`failed` batches is undermined, and audit trail for aggregation nodes is incomplete.

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
- Config profile / env vars: aggregations configured
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: deep dive into system 5 (engine) and look for bugs
- Model/version: GPT-5.2 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code inspection of aggregation executors and recorder batch APIs

## Steps To Reproduce

1. Configure an aggregation with `trigger.count` small enough to flush during a short run.
2. Run a pipeline that hits the aggregation flush path.
3. Inspect the Landscape DB:
   - `batches.status` remains `draft`, and
   - there is no node_state representing the aggregation flush operation (only batch_members insertions).

## Expected Behavior

- When rows are accepted into an aggregation, the audit trail should clearly show:
  - token acceptance into a batch (membership),
  - a flush execution node_state for the aggregation transform, and
  - batch status transitions and completion metadata (including trigger reason and the state_id for the flush).

## Actual Behavior

- Batch flush execution is not auditable as a node_state, and batches never leave `draft` status.

## Evidence

- Batch-aware transform flush is executed directly (no executor wrapper):
  - `src/elspeth/engine/processor.py:193` (`transform.process(buffered_rows, ctx)`)
- Engine-owned aggregation buffering records batch membership but no status transitions:
  - `src/elspeth/engine/executors.py:763` (creates batch)
  - `src/elspeth/engine/executors.py:780` (records `batch_members`)
  - No calls to `update_batch_status` or `complete_batch` exist in `src/elspeth/engine/` (search confirms).
- Recorder has explicit APIs for batch status transitions and completion:
  - `src/elspeth/core/landscape/recorder.py:1331` (`update_batch_status`)
  - `src/elspeth/core/landscape/recorder.py:1363` (`complete_batch`)
- Recovery expects incomplete batches to be in `executing`/`failed`:
  - `src/elspeth/engine/orchestrator.py:928` (recovery handling for `BatchStatus.EXECUTING` / `FAILED`)

## Impact

- User-facing impact: harder to explain/trace aggregation behavior; recovery may not correctly identify interrupted flushes.
- Data integrity / security impact: violates auditability standard for transform boundaries; aggregation is a transform boundary with missing input/output capture and status tracking.
- Performance or cost impact: retries and recovery are harder and may reprocess more than necessary.

## Root Cause Hypothesis

- Aggregation “structural cleanup” removed the old aggregation accept/flush executor path but did not replace it with a flush wrapper that records node_states and updates batch statuses.

## Proposed Fix

- Code changes (modules/files):
  - Introduce an explicit aggregation flush execution wrapper (either extend `TransformExecutor` to support list[dict] inputs when `transform.is_batch_aware`, or add a dedicated `AggregationFlushExecutor`).
  - On flush:
    - transition batch to `executing` with trigger reason + `aggregation_state_id` (node_state id),
    - execute the batch-aware transform with full audit recording,
    - transition batch to `completed`/`failed` based on result, and
    - reset engine buffers deterministically.
  - Ensure exceptions during flush mark batch as failed and are recorded in node_state error_json.
- Config or schema changes:
  - Ensure `TriggerType.END_OF_SOURCE` and other reasons are recorded in `batches.trigger_reason`.
- Tests to add/update:
  - Add a test that flush creates a node_state at the aggregation node and transitions batch status out of `draft`.
  - Add a recovery test ensuring interrupted flush batches are detectable and retryable.
- Risks or migration steps:
  - Requires a clear definition of aggregation flush semantics and how outputs map to tokens (single/passthrough/transform output modes).

## Architectural Deviations

- Spec or doc reference: `CLAUDE.md` (“Transform boundaries - Input AND output captured at every transform”)
- Observed divergence: aggregation flush lacks transform boundary capture and status tracking.
- Alignment plan or decision needed: specify aggregation node_state semantics for per-member accept vs flush execution.

## Acceptance Criteria

- Each aggregation flush creates a node_state at the aggregation node with correct hashes and error capture.
- Batches transition through expected statuses with recorded trigger reasons and flush state linkage.

## Tests

- Suggested tests to run:
  - `pytest tests/engine/test_processor.py -k batch`
  - `pytest tests/engine/test_orchestrator_recovery.py`
- New tests required: yes

## Notes / Links

- Related issues/PRs: `docs/bugs/open/2026-01-15-dag-fork-aggregation-drop.md`
- Related design docs: `docs/contracts/plugin-protocol.md`
