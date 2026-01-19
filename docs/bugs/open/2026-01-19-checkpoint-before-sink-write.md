# Bug Report: Checkpoints are created before sink writes (recovery can skip unwritten outputs)

## Summary

- `Orchestrator` creates checkpoints for `RowOutcome.COMPLETED` and `RowOutcome.ROUTED` **before** writing tokens to sinks. If the process crashes between checkpoint creation and sink write, recovery can treat those rows as already processed and skip sink output, causing missing artifacts.
- `RunResult` counters (e.g., `rows_succeeded`, `rows_routed`) are incremented before sink write succeeds, so they can report success for rows whose sink write ultimately failed.

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
- Config profile / env vars: checkpointing enabled
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: deep dive into system 5 (engine) and look for bugs
- Model/version: GPT-5.2 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code inspection of `orchestrator.py` and checkpoint recovery code

## Steps To Reproduce

1. Enable checkpoints with `frequency: every_row` (or a small `every_n` interval).
2. Use a sink that is not idempotent (or where missing writes are user-visible).
3. Run a pipeline and force a crash **after** `Orchestrator._maybe_checkpoint()` is called for a row but **before** `SinkExecutor.write()` completes (e.g., raise inside `sink.write` after some rows are processed).
4. Resume via `RecoveryManager.get_resume_point()` + `Orchestrator.resume()` (or any future resume path).

## Expected Behavior

- A checkpoint should represent a **durable row boundary**, including sink write completion (or recovery must guarantee sink writes for checkpointed rows are re-attempted).
- Counters should not report success for rows that never reached a sink.

## Actual Behavior

- Checkpoints are created before sink writes. A crash between checkpoint creation and sink write can cause recovery to skip rows whose sink output never happened.
- `rows_succeeded`/`rows_routed` can be incremented even if sink write fails afterward.

## Evidence

- Checkpoint creation happens during result handling, before sink writes:
  - `src/elspeth/engine/orchestrator.py:637` (counts)
  - `src/elspeth/engine/orchestrator.py:647` (calls `_maybe_checkpoint` for COMPLETED)
  - `src/elspeth/engine/orchestrator.py:658` (calls `_maybe_checkpoint` for ROUTED)
- Sink writes happen later, after the source loop finishes:
  - `src/elspeth/engine/orchestrator.py:676` (constructs `SinkExecutor`)
  - `src/elspeth/engine/orchestrator.py:681` (iterates `pending_tokens`)
  - `src/elspeth/engine/orchestrator.py:684` (`sink_executor.write(...)`)
- Recovery skips rows with `row_index > checkpointed_row_index` based on the checkpoint token’s source row:
  - `src/elspeth/core/checkpoint/recovery.py:154`
  - `src/elspeth/core/checkpoint/recovery.py:176`
  - `src/elspeth/core/checkpoint/recovery.py:200`

## Impact

- User-facing impact: missing output artifacts when a crash occurs in the “checkpointed but not yet written” window.
- Data integrity / security impact: audit trail may indicate progress checkpoints that do not correspond to durable sink output; recovery can create “false completeness.”
- Performance or cost impact: reprocessing might be required to recover missing outputs; idempotent sinks may duplicate output.

## Root Cause Hypothesis

- Checkpointing is currently tied to “transform/gate processing completed” rather than “sink output completed,” but recovery derives row completion from checkpoints.

## Proposed Fix

- Code changes (modules/files):
  - Move `_maybe_checkpoint(...)` to occur **after** the token is successfully written to its terminal sink, using the sink node as `node_id` (or at least after successful `SinkExecutor.write` returns).
  - Alternatively: persist a checkpoint state that distinguishes “processed-but-not-sunk” vs “sunk” and ensure recovery replays sink writes for any checkpointed-but-not-sunk rows.
- Config or schema changes:
  - Consider exposing “checkpoint_at: sink|pre_sink” with clear semantics. Default should be sink for durability.
- Tests to add/update:
  - Add an orchestrator recovery test that simulates a crash after checkpoint creation but before sink write, then verifies recovery replays the missing sink write and does not skip that row.
- Risks or migration steps:
  - If checkpoint location changes, ensure `RecoveryManager.get_unprocessed_rows` and any future per-token checkpoint semantics remain correct for forks/expansions.

## Architectural Deviations

- Spec or doc reference: `CLAUDE.md` (“audit trail must withstand formal inquiry”; “no inference - if it’s not recorded, it didn’t happen”)
- Observed divergence: checkpoints imply a durable progress point, but sink output may not exist for that checkpointed row.
- Alignment plan or decision needed: define checkpoint durability boundary explicitly (pre-sink vs post-sink).

## Acceptance Criteria

- A crash after checkpoint creation but before sink write does not result in missing outputs after recovery.
- Checkpoint semantics are documented and enforced consistently.

## Tests

- Suggested tests to run:
  - `pytest tests/engine/test_orchestrator_recovery.py`
  - `pytest tests/engine/test_orchestrator.py`
- New tests required: yes

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `docs/design/architecture.md` (checkpointing + audit durability)
