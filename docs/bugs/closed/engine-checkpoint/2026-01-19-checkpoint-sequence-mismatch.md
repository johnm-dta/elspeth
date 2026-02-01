# Bug Report: Crash recovery misinterprets checkpoint `sequence_number` (row_index vs token/event counter)

## Summary

- `RecoveryManager.get_unprocessed_rows()` treats `checkpoints.sequence_number` as a source `row_index` cutoff, but `Orchestrator._maybe_checkpoint()` increments it as a generic monotonic counter per checkpoint call (per terminal token), so resume boundaries can skip or reprocess rows.

## Severity

- Severity: critical
- Priority: P0

## Reporter

- Name or handle: codex
- Date: 2026-01-19
- Related run/issue ID: N/A

## Environment

- Commit/branch: `main` @ `25468ac9550b481a55b81a05d84bbf2592e6430c`
- OS: Linux (Ubuntu 6.8.0-90-generic)
- Python version: 3.13.1
- Config profile / env vars: N/A (static analysis)
- Data set or fixture: N/A (static analysis)

## Agent Context (if relevant)

- Goal or task prompt: deep dive into subsystems, identify hotspots, write bug reports
- Model/version: GPT-5.2 (Codex CLI)
- Tooling and permissions (sandbox/approvals): sandbox read-only, network restricted
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: inspected orchestrator + checkpoint recovery modules and existing tests/docs

## Steps To Reproduce

1. Enable checkpoints (e.g., `frequency="every_row"` or `frequency="every_n"`).
2. Run a pipeline where a single source row can produce multiple terminal tokens (e.g., a gate that forks into multiple branches).
3. Ensure a checkpoint is written after processing that row.
4. Call `RecoveryManager.get_unprocessed_rows(run_id)` (or use `elspeth resume <run_id>` once resume execution is implemented).
5. Observe the cutoff is computed from `sequence_number`, which is not a source-row index.

Alternative deterministic repro (DB-only):

1. Create a run with rows indexed 0..N.
2. Insert a checkpoint with `sequence_number` larger than the last processed row index.
3. Call `get_unprocessed_rows()` and observe it skips rows incorrectly.

## Expected Behavior

- Resume boundaries are derived from a value semantically tied to source-row progress (or derived via checkpoint token → row mapping), so resume neither skips unprocessed rows nor reprocesses already-processed rows.

## Actual Behavior

- `get_unprocessed_rows()` uses `rows.row_index > checkpoint.sequence_number`, but `sequence_number` is a monotonic counter incremented in `_maybe_checkpoint()` per checkpoint call, which is invoked for terminal token results (not per source row).

## Evidence

- Orchestrator monotonic counter + increment: `src/elspeth/engine/orchestrator.py:117`, `src/elspeth/engine/orchestrator.py:138-156`
- `_maybe_checkpoint()` call sites (per terminal token result while iterating `results`, which can include fork children): `src/elspeth/engine/orchestrator.py:581-626`
- Schema comment: “Monotonic progress marker”: `src/elspeth/core/landscape/schema.py:315-324`
- Recovery treats `sequence_number` as `row_index` boundary: `src/elspeth/core/checkpoint/recovery.py:154-179`
- Tests encode the row-index assumption (write `sequence_number=<row_index>` and assert `row_index > sequence_number`): `tests/core/checkpoint/test_recovery.py:371`, `tests/integration/test_checkpoint_recovery.py:66`
- CLI resume currently surfaces recovery state (execution not yet implemented): `src/elspeth/cli.py:565`

## Impact

- User-facing impact: resume can skip rows (data loss) or reprocess rows (duplicate outputs), depending on forks/failures/batching.
- Data integrity / security impact: incorrect resume violates auditability (“what happened” queries can become wrong).
- Performance or cost impact: wasted recomputation or missed work.

## Root Cause Hypothesis

- `sequence_number` is overloaded: implemented as an event/token counter but consumed as a source-row progress marker.

## Proposed Fix

- Code changes (modules/files):
  - Option A (recommended): keep `sequence_number` as an ordering field, but derive the resume row boundary via DB joins `checkpoints.token_id → tokens.row_id → rows.row_index`, and use that `row_index` cutoff in recovery.
  - Option B: redefine `sequence_number` to mean “last processed row_index” and ensure the orchestrator writes it at most once per source row (even with forks).
  - Option C: store both explicitly (e.g., `event_sequence` and `source_row_index`) to prevent ambiguity.
- Config or schema changes:
  - If Option C: migrate the `checkpoints` schema.
- Tests to add/update:
  - Add a fork scenario where `sequence_number != row_index` and recovery still returns correct unprocessed rows.
  - Add failure/quarantine/batch scenarios where some rows do not produce checkpoints.
- Risks or migration steps:
  - Existing checkpoints may have old semantics; migration/compatibility needs to be explicit.

## Architectural Deviations

- Spec or doc reference: `docs/plans/completed/2026-01-12-phase5-production-hardening.md` (resume described as skipping from checkpoint sequence)
- Observed divergence: runtime uses `sequence_number` as an ordering counter; recovery/tests treat it as row progress.
- Reason (if known): checkpointing added as a monotonic marker; recovery later assumed row boundary.
- Alignment plan or decision needed: pick one semantic contract and enforce it across orchestrator, recovery, CLI, and tests.

## Acceptance Criteria

- Recovery boundary is correct for linear pipelines, forked pipelines (multiple tokens per row), and rows that fail/quarantine/aggregate.
- `get_unprocessed_rows()` returns exactly the rows that still require processing after the checkpoint.
- Tests cover fork and non-checkpointed row outcomes and pass.

## Tests

- Suggested tests to run: `pytest tests/core/checkpoint/test_recovery.py tests/integration/test_checkpoint_recovery.py`
- New tests required: yes (fork + non-checkpointed rows)

## Notes / Links

- Adjacent gap: `CheckpointSettings.frequency == "aggregation_only"` is documented but there is no checkpoint creation outside `Orchestrator._maybe_checkpoint()` (comment says “handled separately” but no implementation path exists).
