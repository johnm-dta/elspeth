# Barrier Machinery: Aggregation and Coalesce Are Structural Twins

**Date:** 2026-06-11
**Audience:** anyone fixing a bug in, or extending, aggregation or coalesce
**Source:** the fork/coalesce architectural misfit assessment,
`notes/fork-coalesce-architecture-assessment-2026-06-10.md` (finding F3)

## The twin structure

ELSPETH has two barrier machineries: **aggregation** (buffer tokens at a
node until a trigger fires, then flush the batch through a batch-aware
transform) and **coalesce** (hold fork-sibling tokens of one `row_id` until
the merge policy is satisfied, then emit one merged token). Both were born
in the same change on 2026-01-12 and were split into separate files on
2026-02-12. They have never been re-unified: each carries its own complete,
parallel implementation of the same barrier shape — in-memory pending
state, checkpoint-state contract, restore path, post-completion arrival
handling, failure arm, and counter bookkeeping.

The practical consequence is that **every barrier-class fix must be made
twice**, and the two halves drift when it is made once. A current example:
`AggregationExecutor.restore_from_checkpoint` rebuilds restored tokens with
the default `resume_attempt_offset=0`, while the incomplete-token resume
path learned to derive `max_attempt + 1` (`processor.py:2323`) — see the
worked example below. The assessment's bug-family evidence includes
elspeth-262911c26b (that drift), elspeth-7294de558e (`rows_coalesce_failed`
has no audit re-derivation arm), elspeth-e1dd5e1303 (`rows_buffered`
live-vs-derive off-by-one) and elspeth-ce3adfb7b7 (resume-only FAILED
counters).

This document is the near-term mitigation: a map of the paired surfaces and
a checklist, so a change to one half is checked against the other half by
default. It is not a substitute for the structural fixes (see the forward
note at the end).

## Paired-surfaces table

Touching a surface in the left column? Check its twin in the right column
before you ship. File paths are relative to `src/elspeth/`.

| Concern | Aggregation | Coalesce |
|---|---|---|
| Executor | `engine/executors/aggregation.py` — `AggregationExecutor` | `engine/coalesce_executor.py` — `CoalesceExecutor` |
| In-memory pending state | `_AggregationNodeState` (buffers, tokens, `batch_id`, `member_count`, `accepted_count_total`, trigger) | `_PendingCoalesce` / `_BranchEntry` |
| Buffer/accept entry point | `AggregationExecutor.buffer_row` | `CoalesceExecutor.accept` |
| Release decision | `should_flush` / `check_flush_status` (trigger evaluator) | `_should_merge` (merge policy / quorum) |
| Release execution | `execute_flush` | `_execute_merge` (driven from `accept`) and `_resolve_pending` |
| Time-based release state | trigger timing offsets restored in `restore_from_checkpoint` (`elapsed_age_seconds`, `count_fire_offset`, `condition_fire_offset`) | `check_timeouts` |
| End-of-source drain | end-of-source flush via `execute_flush` (orchestrator-driven; post-flush assertion in `finalize_source_iteration`, `engine/orchestrator/source_iteration.py`) | `flush_pending` |
| Checkpoint-state contract | `contracts/aggregation_checkpoint.py` — `AggregationTokenCheckpoint`, `AggregationNodeCheckpoint`, `AggregationCheckpointState` | `contracts/coalesce_checkpoint.py` — `CoalesceTokenCheckpoint`, `CoalescePendingCheckpoint`, `CoalesceCheckpointState` |
| Checkpoint write | `AggregationExecutor.get_checkpoint_state` | `CoalesceExecutor.get_checkpoint_state` |
| Checkpoint restore | `AggregationExecutor.restore_from_checkpoint` (plus `restore_state`, `get_restored_state`, `restore_batch`) | `CoalesceExecutor.restore_from_checkpoint` |
| Scheduler-journal re-derivation on resume | `engine/processor.py` — `_restore_scheduler_blocks_from_aggregation_checkpoint` (currently `:514-554`) | `engine/processor.py` — `_restore_scheduler_blocks_from_coalesce_checkpoint` (currently `:556-586`) |
| Post-completion arrival handling | `_reconcile_checkpoint_batch_members` (audit trail advanced beyond the restored checkpoint) | late-arrival rejection in `accept`, backed by `_completed_keys`, `_mark_completed`, `_reconstruct_completed_keys_from_landscape`, `_check_landscape_for_completion` |
| Failure arm | flush failure paths inside `execute_flush` | `_fail_pending`, `notify_branch_lost`, `_evaluate_after_loss` |
| Counter bookkeeping | `rows_buffered` — incremented in `engine/orchestrator/outcomes.py` (the `(None, BUFFERED)` arm), re-derived in `engine/orchestrator/run_status.py` | `rows_coalesce_failed` — incremented in `engine/orchestrator/outcomes.py`, grafted on resume in `engine/orchestrator/resume.py` |

Line numbers are given only where the method name alone is not enough;
verify them against HEAD — the methods are the stable handles, the lines
are not.

One pair the assessment named is already gone at this HEAD: the
checkpoint anchor-fallback chain was deleted with the vestigial
token-anchor (finding F2), so it no longer appears above.

## Paired-change checklist (barrier-class bugs)

1. Find your surface in the table and read **both** halves before changing
   either. Assume the bug exists on both sides until you have evidence
   otherwise.
2. If the fix applies to both halves, land both in the same commit. If you
   cannot, file the twin-half issue immediately and link it as a dependency
   — do not leave the second half undiscovered.
3. If the fix genuinely applies to one half only, say why in the commit
   message or ticket. "The other side does it differently" is a drift
   finding, not a reason to skip it.
4. Changing the shape of buffered state? That is four sites, not one: both
   checkpoint contracts **and** both `_restore_scheduler_blocks_from_*`
   methods in `processor.py` (the resume path manufactures BLOCKED
   scheduler rows from blob contents).
5. Attempt-offset discipline: any path that re-drives a restored token must
   derive `resume_attempt_offset` from the existing `node_states` rows
   (`max_attempt + 1`, as at `processor.py:2323`), never the default `0`.
6. Touching a counter (`rows_buffered`, `rows_coalesce_failed`,
   `rows_aggregated` family)? Update both the live increment **and** the
   resume/derive arm (`run_status.py` re-derivation, `resume.py` graft), or
   the two bookkeepers will disagree after a resume.
7. Add or extend tests on both sides, including a resume-path test if the
   change touches checkpoint or restore code.

## Worked example: elspeth-262911c26b (a drift bug)

`AggregationExecutor.restore_from_checkpoint`
(`engine/executors/aggregation.py:684`) rebuilds `TokenInfo` objects with
the default `resume_attempt_offset=0`. But a crashed run has already
written `node_states` rows for those tokens at attempt 0, so the resumed
flush re-opens attempt 0 and fails with a UNIQUE constraint violation on
`(token_id, step_index, attempt)`. The incomplete-token resume path
already learned this lesson: `resume_incomplete_token` sets
`resume_attempt_offset=spec.max_attempt + 1` (`processor.py:2323`). The
fix existed; it was applied to one barrier path and not the other. That is
the twin-drift failure mode this document exists to prevent, and item 5 of
the checklist is its generalisation.

## Schema-merge duplication (resolved alongside this document)

The assessment also named three parallel schema-merge implementations
(elspeth-2188b142f2): the old `SchemaContract.merge` (AND-only
required-field semantics, which the coalesce executor deliberately routed
around for typed union merges), `merge_union_fields` in
`core/dag/coalesce_merge.py` (the correct, policy-aware implementation),
and a test simulation. That duplication is collapsed in the same
change-set that introduces this document: the one canonical union-merge
algorithm now lives in `contracts/union_merge.py`
(`merge_union_field_flags`), consumed by two thin wrappers —
`merge_union_fields` (build-time, `core/dag/coalesce_merge.py`) and
`merge_union_contracts` (runtime) — so build-time and runtime coalesce
merges cannot diverge. The batch-merge surviving on `SchemaContract`
(renamed `merge_for_batch`; used by the sink executor to combine
sibling-token contracts within a batch) is **intentionally** separate —
it implements different sibling-token semantics, not a leftover
duplicate.

## Forward note

This map is expected to shrink. Finding F1 of the assessment
(journal-as-truth durability unification — a buffered token becomes a
durable BLOCKED scheduler row, and the checkpoint blob shrinks to scalar
barrier metadata) removes most of the checkpoint/restore rows above, and
the longer-term Barrier abstraction (F3-long-term) would collapse the
remaining executor pairs into one machinery. Both are separate, sequenced
efforts — see
`notes/fork-coalesce-architecture-assessment-2026-06-10.md` for the
recommended ordering (F1 lands first; never F1 and the Barrier
abstraction together). Until then, the table and checklist above are the
contract.

## Related

- [ADR-028](adr/028-queue-vs-coalesce-not-duplicates.md) — QUEUE and
  COALESCE are *not* twins; do not unify that pair when the Barrier
  abstraction work happens.
- [ADR-026](adr/026-durable-token-scheduler.md) — the durable token
  scheduler whose journal F1 promotes to the single source of truth.
- `notes/fork-coalesce-architecture-assessment-2026-06-10.md` — the
  assessment this document implements the near-term recommendation of.
