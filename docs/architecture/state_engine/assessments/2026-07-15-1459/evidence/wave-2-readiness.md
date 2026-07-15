# Wave 2 Readiness Evidence

Wave 2 reconnaissance was read-only. The tables below identify production
entries, strongest present evidence, and the next executable discriminator.

## Sink durability

### Sink production chain

```text
LeaderDrainCoordinator.execute
  -> SinkFlushCoordinator.flush_and_write_sinks
  -> SinkFlushCoordinator.write_pending_to_sinks
  -> SinkExecutor.write
  -> scheduler terminalization callback flush
  -> RowProcessor.mark_sink_bound_scheduler_terminal_many
  -> TokenSchedulerRepository.mark_pending_sink_terminal_many
  -> SchedulerDispositionRepository.mark_pending_sink_terminal_many
```

Sink-redrive work enters this chain after resume/recovery claims a prior
PENDING_SINK row. Outcome-witness repair runs before redrive claim and emits no
`RowResult` for the already completed sink.

Strongest current tests:

- `test_scheduler_events.py::test_pending_sink_claim_and_terminalization_record_transition_events`
- `test_scheduler_events.py::test_pending_sink_batch_terminalization_records_per_token_events`
- batch duplicate/missing/wrong-owner tests in `test_scheduler_events.py`
- strict-owner and stale-token cases in `test_leader_fence_stale_token.py`
- stale bulk close and repair in `test_suspended_winner_fences.py`
- `test_scheduler_events.py::test_pending_sink_with_terminal_outcome_is_repaired_without_reclaiming_sink`
- processor resume tests for repair and redrive without transform replay
- real CSV sink artifact/audit integration
- primary sink audit-atomicity and diversion-order tests
- full failsink/discard pipeline counter-parity cases

Next discriminators:

- real primary sink audit before TS-13 close;
- post-outcome callback crash and TS-14 resume without re-emission;
- full row/event rollback on TS-13 guard/event failure;
- no-outcome and event-failure TS-14 rollback;
- failsink/discard callback crash/resume;
- external-flush-before-outcome duplication characterization.

## Barrier completion

### Barrier production chain

```text
SchedulerDrainCoordinator (TS-07 BLOCKED)
  -> BarrierIntakeCoordinator
  -> adopt/reset/replay AUX operations
  -> AggregationExecutor or CoalesceExecutor
  -> RowProcessor completion method
  -> TokenSchedulerRepository.complete_barrier
  -> BarrierJournalRepository.complete_barrier
```

Strongest current tests:

- `test_scheduler_repository_complete_barrier.py::test_complete_barrier_consumes_and_emits_atomically`
- passthrough, READY emission, combined lanes, crash atomicity, late arrival,
  snapshot isolation, scoped coalesce, and duplicate/coverage refusal families
- `test_scheduler_repository_adopt_barrier_item.py` adoption, idempotency,
  stale-token refusal, and rollback-after-CAS cases
- `test_scheduler_repository_coalesce_branch_losses.py` replay, adoption,
  idempotency, and stale-token cases
- suspended-winner E2E cases for completion, adoption, and branch-loss replay
- real aggregation output durability and failed-flush reconcile tests
- real barrier intake tests for restore, branch loss, takeover, and late release
- processor aggregation/coalesce terminal and nonterminal paths

Next discriminators:

- explicit-None strict completion must refuse before all mutation;
- successful AUX-03–05 no-event assertions;
- full F-09 before/after images;
- successful aggregation crash before TS-15;
- non-sink aggregation crash after TS-15 before child TS-00;
- coalesce crash after merge audit before scheduler completion;
- orphan-DRAFT batch restart non-interference.

## Fencing and reads

Strongest current tests:

- `tests/unit/engine/test_unresolved_scheduler_work_invariant.py` for RM-01
- `tests/unit/core/test_count_ready_in_set.py` for RM-05 component reads
- `tests/unit/engine/test_adr030_loosened_invariant_guard.py` for relinquishment
- `tests/unit/core/landscape/test_multi_source_foundation.py` for active and
  unresolved reads
- `tests/unit/core/landscape/test_coordination_fence_constructs.py` for
  membership gates
- `tests/unit/core/landscape/test_evict_worker_housekeeping.py` for
  eviction-before-reap state behavior
- `tests/unit/core/landscape/test_leader_fence_stale_token.py` and
  `tests/e2e/recovery/test_suspended_winner_fences.py` for epoch refusal
- processor-mode and leader/follower orchestration tests for role separation

Next discriminators:

- one focused RM-02/RM-03/RM-04/RM-06 truth-table file;
- peer-attributed PENDING_SINK RM-05 relinquishment;
- production coordinator eviction-before-reap ordering;
- explicit row/event preservation for membership refusals;
- registered process proof under `elspeth-9a52eb80f9`.

## Existing issue ownership discovered during Wave 2

| Scope | Issue |
| --- | --- |
| Common None-to-plain fencing downgrade | `elspeth-e66c371acb` |
| Unfenced legacy barrier wrappers | `elspeth-b68bf5c161` |
| Broad leader-owned missing-token behavior | `elspeth-97c7661957` |
| Registered multiprocess and sink-redrive proof | `elspeth-9a52eb80f9` |
| Unbounded TS-13 token-ID batch | `elspeth-2861b3b0fa` |
| Unbounded TS-14 repair transaction | `elspeth-edc2698211` |

No dedicated owner was found for the read-model truth-table gap or sink
candidates 12/13. The remediation plan delays new issue creation until the
executed ledger records the exact confirmed gap.
