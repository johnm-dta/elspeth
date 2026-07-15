# Source and Test Index

## Primary production files

- `src/elspeth/contracts/plugin_protocols.py`
- `src/elspeth/contracts/scheduler.py`
- `src/elspeth/core/landscape/schema.py`
- `src/elspeth/core/landscape/run_coordination_repository.py`
- `src/elspeth/core/landscape/run_lifecycle_repository.py`
- `src/elspeth/core/landscape/scheduler/`
- `src/elspeth/core/landscape/scheduler_repository.py`
- `src/elspeth/engine/barrier_coordination.py`
- `src/elspeth/engine/coalesce_executor.py`
- `src/elspeth/engine/processor.py`
- `src/elspeth/engine/scheduler_drain.py`
- `src/elspeth/engine/token_traversal.py`
- `src/elspeth/engine/executors/aggregation.py`
- `src/elspeth/engine/executors/gate.py`
- `src/elspeth/engine/executors/sink.py`
- `src/elspeth/engine/executors/transform.py`
- `src/elspeth/engine/orchestrator/cleanup.py`
- `src/elspeth/engine/orchestrator/follower.py`
- `src/elspeth/engine/orchestrator/leader_drain.py`
- `src/elspeth/engine/orchestrator/leader_follower_drain.py`
- `src/elspeth/engine/orchestrator/resume.py`
- `src/elspeth/engine/orchestrator/run_context_factory.py`
- `src/elspeth/engine/orchestrator/sink_flush.py`
- `src/elspeth/engine/orchestrator/source_iteration.py`

## High-value test suites

- `tests/property/engine/test_scheduler_work_item_lifecycle_state_machine.py`
- `tests/unit/core/landscape/test_scheduler_events.py`
- `tests/unit/core/landscape/test_scheduler_repository_complete_barrier.py`
- `tests/unit/core/landscape/test_scheduler_repository_adopt_barrier_item.py`
- `tests/unit/core/landscape/test_scheduler_repository_coalesce_branch_losses.py`
- `tests/unit/core/landscape/test_scheduler_lease_recovery_races.py`
- `tests/unit/core/landscape/test_leader_fence_stale_token.py`
- `tests/unit/core/landscape/test_coordination_fence_constructs.py`
- `tests/unit/core/landscape/test_evict_worker_housekeeping.py`
- `tests/unit/core/test_count_ready_in_set.py`
- `tests/unit/core/test_multi_source_foundation.py`
- `tests/unit/engine/test_adr030_loosened_invariant_guard.py`
- `tests/unit/engine/test_unresolved_scheduler_work_invariant.py`
- `tests/unit/engine/test_scheduler_drain_characterization.py`
- `tests/unit/engine/test_processor.py`
- `tests/unit/engine/test_token_traversal_characterization.py`
- `tests/unit/engine/orchestrator/test_follower_processor.py`
- `tests/unit/engine/orchestrator/test_cleanup_failure_ceremony.py`
- `tests/unit/engine/orchestrator/test_adr030_follower_teardown.py`
- `tests/integration/pipeline/test_barrier_intake_dispositions.py`
- `tests/integration/pipeline/test_aggregation_recovery.py`
- `tests/integration/plugins/sinks/test_durability.py`
- `tests/integration/engine/test_two_process_scheduler_contention.py`
- `tests/integration/pipeline/orchestrator/test_orchestrator_cleanup.py`
- `tests/integration/pipeline/orchestrator/test_orchestrator_execute_run_characterization.py`
- `tests/e2e/recovery/test_follower_join_and_drain.py`
- `tests/e2e/recovery/test_multi_worker_leader_finalize.py`
- `tests/e2e/recovery/test_suspended_winner_fences.py`

## Related decisions and contracts

- `docs/architecture/adr/001-plugin-level-concurrency.md`
- `docs/architecture/adr/019-two-axis-terminal-model.md`
- `docs/architecture/adr/021-sources-and-sinks-uniformly-boundary.md`
- `docs/architecture/adr/025-multi-source-ingestion.md`
- `docs/architecture/adr/026-durable-token-scheduler.md`
- `docs/architecture/adr/028-queue-vs-coalesce-not-duplicates.md`
- `docs/architecture/adr/029-journal-is-barrier-buffer-truth.md`
- `docs/architecture/adr/030-multi-worker-deployment-shape.md`
- `docs/architecture/token-lifecycle.md`
- `docs/architecture/barrier-machinery.md`
- `docs/runbooks/scheduler-lease-recovery.md`

The future comprehensive ADR must state explicitly whether each prior decision
is superseded, amended, or retained for an independent concern.
