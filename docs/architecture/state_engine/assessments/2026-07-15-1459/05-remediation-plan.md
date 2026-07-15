# Token Scheduler State Engine Remediation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert every mapped token-scheduler transition, auxiliary mutation,
plugin boundary, read decision, and forbidden path into an executed fail-closed
contract, then publish one comprehensive state-engine ADR.

**Architecture:** Work in proof packages that each enter through production,
assert complete durable images, and add direct repository detail only where
needed. Fix reproduced subtype/fencing defects behind those tests, then execute
cross-transaction crash and registered-process suites before declaring any
architectural guarantee.

**Tech Stack:** Python 3.13, pytest, SQLAlchemy, SQLite `BEGIN IMMEDIATE`, real
Elspeth plugin fixtures, multiprocessing, Filigree, Loomweave, Markdown, Mermaid.

---

## Execution rules

Apply this loop to every task:

1. Start or confirm the exact Filigree owner before editing.
2. Add the smallest test that expresses the missing proof or reproduced defect.
3. Run the exact node in isolation and retain its result.
4. If the test fails, capture the full durable before/after image before changing
   implementation. If it passes, classify the work as evidence closure rather
   than an implementation fix.
5. Implement only the minimal in-scope correction when an owned defect is
   reproduced.
6. Run the package suite, adjacent state-machine/property suites, and
   `git diff --check`.
7. Update the next dated assessment and Filigree only after the evidence is
   reviewed. Do not rewrite this assessment.
8. Commit one coherent proof/fix package at a time.

## File map

| Responsibility | Planned files |
| --- | --- |
| Read-model truth tables | Create `tests/unit/core/landscape/test_scheduler_read_model_truth_tables.py`; modify `tests/unit/engine/test_adr030_loosened_invariant_guard.py` |
| Maintenance/fencing read consumers | Modify `tests/unit/engine/test_scheduler_drain_characterization.py`, `tests/unit/core/landscape/test_coordination_fence_constructs.py` |
| Sink durability composition | Create `tests/integration/engine/test_sink_scheduler_durability.py` |
| Sink repository rollback | Modify `tests/unit/core/landscape/test_scheduler_events.py`, `tests/unit/core/landscape/test_leader_fence_stale_token.py` |
| Diversion crash/resume | Modify `tests/integration/test_adr_019_resume_counter_parity.py` and sink diversion fixtures |
| Barrier authority/atomic detail | Modify `tests/unit/core/landscape/test_scheduler_repository_complete_barrier.py`, adoption, and branch-loss repository suites |
| Aggregation/coalesce crash seams | Modify `tests/integration/pipeline/test_aggregation_recovery.py`, `tests/integration/pipeline/test_barrier_intake_dispositions.py` |
| Registered multiprocess proof | Extend `tests/integration/engine/test_two_process_scheduler_contention.py` and recovery E2E suites |
| Plugin lifecycle/forbidden paths | Extend orchestrator cleanup, follower, state-machine property, and scheduler characterization suites |
| Evidence closeout | Create a new `docs/architecture/state_engine/assessments/<timestamp>/`; later add the superseding ADR under `docs/architecture/adr/` |

### Task 1: Complete RM-02, RM-03, RM-04, and RM-06 truth tables

**Files:**

- Create: `tests/unit/core/landscape/test_scheduler_read_model_truth_tables.py`
- Reuse: `tests/fixtures/landscape.py`
- Verify: `src/elspeth/core/landscape/scheduler/read_model.py`

- [ ] **Step 1: Add a production-repository fixture**

Create one registered run through `RecorderSetup`. Insert legal rows through
`enqueue_ready`, `claim_ready`, `mark_blocked`, `mark_pending_sink`,
`mark_terminal`, and `mark_failed`. Use direct SQL only for an ownerless malformed
exclusion or exact `lease_expires_at == now` boundary that no legal public verb
can preserve.

- [ ] **Step 2: Add `test_unquiesced_work_truth_table`**

Assert `count_unquiesced_work` and `summarize_unquiesced_work` include exactly:

```text
READY
LEASED with pending_sink_name IS NULL
```

Assert they exclude BLOCKED, PENDING_SINK, sink-redrive LEASED, TERMINAL,
FAILED, and rows belonging to another run. Assert the exact grouped summary
strings, including node and lease owner.

- [ ] **Step 3: Add `test_active_work_truth_table`**

Assert `count_active_work`, `summarize_active_work`, and `active_row_ids` include
READY, transform LEASED, sink-redrive LEASED, BLOCKED, and PENDING_SINK. Assert
they exclude TERMINAL, FAILED, and other-run rows.

- [ ] **Step 4: Add `test_peer_owned_work_truth_table`**

Assert true for another owner's LEASED row and attributed PENDING_SINK row.
Assert false when the only candidates are caller-owned, ownerless, READY,
BLOCKED, TERMINAL, FAILED, or from another run.

- [ ] **Step 5: Add `test_peer_active_leases_truth_table_and_ordering`**

Assert `peer_active_leases` returns distinct, sorted peer owners only for LEASED
rows with `lease_expires_at > now`. Exclude equality, expiry, caller ownership,
ownerless rows, and non-LEASED states.

- [ ] **Step 6: Run the read-model package**

```bash
.venv/bin/pytest -q \
  tests/unit/core/landscape/test_scheduler_read_model_truth_tables.py \
  tests/unit/engine/test_unresolved_scheduler_work_invariant.py
```

**Exit gate:** Every RM-01–RM-04 and RM-06 state/subtype/run/owner arm has one
focused production-repository assertion. Any failing arm becomes a confirmed,
deduplicated defect before implementation changes.

### Task 2: Close RM-05 and maintenance-order consumer gaps

**Files:**

- Modify: `tests/unit/engine/test_adr030_loosened_invariant_guard.py`
- Modify: `tests/unit/engine/test_scheduler_drain_characterization.py`
- Modify: `tests/unit/core/landscape/test_coordination_fence_constructs.py`

- [ ] **Step 1: Add `test_peer_pending_sink_handoff_relinquishes_without_active_lease`**

Construct a pending continuation whose durable row is peer-attributed
PENDING_SINK. Assert:

```text
peer_active_leases() == ()
has_peer_owned_work() is True
drain returns no leader-processed token
the in-memory pending set is cleared
the relinquishment decision is logged
```

- [ ] **Step 2: Add `test_run_maintenance_evicts_each_dead_worker_before_recovering_leases`**

Invoke the production maintenance coordinator. Record repository calls and
assert every dead worker's eviction commits before lease recovery starts. Then
assert the recovered row and its scheduler/coordination events match the
post-eviction liveness arm.

- [ ] **Step 3: Ratchet membership refusal event images**

For absent, departed, and evicted claim/disposition callers, snapshot
`token_work_items` and `scheduler_events`, invoke the public method, and assert
both snapshots are unchanged. Preserve the coordination refusal evidence.

- [ ] **Step 4: Run the consumer package**

```bash
.venv/bin/pytest -q \
  tests/unit/engine/test_adr030_loosened_invariant_guard.py \
  tests/unit/engine/test_scheduler_drain_characterization.py \
  tests/unit/core/landscape/test_coordination_fence_constructs.py
```

**Exit gate:** RM-05 includes the peer PENDING_SINK arm; eviction-before-reap is
proved through the production coordinator; AUX-06/F-06 refusals preserve both
row and scheduler-event images.

### Task 3: Prove primary sink durability through TS-13 and TS-14

**Files:**

- Create: `tests/integration/engine/test_sink_scheduler_durability.py`
- Exercise: `src/elspeth/engine/orchestrator/sink_flush.py`
- Exercise: `src/elspeth/engine/executors/sink.py`
- Exercise: `src/elspeth/core/landscape/scheduler/dispositions.py`

- [ ] **Step 1: Add `test_real_transform_primary_sink_commits_audit_before_ts13_terminalization`**

Run a production `Orchestrator` pipeline using `ListSource`, a real pass-through
transform, and real `CSVSink`. Assert one external record, COMPLETED sink node
state, persisted artifact and hash, one completed token outcome, TERMINAL work
row with scrubbed payload/cleared lease, and exactly one attributed
`PENDING_SINK -> TERMINAL` event.

- [ ] **Step 2: Add `test_post_outcome_pre_terminalization_crash_resumes_without_sink_reemit`**

Enable per-row checkpointing and inject one exception in
`RowProcessor.mark_sink_bound_scheduler_terminal_many`. Before resume, assert
the CSV bytes, artifact, and outcome are durable; the scheduler row remains
PENDING_SINK; and no terminal event exists. Resume through `Orchestrator.resume`
and assert unchanged output bytes/emission count, no duplicate outcome,
TERMINAL state, and one TS-14 repair event.

- [ ] **Step 3: Run the new integration file**

```bash
.venv/bin/pytest -q tests/integration/engine/test_sink_scheduler_durability.py
```

**Exit gate:** A real plugin path proves the outcome-before-close invariant and
the post-witness restart path proves no sink re-emission.

### Task 4: Complete sink guard, rollback, diversion, and fencing proof

**Files:**

- Modify: `tests/unit/core/landscape/test_scheduler_events.py`
- Modify: `tests/unit/core/landscape/test_leader_fence_stale_token.py`
- Modify: `tests/integration/test_adr_019_resume_counter_parity.py`

- [ ] **Step 1: Add complete TS-13 rollback images**

Add:

- `test_pending_sink_batch_missing_token_rolls_back_all_rows_and_events`
- `test_pending_sink_batch_event_failure_rolls_back_all_rows_and_events`

Snapshot every requested row and all scheduler events. Assert byte-for-byte
equivalence after the expected refusal/injected event failure.

- [ ] **Step 2: Add complete TS-14 zero-mutation/rollback images**

Add:

- `test_pending_sink_repair_without_outcome_is_zero_mutation`
- `test_pending_sink_repair_event_failure_rolls_back_rows_and_events`

- [ ] **Step 3: Pin explicit-None repair fencing**

Add `test_explicit_none_repair_token_is_refused_with_zero_mutation`. The test is
expected to fail against the current helper downgrade. Fix it only under
`elspeth-97c7661957`/`elspeth-e66c371acb`, keeping any intentionally unfenced API
explicitly named and unreachable from production.

- [ ] **Step 4: Add failsink and discard callback-crash/resume cases**

For both diversion modes, crash after durable outcome but before scheduler
callback. Resume from the same database and assert no repeated plugin-visible
emission, no duplicate outcome, one terminal scheduler event, and stable
audit-derived counters.

- [ ] **Step 5: Characterize the pre-witness duplication window**

Inject after external failsink flush but before outcome persistence. Record the
restart behavior and plugin call count. Do not label repeated I/O a defect until
the ADR declares whether the plugin contract requires idempotency at that seam.

- [ ] **Step 6: Run sink regressions**

```bash
.venv/bin/pytest -q \
  tests/unit/core/landscape/test_scheduler_events.py \
  tests/unit/core/landscape/test_leader_fence_stale_token.py \
  tests/integration/engine/test_sink_scheduler_durability.py \
  tests/integration/test_adr_019_resume_counter_parity.py
```

**Exit gate:** TS-11–TS-14 and PB-06/07 have complete success, guard, rollback,
repair, and diversion evidence; the external pre-witness boundary is explicitly
characterized.

### Task 5: Make barrier authority, no-event behavior, and F-09 rollback explicit

**Files:**

- Modify: `tests/unit/core/landscape/test_scheduler_repository_complete_barrier.py`
- Modify: `tests/unit/core/landscape/test_scheduler_repository_adopt_barrier_item.py`
- Modify: `tests/unit/core/landscape/test_scheduler_repository_coalesce_branch_losses.py`
- Modify: `tests/unit/core/landscape/test_leader_fence_stale_token.py`

- [ ] **Step 1: Add `test_complete_barrier_explicit_none_token_refuses_before_any_mutation`**

Snapshot complete work rows, scheduler events, branch losses, emissions, batch
membership, and adoption markers. Call the strict public completion method with
explicit `None`. Require an invariant error and unchanged snapshots. This is
expected to fail until the two existing P1 fencing issues are fixed.

- [ ] **Step 2: Assert the intentional AUX event boundary**

For successful AUX-03 adoption, AUX-04 reset, and AUX-05 branch-loss adoption,
assert the auxiliary mutation commits while the scheduler-event snapshot remains
unchanged. Record the result for the ADR's transition-only event-plane decision.

- [ ] **Step 3: Parameterize complete F-09 refusal images**

For duplicate, overlap, missing, foreign, uncovered, cross-group, incomplete
cursor, and out-of-snapshot cases, compare every relevant work row, event,
emission, branch-loss, and auxiliary row before and after refusal.

- [ ] **Step 4: Run barrier repository suites**

```bash
.venv/bin/pytest -q \
  tests/unit/core/landscape/test_scheduler_repository_complete_barrier.py \
  tests/unit/core/landscape/test_scheduler_repository_adopt_barrier_item.py \
  tests/unit/core/landscape/test_scheduler_repository_coalesce_branch_losses.py \
  tests/unit/core/landscape/test_leader_fence_stale_token.py
```

**Exit gate:** Strict completion cannot enter an unfenced arm; every F-09 guard
is transactionally inert; AUX-03–05 event behavior is explicit and tested.

### Task 6: Execute aggregation and coalesce crash-seam discriminators

**Files:**

- Modify: `tests/integration/pipeline/test_aggregation_recovery.py`
- Modify: `tests/integration/pipeline/test_barrier_intake_dispositions.py`

- [ ] **Step 1: Add successful aggregation crash before barrier completion**

Add
`test_successful_transform_flush_crash_before_barrier_completion_resumes_exactly_once`.
Use the real sum-batch transform. Inject once at `_complete_aggregation_flush`.
Assert the durable batch/node/outcome image, BLOCKED inputs, absent scheduler
emission/event, then restart. The restart must either complete exactly once or
produce a stable fail-closed error that becomes a confirmed implementation gap.

- [ ] **Step 2: Add non-sink crash after TS-15 before child TS-00**

Add `test_non_sink_flush_crash_after_barrier_completion_resumes_child_exactly_once`.
Interrupt the first downstream processing call after TS-15. Assert consumed
inputs and the exact missing/present continuation image, restart, and require one
downstream delivery with no duplicate child identity.

- [ ] **Step 3: Add coalesce crash after merge audit before completion**

Add `test_coalesce_crash_after_merge_audit_before_barrier_completion_resumes_exactly_once`.
Use real `CoalesceExecutor`, inject after `accept()` returns the merged result,
assert decision/audit/merged-token durability with inputs still BLOCKED and no
scheduler emission, then restart and require exactly one continuation.

- [ ] **Step 4: Add orphan-DRAFT restart non-interference**

Create a DRAFT aggregation batch, interrupt before AUX-03 membership adoption,
restart intake, and assert the orphan cannot capture, duplicate, or suppress a
later valid batch.

- [ ] **Step 5: Run the barrier integration package**

```bash
.venv/bin/pytest -q \
  tests/integration/pipeline/test_aggregation_recovery.py \
  tests/integration/pipeline/test_barrier_intake_dispositions.py
```

**Exit gate:** Candidates 16, 21, 22, and 23 receive executed classifications
and deterministic restart contracts. File only newly reproduced, deduplicated
defects.

### Task 7: Close remaining Wave 1 proof owners

**Files:**

- Modify the exact production and test files named in issues
  `elspeth-c0d4a28e11`, `elspeth-9cd07962c7`, `elspeth-aafba3b298`,
  `elspeth-76bb92bc7d`, `elspeth-2aba594afb`, `elspeth-f8f9272b68`,
  `elspeth-1076e2716a`, `elspeth-2e66723070`, `elspeth-7cdc4da434`, and
  `elspeth-6f6bbbec00`.

- [ ] **Step 1: Process one issue per coherent close/commit cycle**

Preserve the fail-closed subtype model. Thread verified subtype/authority through
the proper helper boundary instead of loosening a guard.

- [ ] **Step 2: Re-run each exact issue regression and its family suite**

Record complete state/event/effect images in the next assessment. Re-audit
current HEAD after every implementation fix; do not rely on the original
reproducer alone.

**Exit gate:** Every Wave 1 Gap leg is either Confirmed, linked to a still-open
reproduced defect, or governed by an accepted intentionally-absent decision.

### Task 8: Add registered multi-process and long-plugin proof

**Files:**

- Modify: `tests/integration/engine/test_two_process_scheduler_contention.py`
- Modify: `tests/e2e/recovery/test_multi_worker_leader_finalize.py`
- Modify: `tests/e2e/recovery/test_suspended_winner_fences.py`
- Modify the long-plugin test surface named by `elspeth-51a4b5c771`

- [ ] **Step 1: Run contenders as registered workers**

Use separate operating-system processes and database connections with active
`run_workers` rows. Cover READY claim, pending-sink claim, transform recovery,
sink-redrive recovery, TS-13 terminalization, barrier completion, and stale
leader completion.

- [ ] **Step 2: Assert complete winner/loser images**

Prove one owner/effective event, no losing payload mutation, stable work identity
rules, and the expected coordination refusal/liveness evidence.

- [ ] **Step 3: Cross lease TTL and stall budget during a real plugin call**

Suspend a plugin process beyond both thresholds while another worker performs
maintenance. Resume the old process and distinguish heartbeat-detected lease
loss from stale disposition. Assert the plugin effect and durable disposition
occur no more than the declared contract permits.

**Exit gate:** Every concurrency statement in the state-engine ADR is backed by
real processes, not mocks or a single connection.

### Task 9: Close plugin lifecycle and forbidden/dormant paths

**Files:**

- Modify: `tests/integration/pipeline/orchestrator/test_orchestrator_cleanup.py`
- Modify: `tests/unit/engine/orchestrator/test_cleanup_failure_ceremony.py`
- Modify: `tests/unit/engine/orchestrator/test_follower_processor.py`
- Modify: `tests/property/engine/test_scheduler_work_item_lifecycle_state_machine.py`
- Modify: `tests/unit/engine/test_scheduler_drain_characterization.py`

- [ ] **Step 1: Build the PB-09 lifecycle matrix**

For source, transform, and sink plugins, cover fresh execution, resume,
follower, partial-start failure, normal teardown, plugin exception, and teardown
exception. Assert call order, exactly-once close semantics, scheduler state, and
retained primary error.

- [ ] **Step 2: Execute every F-01–F-13 refusal**

Use complete row/event/auxiliary snapshots. Decide queue-only BLOCKED reachability
with a production traversal search plus regression: prove a real producer,
reserve it explicitly, or remove/reject the subtype.

**Exit gate:** PB-09 is Confirmed and every forbidden path is Confirmed or
Intentionally absent with an accepted decision and regression.

### Task 10: Publish the final assessment, tracker reconciliation, ADR, and CI gate

**Files:**

- Create: `docs/architecture/state_engine/assessments/<new-timestamp>/`
- Modify: `docs/architecture/state_engine/README.md`
- Create: next available `docs/architecture/adr/<NNN>-*.md`
- Modify superseded/amended ADR index entries only after acceptance
- Modify the repository's state-engine CI/test selection configuration

- [ ] **Step 1: Produce the final leg matrix**

Require every TS/AUX/PB/RM/F row to show production entry, success, refusal,
rollback, concurrency/plugin evidence, exact command, and final verdict.

- [ ] **Step 2: Reconcile Filigree**

Deduplicate all confirmed gaps, close only issues whose exact regressions pass,
and record operator/policy blockers explicitly rather than coding around them.

- [ ] **Step 3: Draft the comprehensive state-engine ADR**

Define status/subtype vocabulary, identities, transactions, event planes,
fencing, read models, plugin boundaries, recovery seams, sink duplication
contract, and forbidden paths. Explicitly state whether each of ADR-001, 019,
021, 025, 026, 028, 029, and 030 is superseded, amended, or retained.

- [ ] **Step 4: Make the proof matrix mandatory**

Add a CI target that runs the maintained state-engine contract, including
registered process tests where the environment supports them. A skipped
mandatory cell fails the release gate.

- [ ] **Step 5: Update the hub**

Point `docs/architecture/state_engine/README.md` at the new immutable assessment
and replace the current not-complete verdict only if every completeness gate
passes.

**Exit gate:** One accepted ADR matches live code and a mandatory executed
matrix; no leg is Candidate, Unknown, or unowned Gap.

## Recommended execution mode

Use subagent-driven development with one fresh worker per task or tightly
coupled subtask. Review every package twice: first against this assessment's
proof criteria, then against the exact source diff and test output. Do not run
multiple agents against the same files concurrently.
