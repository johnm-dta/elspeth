# Wave 1 Evidence Inventory

## Intake and identity

Primary suites:

- `tests/unit/core/landscape/test_scheduler_events.py`
- `tests/unit/core/landscape/test_coordination_fence_constructs.py`
- `tests/unit/core/landscape/test_leader_fence_stale_token.py`
- `tests/integration/engine/test_two_process_scheduler_contention.py`
- `tests/e2e/recovery/test_suspended_winner_fences.py`
- source-boundary and source-iteration processor/orchestrator tests

Key conclusions:

- TS-03 has complete direct-repository claim proof, including two processes.
- TS-00/01 still lack complete refusal and rollback images.
- TS-02 stale fencing is strong, but successful source composition and the
  post-commit source-state seam remain incomplete.
- PB-01 has broad plugin evidence but incomplete no-scheduler-row assertions.

## Leasing and recovery

Primary suites:

- `tests/property/engine/test_scheduler_work_item_lifecycle_state_machine.py`
- `tests/unit/core/landscape/test_scheduler_lease_recovery_races.py`
- `tests/unit/core/landscape/test_scheduler_events.py`
- `tests/unit/core/landscape/test_evict_worker_housekeeping.py`
- `tests/unit/core/landscape/test_leader_fence_stale_token.py`

Key conclusions:

- TS-05 is Confirmed for its bounded direct-repository claim.
- TS-04 accepted a malformed PENDING_SINK subtype.
- TS-06 has identity/attempt and ABA evidence but not one complete bundle/event
  proof or real sink-redrive contention.
- AUX-01 and AUX-02 have strong fragments but not complete event/abandonment
  packages.
- F-07 is Confirmed; the plugin-outlives-stall-budget case remains separate.

## Claimed dispositions and plugin boundaries

Primary suites:

- `tests/unit/engine/test_scheduler_drain_characterization.py`
- `tests/unit/engine/test_processor.py`
- `tests/unit/engine/test_token_traversal_characterization.py`
- `tests/unit/engine/orchestrator/test_follower_processor.py`
- `tests/unit/core/landscape/test_scheduler_events.py`
- `tests/unit/core/landscape/test_leader_fence_stale_token.py`

Key conclusions:

- TS-07–TS-10 all execute real drain paths, but no leg has the full effects,
  event, guard, subtype, and rollback matrix.
- A sink-redrive LEASED row accepts the normal disposition helper, confirming a
  shared subtype defect.
- PB-02 and PB-03 prove real plugin work and repository disposition separately,
  not as one durable story.
- PB-08 tests follower-shaped outcomes without the real follower construction
  and traversal path.

## Confirmed Wave 1 ownership

| Scope | Filigree issue |
| --- | --- |
| TS-00/01 refusal, replay, atomicity | `elspeth-c0d4a28e11` |
| TS-02/PB-01 ingress and scheduler exclusion | `elspeth-9cd07962c7` |
| TS-02 to source-COMPLETED seam | `elspeth-aafba3b298` |
| Registered multiprocess/sink-redrive proof | `elspeth-9a52eb80f9` |
| Malformed PENDING_SINK claim | `elspeth-d8e172676c` |
| TS-04/06 sink-bundle proof | `elspeth-76bb92bc7d` |
| AUX-01/02 heartbeat/lease-loss integration | `elspeth-2aba594afb` |
| Long plugin lease/stall behavior | `elspeth-51a4b5c771` |
| Sink-redrive normal disposition defect | `elspeth-f8f9272b68` |
| TS-07–10 effects/guards/rollback | `elspeth-1076e2716a` |
| PB-02/03 plugin disposition composition | `elspeth-2e66723070` |
| Child enqueue before parent disposition | `elspeth-7cdc4da434` |
| PB-08 real follower path | `elspeth-6f6bbbec00` |
| Initial-claim membership fence | `elspeth-c25bcf5717` |

These issue links were deduplicated during Wave 1 before filing. The detailed
leg verdicts are preserved in
[03-executed-evidence.md](../03-executed-evidence.md).
