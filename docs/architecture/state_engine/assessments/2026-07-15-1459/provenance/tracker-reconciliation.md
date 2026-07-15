# Tracker Reconciliation

## Existing state-engine owners retained

| Issue | Scope |
| --- | --- |
| `elspeth-c0d4a28e11` | TS-00/01 refusal, replay, and atomicity proof |
| `elspeth-9cd07962c7` | TS-02/PB-01 ingress and scheduler exclusion |
| `elspeth-aafba3b298` | TS-02 to source-COMPLETED crash seam |
| `elspeth-9a52eb80f9` | Registered multiprocess orchestration and sink-redrive proof |
| `elspeth-d8e172676c` | Malformed PENDING_SINK claim defect |
| `elspeth-76bb92bc7d` | TS-04/06 complete sink-bundle proof |
| `elspeth-2aba594afb` | AUX-01/02 heartbeat and lease-loss integration |
| `elspeth-51a4b5c771` | Plugin execution beyond lease/stall budget |
| `elspeth-f8f9272b68` | Sink-redrive normal disposition defect |
| `elspeth-1076e2716a` | TS-07–10 effects, guards, and rollback matrix |
| `elspeth-2e66723070` | PB-02/03 real transform and gate disposition proof |
| `elspeth-7cdc4da434` | Child enqueue before parent disposition crash seam |
| `elspeth-6f6bbbec00` | PB-08 real follower build/drain/plugin path |
| `elspeth-c25bcf5717` | Standalone initial-claim membership fence |
| `elspeth-e66c371acb` | Common fencing helper None downgrade |
| `elspeth-b68bf5c161` | Unfenced barrier legacy wrappers |
| `elspeth-97c7661957` | Broad leader-owned missing-token writes |
| `elspeth-2861b3b0fa` | Unbounded TS-13 token-ID batch |
| `elspeth-edc2698211` | Unbounded TS-14 repair transaction |

## Candidate gaps without a dedicated owner at assessment time

- RM-02–RM-06 complete truth tables and the peer PENDING_SINK RM-05 consumer arm.
- Candidate 12: real non-aggregation transform/gate through sink durability and
  scheduler terminalization.
- Candidate 13: scheduler-backed failsink/discard exact-once crash/resume proof.
- AUX-03–05 explicit no-event/rollback/concurrency package.
- Candidates 16, 21, 22, and 23 barrier/aggregation crash seams.

No new issue was created during Wave 2 reconnaissance or this documentation
task. The campaign will file only a positively executed gap that remains
unowned after a fresh live search.
