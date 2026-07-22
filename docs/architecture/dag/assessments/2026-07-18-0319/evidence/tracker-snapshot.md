# Filigree integration-delta snapshot

**Captured:** 2026-07-18 03:19 AEST
**Query scope:** DAG corpus owner, three repaired hard gates, and directly
remaining crash/contention owners

## Repaired hard gates

| Issue | Live state | Close commit | Reconciled consequence |
| --- | --- | --- | --- |
| `elspeth-a25e9c009e` — expansion parent-outcome bypass | closed | `release/0.7.1@84d296d5b` | Remove the reproduced expansion replay failure and closed owner from the live manifest; retain incomplete scenario proof. |
| `elspeth-3335de38c2` — output-contract last-writer-wins | closed | `release/0.7.1@84d296d5b` | Remove the known output-contract serialization defect from the atomicity hard-gate list. |
| `elspeth-d8d4d2849b` — sidecar journal ahead of commit | closed | `release/0.7.1@84d296d5b` | Remove the known journal publication defect from the atomicity hard-gate list. |

All three closure records include root cause, exact repaired invariant,
cross-backend verification, and the same release close commit.

## Current acceptance and remaining owners

| Theme | Owner | Live state at capture |
| --- | --- | --- |
| Maintained fifteen-scenario corpus | `elspeth-ef29ef6ba4` | in progress P1 |
| Child enqueue before parent disposition | `elspeth-7cdc4da434` | open, ready P1 |
| Source ingress to source `COMPLETED` | `elspeth-aafba3b298` | open, ready P1 |
| Long plugin beyond lease/stall budget | `elspeth-51a4b5c771` | open, ready P1 |
| Strict coordination-token closure | `elspeth-97c7661957` | fixing P2 |
| Registered orchestration and sink-redrive contention | `elspeth-9a52eb80f9` | open, ready P2 |
| Real transform/gate disposition | `elspeth-2e66723070` | open, ready P2 |
| Production follower traversal | `elspeth-6f6bbbec00` | open, ready P2 |

The seven graph-config security owners recorded in the 2026-07-17 assessment
remain open or confirmed. The release URL and branch-loss redaction fixes are
adjacent improvements; they do not close those graph identity, metadata,
settings, gate-condition, or export owners.

## Reconciliation rule

Closed issue state is not sufficient by itself to promote a corpus cell. The
manifest changes in this delta also require current executable evidence. For
that reason only row-expansion Recovery and Concurrency change status; the
output-contract and journal closures update the hard-gate account while their
broader scenario cells retain their prior ratings.
