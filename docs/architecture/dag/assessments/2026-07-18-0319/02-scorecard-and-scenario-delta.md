# DAG integration delta scorecard and scenario matrix

**Baseline:** `codex/dag-scenario-corpus` at
`0235739274b534bd9e4e2b859bdd94a0b6a09651`
**Verdict:** **Not complete**
**Normalized maturity indicator:** **Not calculated** — mandatory dimensions
remain `U`, so the framework prohibits an aggregate.

## Executive assessment

The release bugfix run materially improves the DAG safety floor. Expansion now
consumes an atomic parent or batch claim and refuses duplicate replay;
output-contract inference uses canonical merge plus compare-and-swap; and the
sidecar journal publishes from a committed outbox with idempotent recovery.
The original Filigree reproducers for all three are closed at `84d296d5b`.

These closures remove three reproduced defects from the 2026-07-17 hard-gate
list. They do not make the DAG product complete. Source and parent/child crash
seams, registered-process and long-plugin contention, graph-config secrecy,
guided/browser parity, round-trip coverage, scale limits, and the normative
contract remain incomplete.

## Closed hard gates

| Finding | Current result | Current evidence |
| --- | --- | --- |
| `elspeth-a25e9c009e` — expansion could create children without consuming the parent | Closed. Batch expansion records a terminal batch-parent outcome atomically; sequential replay through another member is refused; concurrent PostgreSQL attempts produce one child set and reject the loser. | `cardinality-identity-09`, `cardinality-identity-10`, `cardinality-identity-11`; focused unit and 5-testcontainer runs |
| `elspeth-3335de38c2` — output-contract updates were last-writer-wins | Closed. Canonical merge and hash CAS preserve compatible concurrent observations and reject incompatible state. | `test_graph_recording.py`, PostgreSQL output-contract concurrency tests |
| `elspeth-d8d4d2849b` — sidecar journal could record an uncommitted transaction | Closed. A transaction-owned outbox publishes after commit and recovers idempotently. | `test_journal.py`, PostgreSQL journal tests |

## Fifteen-dimension scorecard

The numerical rows do not change because each dimension remains limited by a
different mandatory gap. The limiting descriptions do change where the
repaired defects previously appeared.

| Dimension | Score | Status | Current evidence and limiting gate |
| --- | ---: | --- | --- |
| Topology expressiveness | 0 | Fail | Expansion is still not first-class topology metadata and row union remains unsupported or undecided. |
| Canonical configuration | U | Unknown | Deterministic identity passes, but no mandatory cross-input and cross-surface canonical matrix exists. |
| Structural validation | 4 | Pass | Production and graph paths fail closed for the maintained structural invariants. |
| Schema contracts | 0 | Fail | Output-contract CAS is repaired, but mandatory composition/plugin matrices and the row-union contract remain incomplete. |
| Cardinality and identity | 2 | Partial | Atomic expansion and selected cross-backend contention pass; the child-enqueue process-death seam and complete scenario proof remain open. |
| Compositional closure | U | Unknown | Exact sequential nested coalesces build; parallel and policy-complete runtime proof remains unknown. |
| Runtime semantics | U | Unknown | Selected production drain and disposition paths pass; mandatory plugin and composed-runtime cells remain unknown. |
| Durable recovery | U | Unknown | Expansion's reproduced replay defect is closed, but source, child-enqueue, whole-row, and registered-process restart cells remain unknown. |
| Concurrency and fencing | U | Unknown | Selected expansion and direct contention pass; registered orchestration, pending-sink/barrier contention, and long-plugin behavior remain unknown. |
| Atomic evidence | U | Unknown | Output-contract and journal defects are closed; queue/ingress, source-completion, and parent-child process-death proof remains incomplete. |
| Security | 1 | Fail | URL projection and branch-loss redaction improved, but raw graph configuration can still affect identity or persisted/exported evidence. |
| Authoring parity | 0 | Fail | Guided authoring remains topology-incomplete and the browser matrix is not acceptance evidence. |
| Semantic round-trip | U | Unknown | No canonical before/after matrix spans mandatory fixtures and authoring surfaces. |
| Scale | U | Unknown | No supported envelope, threshold behavior, required CI gate, or final owner proof exists. |
| Maintained contract | 1 | Fail | The normative execution-graph contract remains stale and required repository gates are not fully reconciled. |

Eight dimensions remain `U`; no aggregate is permitted.

## Mandatory scenario matrix

Legend: **P** Pass, **△** Partial, **F** Fail, **U** Unknown, **N/A** not
applicable.

| Scenario | Config | Build | Contracts | Runtime | Audit | Recovery | Concurrency | Freeform | Guided | Round-trip | Scale | Overall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Linear source -> transform -> sink | P | P | P | △ | △ | △ | U | P | △ | △ | △ | △ |
| Multiple independent sources | P | P | P | △ | △ | △ | U | P | F | △ | U | F |
| Multi-source queue fan-in | P | P | P | △ | △ | U | U | P | F | △ | U | F |
| Conditional routing | P | P | P | △ | △ | U | U | P | F | △ | U | F |
| Fork to multiple terminals with partial failure | P | P | P | △ | △ | U | U | P | F | U | U | F |
| Fork/coalesce policies and merge strategies | P | P | △ | △ | △ | △ | △ | P | F | △ | U | F |
| Sequential/nested forks and coalesces | P | P | △ | U | U | U | U | P | F | U | U | F |
| Parallel coalesces | P | △ | △ | U | U | U | U | P | F | U | U | F |
| Aggregation and immutable batch closure | P | P | △ | △ | △ | △ | U | P | F | U | U | F |
| Row expansion and parent/child recovery | P | P | △ | △ | △ | △ | △ | P | F | U | U | F |
| Row union/interleave | F | F | F | F | N/A | N/A | N/A | F | F | N/A | N/A | F |
| Retry/quarantine/discard/routed errors | P | P | △ | △ | △ | U | U | P | F | △ | U | F |
| Sink write and pending-sink redrive | P | P | P | △ | △ | △ | △ | P | △ | △ | U | △ |
| Checkpoint and deterministic resume | P | P | △ | △ | △ | △ | U | N/A | N/A | N/A | U | △ |
| Multi-worker execution and late completion | N/A | P | △ | △ | △ | △ | △ | N/A | N/A | N/A | U | △ |

Only the row-expansion row changes from the 2026-07-17 matrix:

- **Recovery: Fail -> Partial.** The reproduced duplicate/remint path is
  repaired and selected durable identity evidence passes. The exact
  child-enqueue-before-parent-disposition process-death seam remains owned by
  `elspeth-7cdc4da434`.
- **Concurrency: Unknown -> Partial.** The PostgreSQL different-member race
  now proves one committed expansion and one refusal. A complete parent/child
  contention corpus case remains owned by `elspeth-ef29ef6ba4`.

Contracts, runtime, and audit remain Partial with updated reasons and evidence.
No cell is promoted to Pass by analogy. Output-contract and journal fixes do
not directly complete any scenario row, so they change the hard-gate narrative
without changing a matrix status.

## Remaining hard gates and next proof

1. Execute source-ingress/source-completion and child-enqueue process-death
   seams (`elspeth-aafba3b298`, `elspeth-7cdc4da434`).
2. Prove registered orchestration, long-plugin lease loss, and real plugin
   disposition (`elspeth-9a52eb80f9`, `elspeth-51a4b5c771`,
   `elspeth-2e66723070`, `elspeth-6f6bbbec00`).
3. Close the remaining graph-config identity, persistence, export, and
   diagnostic secret paths.
4. Deliver guided/freeform/browser parity and decide the row-union contract.
5. Complete the scenario corpus, scale envelope, and CI-bound normative
   contract.

The most direct next DAG remediation remains the P1 crash sequence: source
completion, then child enqueue to parent disposition. The maintained corpus
continues as the acceptance spine rather than treating the repaired local
invariants as end-to-end completion.
