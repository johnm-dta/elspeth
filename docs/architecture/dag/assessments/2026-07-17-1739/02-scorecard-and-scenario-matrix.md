# DAG completeness scorecard and scenario matrix

**Baseline:** `release/0.7.1` at `6e8a6bf5f2f8542bf5b95b1669ce3d3df68d93e3`
**Verdict:** **Not complete**
**Normalized maturity indicator:** **Not calculated** — mandatory dimensions
remain `U`, and the framework prohibits an aggregate in that state.

## Executive assessment

Elspeth's graph and production builder are structurally capable. The exact
nested `fork -> coalesce -> fork -> coalesce` builder regression now passes,
queue-aware configuration is real, structural validation remains strong, and
the scheduler subtype failures from the seed assessment are closed.

The whole DAG product is still not complete. Reproduced secret-bearing graph
identity and audit paths remain open; replay-safe expansion, output-contract
serialization, and sidecar atomicity have open defects; the crash and
registered-process matrices are incomplete; guided authoring cannot express
mandatory non-linear graphs; all six seeded browser correctness cases are
skipped; and the normative execution-graph contract materially contradicts
current source.

## Fifteen-dimension scorecard

Scores follow the evergreen 0-5 scale and use the lowest applicable mandatory
evidence in each dimension. `U` is retained wherever the mandatory scope still
contains Unknown evidence; numeric scores in other rows are local indicators,
not an aggregate.

| Dimension | Score | Status | Current evidence and limiting gate |
|---|---:|---|---|
| Topology expressiveness | 0 | Fail | Seven node types, plural sources, queue, aggregation, and coalesce build through production, but expansion is not first-class topology metadata and row union remains unsupported/undecided rather than consistently rejected by contract. |
| Canonical configuration | U | Unknown | Deterministic node/topology identity passes, but no mandatory cross-input/cross-surface canonical graph matrix exists. |
| Structural validation | 4 | Pass | Production and graph paths fail closed on cycles, roots/terminals, reachability, duplicate labels, illegal fan-in, routes, and coalesce definitions. |
| Schema contracts | 0 | Fail | Edge/output checks and graph-order pass-through propagation pass, but mandatory composition and plugin-declaration matrices remain incomplete and the row-union support/rejection contract is unresolved. |
| Cardinality and identity | 2 | Partial | Happy-path batch/expansion identity and selected contention pass; `elspeth-a25e9c009e` and the documented child-remint replay boundary remain open. |
| Compositional closure | U | Unknown | Exact sequential nested coalesces now build; equivalent production/runtime evidence for parallel and all policy combinations is Unknown. |
| Runtime semantics | U | Unknown | Disposition and selected production-drain paths pass; PB-02, PB-03, PB-08, nested/parallel compositions, and other mandatory runtime cells remain Unknown. |
| Durable recovery | U | Unknown | Selected lease, aggregation, barrier, pending-sink, and sink-effect restart seams pass, while the expansion replay defect is open and source, parent/child, whole-row, and process-death cells remain Unknown. |
| Concurrency and fencing | U | Unknown | Direct processes, strict initial claims, stale leaders, and selected reclaim paths pass; registered orchestration, pending-sink/barrier contention, and long-plugin behavior remain Unknown. |
| Atomic evidence | U | Unknown | TS-07-10 rollback and selected scheduler/barrier faults pass; queue/ingress and parent-child cells remain Unknown, while output-contract and journal atomicity defects are open. |
| Security | 1 | Fail | Raw plugin config affects node identity/topology hash and reproduced run/config/URL/gate-condition paths reach audit/export. Seven security/integrity issues remain open. |
| Authoring parity | 0 | Fail | Guided authoring remains a transform-chain surface. The 27-case parity matrix is absent and all six seeded browser correctness tests are skipped. |
| Semantic round-trip | U | Unknown | Queue/gate/aggregation/coalesce YAML helpers pass; no canonical before/after equality matrix spans mandatory fixtures and all authoring surfaces. |
| Scale | U | Unknown | Functional deep/wide/row-volume slices pass, but mandatory scenario limits remain Unknown and no supported envelope, observable threshold, required CI gate, or named owner exists. |
| Maintained contract | 1 | Fail | The `FINAL` contract still says six node types, exactly one source, no queue, and a singular-source builder. Required repository gates are also red. |

No calculation is permitted: eight dimensions remain `U`. Reporting the mean
of only the numeric rows would overstate certainty and violate the framework.

The seed assessment's 2.4/5 value used a different 11-dimension method. The two
numbers must not be interpreted as a trend or percentage change.

## Mandatory scenario matrix

Legend: **P** Pass, **△** Partial, **F** Fail, **U** Unknown, **N/A** not
applicable with the reason described below.

| Scenario | Config | Build | Contracts | Runtime | Audit | Recovery | Concurrency | Freeform | Guided | Round-trip | Scale | Overall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Linear source -> transform -> sink | P | P | P | △ | △ | △ | U | P | △ | △ | △ | △ |
| Multiple independent sources | P | P | P | △ | △ | △ | U | P | F | △ | U | F |
| Multi-source queue fan-in | P | P | P | △ | △ | U | U | P | F | △ | U | F |
| Conditional routing | P | P | P | △ | △ | U | U | P | F | △ | U | F |
| Fork to multiple terminals with partial failure | P | P | P | △ | △ | U | U | P | F | U | U | F |
| Fork/coalesce policies and merge strategies | P | P | △ | △ | △ | △ | △ | P | F | △ | U | F |
| Sequential/nested forks and coalesces | P | P | △ | U | U | U | U | P | F | U | U | F |
| Parallel coalesces | P | △ | △ | U | U | U | U | P | F | U | U | F |
| Aggregation and immutable batch closure | P | P | △ | △ | △ | △ | U | P | F | U | U | F |
| Row expansion and parent/child recovery | P | P | △ | △ | △ | F | U | P | F | U | U | F |
| Row union/interleave | F | F | F | F | N/A | N/A | N/A | F | F | N/A | N/A | F |
| Retry/quarantine/discard/routed errors | P | P | △ | △ | △ | U | U | P | F | △ | U | F |
| Sink write and pending-sink redrive | P | P | P | △ | △ | △ | △ | P | △ | △ | U | △ |
| Checkpoint and deterministic resume | P | P | △ | △ | △ | △ | U | N/A | N/A | N/A | U | △ |
| Multi-worker execution and late completion | N/A | P | △ | △ | △ | △ | △ | N/A | N/A | N/A | U | △ |

`N/A` is narrow:

- row union has no supported construct, so post-build audit/recovery/concurrency
  do not apply after configuration, build, contract, and runtime already fail;
- checkpoint/resume is a runtime lifecycle, not an authored topology; and
- worker multiplicity is deployment/runtime configuration, not DAG authoring.

Security is cross-cutting rather than repeated as a matrix column. Every
credential-bearing source, transform, or sink scenario remains subject to the
reproduced graph-identity and audit/export hard gate.

No mandatory scenario reaches production support across all applicable cells.

## Hard-gate verdict

The verdict is **Not complete** because all of the following remain true:

- expansion replay can duplicate or ambiguously remint effective work;
- output-contract and sidecar-journal writes have open atomicity defects;
- raw secrets can affect graph identity or reach persisted/exported evidence;
- registered multi-process crash/reclaim/late-worker behavior is incomplete;
- guided authoring cannot represent mandatory supported non-linear topology;
- mandatory browser, recovery, contention, round-trip, and scale cells are
  skipped, Unknown, or plan-only; and
- the normative execution-graph contract materially contradicts live code.
