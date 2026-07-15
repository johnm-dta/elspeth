# Core DAG completeness evidence

Assessment date: 2026-07-15
Indexed revision: `0dcd61acaa44082d93ec205683700e798748ee6d`
Scope: the canonical node model, `ExecutionGraph`, `build_execution_graph()`, schema propagation, traversal metadata, and focused topology evidence. Runtime scheduling, recovery, authoring UX, and public documentation are assessed by the other workstreams.

## Verdict

The core DAG is **substantially implemented but not complete**. The graph model and builder cover the seven canonical node types, plural sources, explicit queue fan-in, conditional routing, forks, aggregations, coalesces, and sinks. Structural validation and policy-aware schema handling are unusually strong. The remaining risks are concentrated at composition boundaries rather than in basic DAG representation.

Using the shared 0–5 scale (`0` unsupported, `1` modeled, `2` compiles, `3` happy-path, `4` production-supported, `5` maintained contract), the core layer is approximately **3/5 overall**:

| Dimension | Score | Evidence-backed assessment |
|---|---:|---|
| Topology model | 4 | Seven node types are canonical; multiple sources and explicit queue/coalesce joins are represented. |
| Structural validation | 4 | Cycles, missing roots/sinks, unreachable nodes, implicit ordinary-node fan-in, duplicate outgoing labels, and incomplete gate-to-sink route maps fail closed. |
| Schema contracts | 3 | Edge and coalesce schema validation are implemented, but known composition and plugin-schema gaps prevent a production-support verdict. |
| Compositional closure | 2 | Sequential/parallel complex shapes are structurally modeled, but the key fork→coalesce→fork shape is not proven through the real builder and has a live failure report. |
| Traversal metadata | 3 | Step/branch maps exist and are tested, but compilation is split across core DAG and engine wiring. |
| Security of graph metadata | 2 | A confirmed open issue reports unredacted plugin configuration copied into DAG metadata. |
| Maintained contract evidence | 3 | Broad property and focused tests exist, but no canonical real-builder scenario matrix closes every advertised composition. |

The core cannot be called complete while either of these hard gates remains open:

1. An advertised composition can validate structurally yet fail in real builder schema propagation.
2. Runtime secrets may enter graph/audit metadata without a redacted public-config contract.

## Implemented capability inventory

### Canonical node and join semantics

- `NodeType` defines `SOURCE`, `QUEUE`, `TRANSFORM`, `GATE`, `AGGREGATION`, `COALESCE`, and `SINK` (`src/elspeth/contracts/enums.py:91-103`).
- `build_execution_graph()` takes plural `sources` and `source_settings_map`, plus transforms, sinks, aggregations, gates, coalesces, and queues (`src/elspeth/core/dag/builder.py:158-169`). It rejects mismatched source/settings names (`:184-190`).
- Queue nodes are explicitly coordination-only. `QueueSettings` states that queues do not merge row data, join schemas, or change token identity (`src/elspeth/core/config.py:1076-1088`). The builder gives them observed output schema and wires all declared producers into the queue (`builder.py:315-331`, `:626-760`).
- Coalesce is the row-sibling join primitive. Configuration supports `require_all`, `quorum`, `best_effort`, and `first`; `union`, `nested`, and `select`; collision policies; timeouts; and quorum/select validation (`src/elspeth/core/config.py:846-1044`).
- Coalesce schema materialization is centralized in `merge_coalesce_schema()`. It handles policy-aware guarantees, collision policy, audit-field union, selected-branch schemas, and optional nested branches (`src/elspeth/core/dag/coalesce_merge.py:50-107`).

### Structural correctness

`ExecutionGraph.validate()` checks (`src/elspeth/core/dag/graph.py:232-378`):

- acyclicity;
- one or more source roots and at least one sink;
- reachability of every node from at least one source;
- explicit queue fan-in for ordinary executable nodes;
- unique outgoing edge labels per node;
- complete route-label metadata for direct gate-to-sink `MOVE` edges.

Direct multi-source fan-in is intentionally allowed only at queue/coalesce structural nodes and terminal sinks (`graph.py:289-309`). This is a crisp, fail-closed distinction between independent source branches, scheduling fan-in, sibling row merge, and terminal writes.

The builder then materializes coalesce schemas, performs phase-two edge compatibility validation, records coalesce warnings, builds the step map, and freezes graph metadata (`builder.py:1013-1103`). `ExecutionGraph.validate_edge_compatibility()` delegates to the dedicated schema validator for edges, coalesce branches, and required sink fields (`graph.py:1073-1080`).

## Scenario evidence matrix

| Scenario | Core verdict | Evidence | Remaining gap |
|---|---|---|---|
| Linear source→transform→sink | Supported (4) | Property generation and graph consistency coverage in `tests/property/core/test_dag_properties.py`. | None material in core. |
| Multiple independent source roots | Supported (4) | `test_graph_allows_multiple_source_roots_when_reachable` and production-path `test_two_independent_source_branches_end_to_end` in `tests/unit/core/test_multi_source_foundation.py`. | Runtime/recovery evidence belongs to other workstreams. |
| Multi-source fan-in through queue | Supported (4) | Builder queue wiring plus `TestComposerRuntimeQueueAgreement.test_queue_round_trips_composer_import_export_and_runtime_graph` (`tests/integration/pipeline/test_composer_runtime_agreement.py:4157-4195`). | Queue is deliberately not a row/schema union. |
| Fan-in to ordinary node without queue | Rejected correctly (4) | Graph invariant and `test_graph_rejects_fan_in_without_queue` (`tests/unit/core/test_multi_source_foundation.py:415-426`). | None. |
| Single fork/coalesce | Supported (4) | Real-builder coalesce schema tests, property audit token accounting, merged-data and lineage tests. | Recovery/concurrency assessed elsewhere. |
| N-branch coalesce policies and schema guarantees | Supported (4) | `TestCoalesceNBranchProperties`, `TestCoalesceFieldOptionalityProperties`, and builder end-to-end optionality tests. | Needs inclusion in one canonical scenario suite. |
| Sequential fork→coalesce→fork→coalesce | **Unproven (2)** | `sequential_fork_pipelines` models the topology, but manually calls `ExecutionGraph.add_node/add_edge` (`tests/property/core/test_dag_complex_topologies.py:58-104`) and bypasses the real builder/schema pipeline. | Live issue `elspeth-a6ca0bef77` reports a possible `FrameworkBugError` when the second fork consumes coalesce output. |
| Parallel coalesces | Structurally supported (3) | Property topology in `tests/property/core/test_dag_complex_topologies.py:107-157`. | The property fixture also constructs the graph directly; add a real-builder and runtime fixture. |
| Aggregation in traversal graph | Partially supported (3) | Builder creates aggregation nodes and dynamic schemas (`builder.py:378-425`). | Verified issue `elspeth-a1d9c01bad`: non-batch-aware or duplicate transforms can claim aggregation node IDs during engine graph wiring. |
| Edge/schema compatibility | Partially supported (3) | Dedicated validator and coalesce materialization are present. | Confirmed plugin-schema defect `elspeth-bd432a86a7`; sequential coalesce-output propagation remains unproven. |

## Gaps to shore up

### C-DAG-01 — Close real-builder compositional closure (P1)

`elspeth-a6ca0bef77` describes an exact source→fork→coalesce→fork→coalesce→sink topology that the web model accepts but the real engine path may reject because the second fork gate has no `output_schema_config`. Existing sequential-fork property coverage is insufficient evidence because it constructs a bare `ExecutionGraph` and verifies only graph structure/topological ordering.

Shore-up action:

- Add a fixture that enters through canonical settings/plugin assembly and `ExecutionGraph.from_plugin_instances()`.
- Exercise observed and explicit schema modes.
- Assert graph construction, edge compatibility, coalesce materialized schemas, branch-first-node map, step map, and a minimal runtime execution.
- If the failure reproduces, fix schema propagation without weakening fail-closed edge validation.

Exit criterion: the exact live-issue topology passes through the real builder and runtime, or the product rejects it consistently at every authoring/validation boundary with a documented limitation.

### C-DAG-02 — Separate runtime secrets from graph/audit configuration (P1 hard gate)

Confirmed issue `elspeth-c4080bfb06` reports that source, sink, and transform plugin configs are copied into DAG node metadata without a redaction/allowlist contract. This is a completeness blocker because graph metadata is an audit and diagnostics surface; secret exposure cannot be scored as production-supported.

Shore-up action:

- Define a typed `public_graph_config()`/redacted-settings contract for every plugin class.
- Store secret handles or redacted placeholders, never runtime credential values, in `NodeInfo.config` and topology hashes.
- Add fail-closed tests for `password`, `token`, `api_key`, `secret`, and credential-bearing URLs across source, transform, and sink nodes.

Exit criterion: graph snapshots, audit export, debug serialization, and topology hashing demonstrably contain no runtime secret values.

### C-DAG-03 — Make aggregation identity fail closed (P2)

Verified issue `elspeth-a1d9c01bad` shows that traversal wiring can accept a non-batch-aware transform or duplicate transform claiming an aggregation node ID, after which `node_to_plugin` overwrites can desynchronize runtime traversal from the canonical graph.

Shore-up action:

- Reject duplicate transform node IDs before traversal-map construction.
- Require aggregation node IDs to bind exactly one batch-aware transform and the matching aggregation settings entry.
- Add negative tests for both invalid ownership cases.

Exit criterion: there is a one-to-one, type-correct binding from every aggregation node to its runtime plugin.

### C-DAG-04 — Establish one traversal compiler (P2 architectural risk)

Open issues `elspeth-d4e15aee36` and `elspeth-a2905d4964` identify two coupled risks: `build_execution_graph()` is a roughly 950-line phase script, and traversal compilation is split across `ExecutionGraph`, engine `graph_wiring`, and `DAGNavigator`. This does not itself prove incorrect behavior, but it makes every new topology a synchronized multi-module change and explains why structural tests can diverge from the real engine path.

Shore-up action:

- Keep the public builder facade, but split construction into typed phases: node specs, namespace/connection resolution, fork-coalesce plan, schema propagation, validation/finalization.
- Produce a single immutable traversal plan containing node successors, structural nodes, branch starts/endpoints, terminal sinks, and node-to-plugin binding requirements.
- Contract-test that plan against the canonical graph rather than recomputing it in engine components.

Exit criterion: topology changes have one compilation authority and one table-driven contract suite.

### C-DAG-05 — Seal schema-contract escape hatches (P2)

Confirmed issue `elspeth-bd432a86a7` shows blob transforms can carry guaranteed fields that are absent from their emitted field declarations. Although the defect is in plugin transforms, it crosses the DAG contract boundary and can make downstream validation reject or misinterpret an otherwise data-preserving pipeline.

Shore-up action:

- Require every synthesized explicit `SchemaConfig` to be internally valid at construction.
- Add a graph-level invariant/test that guaranteed fields are declared when explicit fields exist.
- Add observed-schema blob fixtures to the canonical topology suite.

Exit criterion: no plugin can inject an internally inconsistent schema object into graph propagation.

### C-DAG-06 — Replace scattered evidence with a canonical scenario suite (P2)

The repository has broad test volume, but evidence is split among direct-graph property tests, builder unit tests, audit property tests, composer/runtime agreement tests, and engine tests. Direct `ExecutionGraph` tests prove graph algorithms; they do not prove configuration, builder schema propagation, traversal compilation, or runtime agreement.

Shore-up action:

Create a table-driven suite where every mandatory scenario is exercised through:

1. canonical configuration parsing;
2. plugin/settings assembly;
3. real graph building;
4. structural and schema validation;
5. canonical graph/traversal-plan snapshot;
6. minimal runtime execution;
7. audit/recovery assertions where relevant.

At minimum include linear, independent multi-source, queue fan-in, conditional routing, fork-to-sinks, each coalesce policy/merge strategy, sequential forks, parallel coalesces, aggregation, row expansion, error routing, and checkpoint/replay variants.

Exit criterion: every mandatory scenario scores at least 4, with no unknowns and no test that substitutes a manually assembled graph for the production build path.

## Prioritized shore-up order

1. **Reproduce and close C-DAG-01**; it determines whether an advertised composition is real or merely modeled.
2. **Close C-DAG-02** before declaring graph/audit metadata production-safe.
3. **Fix C-DAG-03** so runtime plugin identity cannot diverge from graph identity.
4. **Build C-DAG-06** as the acceptance harness, including C-DAG-05 regressions.
5. **Refactor toward C-DAG-04** behind the new acceptance harness; do not refactor the builder first without those invariants pinned.

## Verification evidence and limitations

Loomweave was fresh for revision `0dcd61acaa44082d93ec205683700e798748ee6d` (51,337 entities, 120,614 edges; completed analysis run `388ad85e-f9ae-43ec-ba3b-253128dd2936`). Source spans and test identities above came from that live index. Filigree issue statuses were queried live.

Root-run evidence from this same assessment session, not rerun by this subagent:

```text
uv run pytest \
  tests/property/core/test_dag_properties.py \
  tests/property/core/test_dag_complex_topologies.py \
  tests/property/core/test_dag_step_map_properties.py -q

62 passed in 3.78s
```

This verifies the property suites at the session revision, but it does **not** clear C-DAG-01 because `test_dag_complex_topologies.py` uses directly assembled graphs for the sequential and parallel shapes.

No additional shell-dependent tests were run by this subagent: the shared execution service was returning `No file descriptors available (os error 24)`. The assessment therefore does not claim fresh execution of the integration, multi-source, schema-propagation, or audit tests named above.
