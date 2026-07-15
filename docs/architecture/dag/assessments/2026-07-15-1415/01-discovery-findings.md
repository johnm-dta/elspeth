# DAG Completeness Discovery Findings

## System boundary

The assessment treats the DAG as an end-to-end product rather than only an in-memory graph. Five responsibilities form the boundary:

| Surface | Primary locations | Responsibility |
| --- | --- | --- |
| Configuration and authoring | `src/elspeth/core/config.py`, `src/elspeth/web/composer/` | Express a supported topology without losing intent. |
| Graph model and compiler | `src/elspeth/core/dag/` | Build one canonical graph and reject structurally invalid configurations. |
| Runtime scheduling | `src/elspeth/engine/` | Execute graph semantics, including routing, joins, aggregation, expansion, and sink delivery. |
| Durable state and evidence | `src/elspeth/state/`, audit tests | Preserve execution state, recovery fences, and explainable outcomes. |
| Contracts and verification | `docs/contracts/`, `docs/architecture/`, `tests/`, `evals/` | Define and continuously prove supported behavior. |

## Architecture shape

Configuration and Composer surfaces feed the graph builder. The builder produces the execution graph consumed by traversal and scheduler components. Durable state tracks tokens, rows, routes, attempts, and sink delivery. Audit and documentation surfaces expose the resulting contract to operators and authors.

The assessment therefore distinguishes:

- **Modeled:** the graph has a node or edge representation.
- **Compiled:** supported configuration builds and validates.
- **Executed:** the runtime produces the correct happy-path result.
- **Recoverable:** failure, retry, crash, and contention preserve the same result.
- **Product-complete:** all supported authoring and documentation surfaces expose the capability and tests maintain it.

## Evidence hierarchy

1. Current executable tests and source at the recorded commit.
2. Current generated or hand-written contracts that match the source.
3. Live Filigree issue state and approved implementation plans.
4. Historical notes only when re-verified against current repository state.

Skipped tests, plans, and issue descriptions are evidence of intent or missing proof, not evidence that a capability works.

## Preliminary subsystem inventory

| Subsystem | Inbound dependencies | Outbound dependencies | Initial confidence |
| --- | --- | --- | --- |
| Config and plugin resolution | YAML/settings and Composer export | DAG builder, plugin registry | Medium |
| Graph compilation and validation | Resolved settings and plugins | Runtime traversal plan | High |
| Transform/gate/queue/coalesce/aggregation semantics | Graph nodes and row contracts | Scheduler and state store | Medium |
| Scheduler and durable state | Canonical graph, input rows | Plugins, sinks, audit trail | Medium |
| Composer freeform/guided authoring | User intent and forms | Runtime YAML/config | High |
| Contracts, tests, and evidence | All preceding surfaces | Maintainers and release gates | Medium |

## Questions assigned for evidence collection

- Which topology and composition patterns compile and validate today?
- Where do schema, cardinality, or plugin capability assumptions fail to compose?
- Which runtime paths have durable recovery and multi-worker proof?
- Which advertised capabilities are missing from guided authoring or round-trip verification?
- Which documentation claims are stale relative to the live graph model?
- Which gaps can cause data loss, duplication, unsafe replay, or secret disclosure?

## Known limitations at discovery time

- This is a focused DAG assessment, not a full repository architecture catalogue.
- Performance and scale receive an evidence-gap verdict unless a repeatable threshold test exists.
- External plugins are assessed through declared contracts and repository fixtures; the analysis does not exhaustively execute every third-party combination.
