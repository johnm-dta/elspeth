# 01 — engine/ cluster discovery (L2 holistic scan)

## Cluster shape

The engine cluster is **36 .py files, 17,425 LOC**, distributed in a strongly bimodal shape: two structured sub-packages (`orchestrator/`, `executors/`) account for **10,723 LOC (~62%)**, three standalone L3-deep-dive-flagged files (`processor.py`, `coalesce_executor.py`, plus `orchestrator/core.py` inside the sub-package) account for **7,584 LOC (~43%)**, and the remaining nine standalone modules are short utilities (91–399 LOC each, 1,995 LOC combined, ~11%). LOC concentration is severe: three files alone hold ~43% of the cluster, exactly as flagged in `04-l1-summary.md` §7 Priority 1 and KNOW-A70.

Sub-areas sorted by LOC (verified inventory):

| LOC | Sub-area | Files |
|---:|---|---:|
| 5,550 | `engine/executors/` | 16 |
| 5,173 | `engine/orchestrator/` | 7 (incl. `core.py` 3,281 — flag) |
| 2,700 | `engine/processor.py` | 1 (flag) |
| 1,603 | `engine/coalesce_executor.py` | 1 (flag) |
| 399 | `engine/tokens.py` | 1 |
| 324 | `engine/triggers.py` | 1 |
| 316 | `engine/dag_navigator.py` | 1 |
| 284 | `engine/batch_adapter.py` | 1 |
| 279 | `engine/spans.py` | 1 |
| 173 | `engine/dependency_resolver.py` | 1 |
| 138 | `engine/bootstrap.py` | 1 |
| 137 | `engine/commencement.py` | 1 |
| 137 | `engine/retry.py` | 1 |
| 121 | `engine/clock.py` | 1 |
| 91 | `engine/__init__.py` | 1 |

## Entry points and runtime surfaces

The engine ships no console_scripts of its own; it is invoked exclusively by `elspeth.cli` (CLI verbs `run`, `resume`, `validate` per KNOW-C34). Public symbols are re-exported through `engine/__init__.py` (91 lines, read in full). The `__all__` list is 25 names long and groups into:

- **Top-level façade classes:** `Orchestrator` (run lifecycle), `RowProcessor` (per-row DAG traversal), `CoalesceExecutor` (fork/join merge barrier), `TokenManager` (token identity), `RetryManager` (tenacity-backed retries), `SpanFactory` (OpenTelemetry spans).
- **Executor classes:** `AggregationExecutor`, `GateExecutor`, `SinkExecutor`, `TransformExecutor` (re-exported from `engine/executors/`).
- **Result/config dataclasses:** `PipelineConfig`, `RunResult`, `RowPlugin`, `RowResult`, `TokenInfo`, `RuntimeRetryConfig`, `AggregationFlushResult`, `ExecutionCounters`, `CoalesceOutcome`.
- **Errors:** `MaxRetriesExceeded`, `MissingEdgeError`, `RouteValidationError`.
- **Re-exports from L1 `core/`:** `ExpressionParser`, `ExpressionSecurityError`, `ExpressionSyntaxError` (these live at `core/expression_parser.py` per KNOW-A27 — surfaced through engine for plugin convenience).

The example in the module docstring (lines 10–32) confirms the canonical run shape: build `ExecutionGraph.from_plugin_instances(...)`, instantiate `Orchestrator(db)`, call `orchestrator.run(config, graph=..., payload_store=...)`. This matches the test-integrity rule in KNOW-C44.

## SDA model orientation

The SDA model (KNOW-C26: SENSE → DECIDE → ACT) maps onto engine sub-areas at runtime: Sources and Sinks are L3 plugins, but their **execution mechanics** live here. `Orchestrator` (orchestrator/, 5,173 LOC) coordinates the full run lifecycle — source pull, DAG traversal driver, run-level Landscape recording, ArtifactPipeline. `RowProcessor` (processor.py, 2,700 LOC) processes a single row through its DAG path — i.e. the per-token loop calling executors in sequence. `executors/` (5,550 LOC across 16 files: aggregation, gate, transform, sink, pass_through, declaration_*, schema_config_mode, source_guaranteed_fields, etc.) handles the per-step DECIDE mechanics including ADR-007/008/009/010 contract enforcement (KNOW-ADR-007–010). `CoalesceExecutor` (coalesce_executor.py, 1,603 LOC) is the merge barrier where parallel fork-paths join (KNOW-A28, KNOW-C30). Token identity (KNOW-A46, KNOW-C29) — `row_id`, `token_id`, `parent_token_id` — is the lineage spine that survives forks/joins, and the engine is responsible for its ~exact-once-terminal invariant (KNOW-A44, KNOW-C43). See ARCHITECTURE.md §3.1 Engine Components and CLAUDE.md "DAG Execution Model" for the canonical statement.

## Layer position and dependency surface

engine/ is **L2** (per `scripts/cicd/enforce_tier_model.py:239` `"engine": 2`, also KNOW-C47). It may import only from `{contracts, core}` (L0+L1). The Phase 0 L3↔L3 oracle filtered to engine/ is necessarily empty — `temp/intra-cluster-edges.json` reports `intra_node_count = 0`, `intra_edge_count = 0`, `inbound_edge_count = 0`, `outbound_edge_count = 0`. **This is expected and correct, not a gap**: the source oracle filters L3-only and engine is L2; its downward edges to `core/` and `contracts/` are layer-permitted by construction and not graph-enumerated by Phase 0. The cluster catalog wave will record engine→core / engine→contracts edges at sub-subsystem granularity from imports as they become relevant; cross-cluster oracle data is not the right tool here.

## Layer conformance status

Two artefacts in `temp/`:

- `layer-check-engine.txt` — scoped run with the production allowlist (`config/cicd/enforce_tier_model/`).
- `layer-check-engine-empty-allowlist.txt` — re-run against an empty allowlist to isolate genuine layer-import findings from already-allowlisted defensive patterns.

The empty-allowlist run reports **CHECK FAILED** with **69 already-allowlisted defensive-pattern findings re-surfaced** (rules R1/R4/R5/R6/R9 — `isinstance`, `getattr`, `hasattr`, silent except, etc.). The scoped exit-1 status comes from these allowlist key prefixes not matching when the scope narrows below the allowlist's whole-tree key prefix; it is **not** a layer-import failure. The load-bearing claim — verified by `grep -c '^Rule: L1'` and `grep -c '^Rule: TC'` against the empty-allowlist file — is:

> **0 L1 layer-import violations, 0 TC TYPE_CHECKING layer warnings inside engine/.**

Interpretation: engine/ is fully layer-conformant with respect to upward imports. The 69 defensive-pattern findings are all in-cluster code-quality items already governed by per-file allowlists with owner/reason/safety annotations; the catalog wave inherits them as known and does not need to re-triage.

## Test surface

`find tests/unit/engine -maxdepth 2 -type d` yields:

- `tests/unit/engine/`
- `tests/unit/engine/orchestrator/`

Test count: `find tests/unit/engine -name 'test_*.py' | wc -l` → **56**. There is **no `tests/integration/engine/` directory** (`ls tests/integration/ | grep -i engine` returns nothing); engine integration coverage lives elsewhere in the integration tree and is in scope for the catalog wave to locate.

**Standing note (KNOW-C44 / CLAUDE.md "Critical Implementation Patterns"):** integration tests MUST use `ExecutionGraph.from_plugin_instances()` and `instantiate_plugins_from_config()` and never bypass production code paths. The catalog wave should spot-check whether engine tests honour this rule, particularly any test that constructs `Orchestrator`, `RowProcessor`, or `CoalesceExecutor` directly without going through the production graph builder.

## Open questions for the catalog wave

1. **Terminal-state invariant enforcement.** Where in the engine is "every row reaches exactly one terminal state" (KNOW-A44, KNOW-C43) actually enforced and tested? Likely candidates: `processor.py`, `tokens.py`, `coalesce_executor.py` — the catalog wave should locate the choke-point and the corresponding test(s).
2. **`processor.py` cohesion (Q5 from §7.5).** Does `processor.py` (2,700 LOC) own one cohesive responsibility (RowProcessor end-to-end) or has it accreted multiple concerns? This is the L3 deep-dive question deferred from L1.
3. **Token-identity ownership boundary.** How is token identity (`row_id`/`token_id`/`parent_token_id`, KNOW-A46) split between `tokens.py` (399 LOC, `TokenManager` re-exported) and `orchestrator/`? Is `TokenManager` the sole authority, or do the orchestrator and processor mint tokens directly?
4. **ADR-009/010 dispatch sites.** ADR-009 collapses the duplicated walkers and ADR-010 nominates 4 dispatch sites with audit-complete dispatch (per the project_adr010_dispatch_shape memory). Which engine files own each of the 4 sites? `executors/pass_through.py` is named in KNOW-ADR-009b; the other three need explicit attribution.
5. **`orchestrator/core.py` decomposability.** Does the 3,281-LOC `orchestrator/core.py` factor into the other six files in `orchestrator/` (`aggregation.py`, `export.py`, `outcomes.py`, `types.py`, `validation.py`, `__init__.py`) cleanly, or is `core.py` a god-class with the others as thin helpers?
6. **`coalesce_executor.py` policy surface.** KNOW-A28 says CoalesceExecutor provides "policy-driven merging". What is the policy enumeration, where are policies declared, and how do they interact with the terminal-state invariant on the joined token?

## L1 cross-references

Supplements 02-l1-subsystem-map.md §3 (engine/) — adds sub-subsystem inventory and entry-point identification.

Supplements 04-l1-summary.md §7 Priority 1 — confirms the 3-file 1,500+ LOC concentration list (`orchestrator/core.py` 3,281, `processor.py` 2,700, `coalesce_executor.py` 1,603 = ~43% of engine LOC) and ratifies engine as the L2 priority-1 cluster.

Standing note from §7.5: ~97% of L3 edges are unconditional runtime coupling — the engine cluster is L2, so the standing note applies to engine's L3 callers (plugins/web/cli/mcp/tui/testing) when they're analysed in their own clusters; it does not relax suspicion of conditional/TYPE_CHECKING coupling **inside** engine, which is governed instead by the layer-check artefacts above (0 TC warnings).
