# L2 #4 — `plugins/` cluster discovery findings

## Cluster scope and census

`src/elspeth/plugins/` is the **L3 plugin ecosystem** — system-owned (NOT user-extensible) Sources, Transforms, Sinks, plus the shared `infrastructure/` spine providing pluggy hookspecs, audited clients, base classes, batching/pooling, and configuration scaffolding. [CITES KNOW-A16] [CITES KNOW-A35] [CITES KNOW-C21] [CITES KNOW-C22] [CITES KNOW-P2]

Per L1 §4: 98 files, 30,399 LOC; the largest subsystem in the codebase. Composite at L1 (Δ4 fires on all three: ≥4 sub-packages, ≥10k LOC, ≥20 files).

This pass supplements `02-l1-subsystem-map.md §4 plugins/` at one level deeper (sub-subsystem). Two of the four sub-subsystems (`infrastructure/`, `transforms/`) themselves qualify as composite at L2 depth and are flagged accordingly per Δ L2-3; depth-cap holds.

### Sub-subsystem inventory (verified by `find … -name '*.py'` and `wc -l`)

| Sub-subsystem | Files | LOC | Composite-at-L2? | L3 deep-dive flags |
|---|---:|---:|---|---|
| `infrastructure/` | 41 | 10,782 | **Yes** (≥10k LOC AND ≥20 files AND 3 sub-pkgs `batching/`, `clients/`, `pooling/` + nested `clients/retrieval/`) | None (largest file `base.py` at 1,159 LOC, under 1,500 threshold) |
| `sources/` | 8 | 3,519 | No (flat layout) | None |
| `transforms/` | 41 | 12,575 | **Yes** (≥10k LOC AND ≥20 files AND 3 sub-pkgs `azure/`, `llm/`, `rag/` + nested `llm/providers/`) | `transforms/llm/azure_batch.py` (1,592 LOC) |
| `sinks/` | 7 | 3,515 | No (flat layout) | None |
| **Total** | **97** | **30,391** | — | 1 |

(Discrepancy with L1 §4 "98 files / 30,399 LOC": L1 counted `__pycache__/` artefacts; this pass excludes them. The 1-file / 8-LOC delta corresponds to `plugins/__init__.py` plus pycache exclusions and is non-substantive.)

## Plugin count (per cluster priority 5)

**Counting method:** `grep -rE "^    name\s*=" src/elspeth/plugins --include='*.py'` for `name = "..."` class attributes — the registration key used by `infrastructure/discovery.py` and `manager.py` to register a class as a discoverable plugin.

**Result: 29 distinct registered plugins.**

| Type | Count | Names |
|---|---:|---|
| Sources | 6 | `csv`, `json`, `azure_blob`, `dataverse`, `text`, `null` |
| Transforms | 17 | `passthrough`, `value_transform`, `type_coerce`, `field_mapper`, `keyword_filter`, `truncate`, `line_explode`, `json_explode`, `batch_replicate`, `batch_stats`, `web_scrape`, `llm`, `azure_batch_llm`, `openrouter_batch_llm`, `rag_retrieval`, `azure_content_safety`, `azure_prompt_shield` |
| Sinks | 6 | `csv`, `json`, `azure_blob`, `dataverse`, `database`, `chroma_sink` |

**Drift recorded (per Δ L2-3 priority 5; not resolved):**

- [DIVERGES FROM KNOW-A35] ARCHITECTURE.md §3.3 says "25 plugins"; observed count is **29** (drift +4). The 4-plugin difference plausibly reflects post-doc growth (RAG, prompt shield, content safety, openrouter_batch_llm).
- [DIVERGES FROM KNOW-A72] ARCHITECTURE.md Summary says "46 plugins"; observed count is **29** (drift -17). KNOW-A72 is unsourced and inconsistent with the per-category enumeration in KNOW-A35 / §3.3 — flagging without resolving, per the L1 dispatch instruction "do not resolve, just flag."

Doc-correctness pass is out of scope for this L2 archaeology.

## L3 oracle status (Δ L2-2)

`temp/intra-cluster-edges.json` (filtered from `temp/l3-import-graph.json`):

```
intra_edge_count:        23
intra_node_count:        12
inbound_cross_cluster:   7  (total weight 43)
outbound_cross_cluster:  0  (plugins/ is a sink in the L3↔L3 graph)
sccs_touching_cluster:   1  (SCC #1)
```

### Intra-cluster edges by weight (top 10)

| Source | Target | Weight | Pattern |
|---|---|---:|---|
| `plugins/sinks` | `plugins/infrastructure` | 45 | Heaviest single L3 edge in the entire codebase (F3) |
| `plugins/transforms` | `plugins/infrastructure` | 40 | F3 |
| `plugins/sources` | `plugins/infrastructure` | 17 | F3 |
| `plugins/transforms/llm` | `plugins/infrastructure` | 17 | F3 |
| `plugins/transforms/llm/providers` | `plugins/infrastructure/clients` | 12 | LLM provider → audited LLM client |
| `plugins/transforms/llm` | `plugins/transforms/llm/providers` | 10 | **SCC #1 forward edge** |
| `plugins/transforms/rag` | `plugins/infrastructure/clients/retrieval` | 9 | RAG → retrieval client |
| `plugins/transforms/azure` | `plugins/infrastructure` | 6 | Azure safety → spine |
| `plugins/sources` | `plugins/infrastructure/clients` | 5 | Sources use audited clients (Dataverse) |
| `plugins/transforms/llm` | `plugins/transforms/llm/providers` | 5 | **SCC #1** (additional weight from `transform.py` registry) |

The infrastructure-as-spine pattern dominates: edges into `plugins/infrastructure` (and its sub-pkgs) account for **~85% of intra-cluster edge weight**.

### Inbound cross-cluster edges (top)

| Source (other cluster) | Target (plugin path) | Weight |
|---|---|---:|
| `web/composer` | `plugins/infrastructure` | 22 |
| `.` (cli root) | `plugins/infrastructure` | 7 |
| `testing` | `plugins/infrastructure` | 4 |
| `web/execution` | `plugins/infrastructure` | 4 |
| `web/catalog` | `plugins/infrastructure` | 3 |
| `.` (cli root) | `plugins/sources` | 2 |
| `web` | `plugins/infrastructure` | 1 |

All 7 inbound cross-cluster edges target `plugins/infrastructure` or `plugins/sources` directly; no other cluster reaches into `transforms/` or `sinks/` at L3↔L3 granularity. (Engine/contracts/core consume plugins via lower-layer protocols, not in the L3↔L3 graph.)

### Outbound cross-cluster L3↔L3 edges: 0

`plugins/` is a **sink in the L3 import graph** — it depends downward on `{contracts, core, engine}` (which are L0/L1/L2 and not in this graph) but has no L3↔L3 outbound edges. This is architecturally correct: plugins are the leaves of the application layer.

## SCC #1 evidence

Single SCC touching the cluster (per `temp/intra-cluster-edges.json:sccs_touching_cluster`):

```json
{"index": 1, "members": ["plugins/transforms/llm", "plugins/transforms/llm/providers"]}
```

**Import-level diagnosis (file:line):**

- Forward edge `llm → llm/providers`:
  - `transforms/llm/transform.py:64` — `from elspeth.plugins.transforms.llm.providers.azure import AzureLLMProvider, AzureOpenAIConfig, _configure_azure_monitor`
  - `transforms/llm/transform.py:65` — `from elspeth.plugins.transforms.llm.providers.openrouter import OpenRouterConfig, OpenRouterLLMProvider`
- Reverse edge `llm/providers → llm`:
  - `transforms/llm/providers/azure.py:23` — `from elspeth.plugins.transforms.llm.base import LLMConfig`
  - `transforms/llm/providers/azure.py:24` — `from elspeth.plugins.transforms.llm.provider import FinishReason, LLMQueryResult, parse_finish_reason`
  - `transforms/llm/providers/azure.py:25` — `from elspeth.plugins.transforms.llm.tracing import AzureAITracingConfig, TracingConfig`
  - `transforms/llm/providers/openrouter.py:35-37` — same pattern: imports `LLMConfig`, `LLMQueryResult`, `parse_finish_reason`, `reject_nonfinite_constant`

**Structural reading:** the **LLM provider-registry pattern**. `transform.py` aggregates concrete providers at module load; providers depend on shared base classes (`LLMConfig`), the protocol (`LLMProvider`, `LLMQueryResult`), and shared utilities (`validation`, `tracing`). The cycle is **module-level**, not class-level — neither side imports the other's class hierarchy via attribute access at runtime.

**Provider lifecycle:** `transforms/llm/transform.py:9-13` documents that "Provider instantiation is deferred to on_start() when recorder/telemetry become available. __init__ stores provider_cls + config for later use." So the *runtime* coupling is deferred even though the *import-time* coupling forms the cycle. This distinction matters for any future decomposition discussion (out of scope here per Δ L2-7).

Detailed cycle-intent and surfacing analysis: see `04-cluster-report.md §SCC analysis`.

## Layer conformance status (Δ L2-6 verdict)

- **Authoritative whole-tree check:** `enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model` → **clean**: "No bug-hiding patterns detected. Check passed." The Phase 0 oracle (`temp/tier-model-oracle.txt`) records this clean state.
- **Engine-scoped check** (`--root src/elspeth/plugins`): exit 1, 291 findings — but these are R-rule **defensive-pattern findings**, not L1 layer-import violations. Counts: R1=66, R2=6, R4=15, R5=140 (`isinstance`), R6=52 (silent-except), R8=3, R9=9. The exit-1 is two-fold: (a) allowlist-prefix mismatch (allowlist keys are project-root-relative; scoped paths drop the `plugins/` prefix), and (b) defensive-pattern findings are reported regardless of allowlist match.
- **Determinism re-derivation (Δ L2-6 dump-edges):** cluster-scoped `dump-edges --root src/elspeth/plugins` produces 0 edges by tool design (only emits inter-subsystem L3↔L3 edges; the entire scope collapses to a single subsystem). The substantive determinism contract was verified by re-running `dump-edges` on the whole tree and filtering to plugin nodes — **23 plugin intra-edges, byte-equivalent with the Δ L2-2 filter** modulo `samples` and ordering. SCC #1 reproduced identically.

**Verdict:** plugins/ is layer-conformant. The cluster-scoped oracle output requires interpretation but contains no actual layer-import violations.

## Test integration (per Δ L2-5)

The plugin test suite is substantial: **164 unit tests** under `tests/unit/plugins/` and **19 integration tests** under `tests/integration/plugins/`. Layout mirrors source:

- `tests/unit/plugins/{sources,sinks,transforms,llm,batching,clients,config,infrastructure,pooling}/`
- `tests/unit/plugins/test_base.py`, `test_base_signatures.py`, `test_base_sink_contract.py`, `test_base_source_contract.py` — base-class invariants
- `tests/unit/plugins/test_discovery.py`, `test_manager.py`, `test_manager_singleton.py`, `test_hookimpl_registration.py` — pluggy + discovery contracts
- `tests/unit/plugins/test_invariant_probe_execution.py`, `test_post_init_validations.py` — invariant assertions
- `tests/unit/plugins/test_builtin_plugin_metadata.py` — plugin metadata
- `tests/unit/plugins/llm/test_provider_protocol.py`, `test_provider_azure.py`, `test_provider_openrouter.py`, `test_provider_lifecycle.py`, `test_plugin_registration.py` — SCC #1 surface tests
- `tests/property/plugins/test_schema_coercion_properties.py` — coercion behaviour at the source boundary (Tier 3)

This pass cites tests selectively as evidence for invariant claims; a complete test catalog is out of scope (deferred per the L1 standing note).

## Reading order applied (F3)

Executed in F3-mandated order: `infrastructure/` → `sinks/` → `sources/` → `transforms/`. The catalog (`02-cluster-catalog.md`) presents entries in this order; client-side entries cite the spine without re-deriving its responsibilities.

## Highest-confidence preliminary findings

1. **Layer conformance is intact.** plugins/ has 0 upward L1 import violations (whole-tree check clean). All 7 inbound cross-cluster L3↔L3 edges and 23 intra-cluster edges are layer-permitted.
2. **F3 spine pattern is empirically dominant.** Edges into `plugins/infrastructure` account for ~85% of intra-cluster edge weight; reading-order discipline is structural, not preferential.
3. **Trust-tier discipline is documented in every plugin file.** Sources state `allow_coercion=True` ("ONLY place coercion allowed"); sinks/transforms state `allow_coercion=False` ("Wrong types = upstream bug = crash"). The contract is repeated as docstring text in every plugin module — strong cultural signal beyond the schema-level enforcement.
4. **SCC #1 is structurally explicable.** Provider-registry pattern with deferred runtime instantiation; cycle is module-level only.

## Highest-uncertainty preliminary questions

1. **Whether the 29-vs-25 plugin-count drift is doc rot or genuine architectural change.** Out of scope here; flagged for doc-correctness pass.
2. **Whether the SCC #1 cycle is load-bearing.** The deferred-instantiation pattern (`transform.py:9-13`) suggests intent; whether decomposition would cost more than it gains is for the architecture pack.
3. **Whether all 7 cross-cluster inbound edges are well-typed at the import boundary.** F5 (~97% of L3 edges are unconditional runtime coupling) means these are real hard imports; no probe of whether `web/composer → plugins/infrastructure (w=22)` represents one big surface or many small ones at this depth.
