# L2 #4 — `plugins/` cluster diagrams

These diagrams describe **only** the plugins/ cluster's internal structure and its directly-cited cross-cluster boundary. Per Δ L2-4, no claims are made about other clusters' internals.

---

## §1. C4 Container view — plugins/ cluster

Edge weights cite `temp/intra-cluster-edges.json`. Inbound cross-cluster edges cite the L3 oracle.

```mermaid
flowchart LR
    classDef cluster fill:#fef3c7,stroke:#92400e,stroke-width:2px,color:#000
    classDef other fill:#ffffff,stroke:#9ca3af,stroke-width:1px,stroke-dasharray:5 3,color:#000
    classDef spine fill:#fde68a,stroke:#b45309,stroke-width:3px,color:#000
    classDef scc fill:#fecaca,stroke:#b91c1c,stroke-width:2px,color:#000

    subgraph PLUGINS["plugins/ cluster (L3, scope: src/elspeth/plugins/)"]
      INFRA["plugins/infrastructure/<br/>41 files, 10,782 LOC<br/>L3 candidate: composite at L2"]:::spine
      SOURCES["plugins/sources/<br/>8 files, 3,519 LOC<br/>6 plugins (csv, json,<br/>azure_blob, dataverse,<br/>text, null)"]:::cluster
      SINKS["plugins/sinks/<br/>7 files, 3,515 LOC<br/>6 plugins (csv, json,<br/>azure_blob, dataverse,<br/>database, chroma_sink)"]:::cluster
      TRANSFORMS["plugins/transforms/<br/>41 files, 12,575 LOC<br/>17 plugins<br/>L3 candidate: composite at L2"]:::cluster
    end

    SINKS -->|w=45 heaviest L3 edge| INFRA
    TRANSFORMS -->|w=40| INFRA
    SOURCES -->|w=17| INFRA
    SOURCES -->|w=5 audited HTTP| INFRA

    %% External inbound (cited from L3 oracle, not modelled)
    EXT_WEB_COMPOSER["web/composer<br/>(other cluster)"]:::other
    EXT_CLI[". (cli root)<br/>(other cluster)"]:::other
    EXT_TEST["testing<br/>(other cluster)"]:::other
    EXT_WEB_EXEC["web/execution<br/>(other cluster)"]:::other
    EXT_WEB_CAT["web/catalog<br/>(other cluster)"]:::other

    EXT_WEB_COMPOSER -.->|w=22| INFRA
    EXT_CLI -.->|w=7| INFRA
    EXT_CLI -.->|w=2| SOURCES
    EXT_WEB_EXEC -.->|w=4| INFRA
    EXT_TEST -.->|w=4| INFRA
    EXT_WEB_CAT -.->|w=3| INFRA

    %% Outbound (downward, not in L3↔L3 graph)
    DOWN["contracts / core / engine<br/>(L0/L1/L2 — outbound,<br/>0 L3↔L3 edges)"]:::other
    INFRA -.->|downward only| DOWN
    SOURCES -.->|downward only| DOWN
    SINKS -.->|downward only| DOWN
    TRANSFORMS -.->|downward only| DOWN
```

**Key observations from this view:**

- All three client sub-packages (sources, sinks, transforms) terminate on `infrastructure/`. The F3 reading-order rule is structural, not preferential: edge weights 45 / 40 / 17 confirm `infrastructure/` is the only common dependency.
- Inbound L3↔L3 traffic to plugins/ all targets `infrastructure/` (or sources/ in the cli case). No external cluster reaches into transforms/ or sinks/ at L3↔L3 granularity.
- Outbound L3↔L3 from plugins/ is **zero**. plugins/ is a sink in the L3 graph; downward dependencies (contracts/core/engine) are out-of-graph.

---

## §2. Component view — `plugins/infrastructure/`

The spine, expanded one level. Per Δ L2-3 this is **flag-and-stop** depth — the diagram reproduces the sub-package layout but does not enumerate per-class component coupling (that requires opening file bodies past the L2 depth cap).

```mermaid
flowchart TB
    classDef root fill:#fde68a,stroke:#b45309,stroke-width:2px,color:#000
    classDef sub fill:#fef3c7,stroke:#92400e,stroke-width:1px,color:#000
    classDef nested fill:#fef9c3,stroke:#a16207,stroke-width:1px,stroke-dasharray:3 3,color:#000

    subgraph INFRA["plugins/infrastructure/"]
      ROOT["root files — 16 files, 3,804 LOC<br/>__init__.py (1 LOC)<br/>hookspecs.py (72) — pluggy specs<br/>manager.py (321) — PluginManager<br/>discovery.py (337) — folder-scan discovery<br/>base.py (1,159) — BaseSource/Transform/Sink<br/>config_base.py (544) — Pydantic config bases<br/>validation.py, schema_factory.py,<br/>probe_factory.py, results.py,<br/>output_paths.py, display_headers.py,<br/>sentinels.py, templates.py, utils.py,<br/>azure_auth.py"]:::root

      CLIENTS["clients/ — 9 files, 3,790 LOC<br/>__init__.py (89) — re-export AuditedLLM/HTTP<br/>base.py (127) — AuditedClientBase<br/>http.py (854) — AuditedHTTPClient (SSRF-safe)<br/>llm.py (719) — AuditedLLMClient<br/>replayer.py (290) — CallReplayer<br/>dataverse.py, fingerprinting.py,<br/>json_utils.py, verifier.py"]:::sub

      RETRIEVAL["clients/retrieval/<br/>6 files, 1,031 LOC<br/>azure_search.py, chroma.py,<br/>connection.py, types.py,<br/>base.py, __init__.py"]:::nested

      BATCHING["batching/ — 4 files, 1,024 LOC<br/>__init__.py (37)<br/>mixin.py (497) — BatchTransformMixin<br/>ports.py (82) — input/output Port abstractions<br/>row_reorder_buffer.py"]:::sub

      POOLING["pooling/ — 6 files, 1,133 LOC<br/>__init__.py (18)<br/>executor.py (651) — PooledExecutor + AIMD<br/>config.py, errors.py,<br/>reorder_buffer.py, throttle.py"]:::sub

      ROOT --> CLIENTS
      CLIENTS --> RETRIEVAL
      ROOT --> BATCHING
      ROOT --> POOLING
    end

    %% Dependency arrows summarising cited edges
    SCC_EDGE["transforms/llm/providers<br/>→ infrastructure/clients<br/>(w=12)"]
    RAG_EDGE["transforms/rag<br/>→ infrastructure/clients/retrieval<br/>(w=9)"]

    SCC_EDGE -.->|cite| CLIENTS
    RAG_EDGE -.->|cite| RETRIEVAL
```

**Key observations:**

- Three structural sub-packages (`clients/`, `batching/`, `pooling/`) plus a nested `clients/retrieval/`. Each has a clear single-responsibility framing visible in its `__init__.py` docstring.
- `clients/retrieval/` is the **only** nested sub-sub-package in the cluster; it exists because retrieval clients (Azure Cognitive Search, Chroma) share enough surface to factor into a sub-package, distinct from the LLM and HTTP clients above it.
- Root-level files are heterogeneous (16 files spanning hookspecs, manager, discovery, base classes, config, validation, factories, sentinels, utilities). `base.py` (1,159 LOC) is the largest single file — under the 1,500 L3-deep-dive threshold but a candidate for the architecture-pack pass to consider splitting.

---

## §3. SCC #1 callout — `transforms/llm` ↔ `transforms/llm/providers`

Per Δ L2-7, surfacing the cycle structure without prescribing decomposition.

```mermaid
flowchart LR
    classDef sccnode fill:#fecaca,stroke:#b91c1c,stroke-width:2px,color:#000
    classDef shared fill:#fed7aa,stroke:#9a3412,stroke-width:1px,color:#000
    classDef other fill:#ffffff,stroke:#9ca3af,stroke-width:1px,stroke-dasharray:3 3,color:#000

    subgraph SCC["SCC #1 — module-level cycle"]
      LLM["plugins/transforms/llm/<br/>(transform.py, base.py, provider.py,<br/>validation.py, tracing.py, templates.py,<br/>multi_query.py, langfuse.py)"]:::sccnode
      PROVIDERS["plugins/transforms/llm/providers/<br/>(__init__.py, azure.py, openrouter.py)"]:::sccnode
    end

    LLM -->|"transform.py:64-65<br/>imports AzureLLMProvider,<br/>OpenRouterLLMProvider<br/>(forward edge w=10+5)"| PROVIDERS
    PROVIDERS -->|"providers/{azure,openrouter}.py:23-25 / 35-37<br/>imports LLMConfig (base.py),<br/>LLMQueryResult / FinishReason (provider.py),<br/>reject_nonfinite_constant (validation.py),<br/>AzureAITracingConfig (tracing.py)<br/>(reverse edge — closes SCC)"| LLM

    %% Outbound from SCC to spine (not in cycle)
    INFRA_CLIENTS["plugins/infrastructure/clients/<br/>(spine — outside SCC)"]:::shared
    INFRA["plugins/infrastructure/<br/>(spine — outside SCC)"]:::shared

    PROVIDERS -->|w=12| INFRA_CLIENTS
    LLM -->|w=17| INFRA
```

**Key observations from this view:**

- The cycle is **module-level** (import-time), not class-level. Neither side reaches into the other's class hierarchy via attribute access at runtime.
- `transforms/llm/transform.py` documents that "Provider instantiation is deferred to `on_start()`" — runtime coupling is decoupled from import-time coupling.
- Outbound edges from both SCC nodes terminate on the `infrastructure/` spine (consistent with F3); the cycle does not pull in any other cluster.

**Cycle composition (file:line citations):**

| Direction | Site | Imported names |
|---|---|---|
| `llm` → `llm/providers` | `transforms/llm/transform.py:64` | `AzureLLMProvider`, `AzureOpenAIConfig`, `_configure_azure_monitor` |
| `llm` → `llm/providers` | `transforms/llm/transform.py:65` | `OpenRouterConfig`, `OpenRouterLLMProvider` |
| `llm/providers` → `llm` | `transforms/llm/providers/azure.py:23` | `LLMConfig` |
| `llm/providers` → `llm` | `transforms/llm/providers/azure.py:24` | `FinishReason`, `LLMQueryResult`, `parse_finish_reason` |
| `llm/providers` → `llm` | `transforms/llm/providers/azure.py:25` | `AzureAITracingConfig`, `TracingConfig` |
| `llm/providers` → `llm` | `transforms/llm/providers/openrouter.py:35` | `LLMConfig` |
| `llm/providers` → `llm` | `transforms/llm/providers/openrouter.py:36` | `LLMQueryResult`, `parse_finish_reason` |
| `llm/providers` → `llm` | `transforms/llm/providers/openrouter.py:37` | `reject_nonfinite_constant` |

---

## §4. Reading-order graph (F3 confirmation)

A simple dependency-flow diagram showing F3's structural justification at a glance.

```mermaid
flowchart LR
    classDef step1 fill:#fde68a,stroke:#b45309,stroke-width:3px,color:#000
    classDef step2 fill:#fef3c7,stroke:#92400e,stroke-width:1px,color:#000

    READ1["1. Read first:<br/>plugins/infrastructure/"]:::step1
    READ2A["2a. Sinks/<br/>(w=45 → infra)"]:::step2
    READ2B["2b. Transforms/<br/>(w=40 → infra)"]:::step2
    READ2C["2c. Sources/<br/>(w=17 → infra)"]:::step2

    READ1 -->|"informs catalog<br/>spine patterns"| READ2A
    READ1 -->|"informs catalog<br/>spine patterns"| READ2B
    READ1 -->|"informs catalog<br/>spine patterns"| READ2C
```

This pass executed in this order; the catalog reflects the order; client-side entries cite spine patterns without re-deriving them.
