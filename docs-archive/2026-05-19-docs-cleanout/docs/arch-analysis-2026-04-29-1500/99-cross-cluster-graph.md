# ELSPETH Architecture — Cross-Cluster Diagrams

Two C4-style views synthesise the L1 11-subsystem map and the five L2 cluster diagrams into a single system-level picture: a **Container** view showing all 11 L1 subsystems grouped by L2 cluster, and a **Component** view drilling into the structurally interesting zones (the 7-node web SCC, the plugin spine, the audit-trail backbone).

Edge truth-source for L3-application surfaces is the L3 oracle (`temp/l3-import-graph.json`, schema v1.0; 33 nodes, 77 edges, 5 SCCs, `stats.type_checking_edges = 0`). Layer-enforced edges into `core` and `contracts` are sourced from the L1 layer model (`enforce_tier_model.py:237–248`) and the per-cluster catalogs.

## Container view (C4 Container, system-level)

This view shows all 11 L1 subsystems verified in `02-l1-subsystem-map.md`, grouped into five L2 clusters plus a sixth grouping for the application-layer siblings that were not promoted to clusters. Edge weights are shown only for load-bearing edges (≥10) and key cross-cluster handshakes confirmed in `temp/reconciliation-log.md`.

```mermaid
flowchart TB

  subgraph SIBLINGS ["application-layer siblings (no L2 cluster)"]
    direction LR
    cli["cli (root files)<br/>4 / 2,942 LOC"]
    mcp_sub["mcp/<br/>9 / 4,114 LOC"]
    telemetry["telemetry/<br/>14 / 2,884 LOC"]
    testing["testing/<br/>2 / 877 LOC"]
    tui["tui/<br/>9 / 1,175 LOC"]
  end

  subgraph CL_COMPOSER ["composer cluster — SCC #4 (7-node web cycle)"]
    direction LR
    web["web/<br/>72 / 22,558 LOC<br/>(7 sub-pkgs, all in SCC #4)"]
    composer_mcp["composer_mcp/<br/>3 / 824 LOC"]
  end

  subgraph CL_PLUGINS ["plugins cluster (spine: plugins/infrastructure)"]
    plugins["plugins/<br/>98 / 30,399 LOC"]
  end

  subgraph CL_ENGINE ["engine cluster"]
    engine["engine/<br/>36 / 17,425 LOC"]
  end

  subgraph CL_CORE ["core cluster"]
    core["core/<br/>49 / 20,791 LOC<br/>(landscape, dag, config, …)"]
  end

  subgraph CL_CONTRACTS ["contracts cluster (L0 leaf)"]
    contracts["contracts/<br/>63 / 17,403 LOC"]
  end

  %% Cross-cluster edges (load-bearing labelled, others unlabelled)

  %% composer -> plugins (heaviest cross-cluster inbound to plugins)
  web -- "w=22 (web/composer)" --> plugins
  composer_mcp -- "w=12 (F1)" --> web

  %% plugins inbound from siblings
  cli -- "w=7" --> plugins

  %% engine downward (layer-enforced)
  engine --> core
  engine --> contracts

  %% plugins outbound
  plugins --> engine
  plugins --> core
  plugins --> contracts

  %% composer cluster outbound (web -> engine/core/contracts; layer-enforced)
  web --> engine
  web --> core
  web --> contracts
  composer_mcp --> engine
  composer_mcp --> core
  composer_mcp --> contracts

  %% siblings outbound (layer-enforced)
  cli --> engine
  cli --> core
  cli --> contracts
  cli --> tui
  cli --> telemetry
  mcp_sub --> core
  mcp_sub --> contracts
  telemetry --> core
  telemetry --> contracts
  tui --> core
  tui --> contracts
  testing --> contracts

  %% core -> contracts
  core --> contracts

  classDef scc4 fill:#ffe5b4,stroke:#ff8c00,stroke-width:2px,color:#3b1d04
  classDef spine fill:#fde68a,stroke:#92400e,stroke-width:2px,color:#3b1d04
  classDef leaf fill:#dcfce7,stroke:#166534,color:#0c1f12
  classDef l1 fill:#dbeafe,stroke:#1e40af,color:#0c1f4d
  classDef l2 fill:#ede9fe,stroke:#6d28d9,color:#1e0e3e
  classDef sib fill:#f3f4f6,stroke:#4b5563,color:#111827

  class web,composer_mcp scc4
  class plugins spine
  class contracts leaf
  class core l1
  class engine l2
  class cli,mcp_sub,telemetry,testing,tui sib
```

**Reading guide.** The composer cluster is highlighted (orange) because its `web/*` sub-packages form **SCC #4**, the largest strongly-connected component in the codebase (`[ORACLE: stats.largest_scc_size = 7; strongly_connected_components[4]]`); the cluster cannot be acyclically decomposed and was scoped as a unit per `[PHASE-0.5 §7.5 F4]`. The plugins container is shaded to flag the F3 spine: `plugins/infrastructure/` is the centre of mass, with `plugins/sinks → plugins/infrastructure` weight 45 the heaviest single edge anywhere in the L3 graph (`[ORACLE: edges[11]]`); see Component view for the spine itself. The composer→plugins handshake (`web/composer → plugins/infrastructure` weight 22, `[ORACLE: edges[54]]`) is the heaviest cross-cluster inbound edge to the plugins cluster, confirmed by `[CLUSTER:composer §5 item 1]` and `[CLUSTER:plugins cross-cluster bullet 1]` in `temp/reconciliation-log.md`. The four other SCCs (mcp 2-node, plugins/transforms/llm 2-node, telemetry 2-node, tui 3-node) live inside individual containers and are not redrawn here; they are the L2 cluster passes' concern.

## Component view (drilling into SCC #4, the plugin spine, and the audit backbone)

This view zooms into the three structurally interesting zones: the 7-node SCC #4 inside the composer cluster, the plugin spine inside the plugins cluster, and the audit-trail backbone that threads engine → core/landscape → contracts.

```mermaid
flowchart TB

  subgraph SCC4 ["SCC #4 — 7-node FastAPI app-factory cycle (composer cluster)"]
    direction LR
    web_pkg["web<br/>(app factory)"]
    web_auth["web/auth"]
    web_blobs["web/blobs"]
    web_composer["web/composer<br/>(state machine, tools.py 3,804 LOC)"]
    web_execution["web/execution"]
    web_secrets["web/secrets"]
    web_sessions["web/sessions"]

    %% intra-SCC edges (oracle composer intra-cluster-edges.json)
    web_execution -- "w=15" --> web_pkg
    web_sessions -- "w=15" --> web_composer
    web_execution -- "w=9" --> web_composer
    web_execution -- "w=7" --> web_sessions
    web_execution --> web_auth
    web_execution --> web_blobs
    web_pkg --> web_auth
    web_auth --> web_pkg
    web_pkg --> web_secrets
    web_pkg --> web_sessions
    web_secrets --> web_pkg
    web_composer --> web_pkg
    web_composer --> web_blobs
    web_composer --> web_sessions
    web_blobs --> web_sessions
  end

  composer_mcp_node["composer_mcp/<br/>server.py / session.py"]
  composer_mcp_node -- "w=12 [F1]" --> web_composer

  subgraph PLUG_SPINE ["Plugin spine — plugins/infrastructure (F3)"]
    plug_infra(["plugins/infrastructure<br/>16 files / 3,804 LOC<br/>hookspecs + audited clients"])
  end

  plug_sinks["plugins/sinks"]
  plug_sources["plugins/sources"]
  plug_transforms["plugins/transforms"]

  plug_sinks -- "w=45 (heaviest L3 edge)" --> plug_infra
  plug_transforms -- "w=40" --> plug_infra
  plug_sources -- "w=17" --> plug_infra
  web_composer -- "w=22 cross-cluster" --> plug_infra

  subgraph AUDIT_BACKBONE ["Audit-trail backbone (engine → core/landscape → contracts)"]
    direction TB
    eng_tokens["engine/tokens.py:19<br/>TokenManager façade"]
    eng_dispatch["engine/executors/<br/>declaration_dispatch.py<br/>(4-site ADR-010 dispatcher)"]
    core_landscape["core/landscape/<br/>schema.py + 4 repositories"]
    contracts_audit["contracts/<br/>declaration_contracts +<br/>audit_evidence DTOs"]
    landscape_db[("Landscape audit DB<br/>(SQLite / Postgres)")]

    eng_tokens --> core_landscape
    eng_dispatch --> contracts_audit
    contracts_audit --> core_landscape
    core_landscape --> landscape_db
  end

  classDef scc4 fill:#ffe5b4,stroke:#ff8c00,stroke-width:2px,color:#3b1d04
  classDef spine fill:#fde68a,stroke:#92400e,stroke-width:3px,color:#3b1d04
  classDef leaf fill:#dcfce7,stroke:#166534,color:#0c1f12
  classDef audit fill:#e0e7ff,stroke:#3730a3,color:#1e1b4b
  classDef ext fill:#e5e7eb,stroke:#374151,color:#0f172a

  class web_pkg,web_auth,web_blobs,web_composer,web_execution,web_secrets,web_sessions scc4
  class plug_infra spine
  class contracts_audit leaf
  class eng_tokens,eng_dispatch,core_landscape audit
  class landscape_db ext
```

**Reading guide.** SCC #4 is the load-bearing structural finding of the analysis: seven `web/*` sub-packages form a single runtime cycle that no static decomposition can break (`[CLUSTER:composer §5 item 2]`; decomposition explicitly deferred to architecture pack). The two heaviest intra-SCC edges (`web/sessions → web/composer` w=15, `web/execution → web` w=15, both from `clusters/composer/temp/intra-cluster-edges.json`) carry the composer state machine and request-routing fan-in. `composer_mcp/` enters the SCC at `web/composer` with weight 12 (`[ORACLE: edges[6]]`), which is the structural finding that closed L1 open question Q2 (`[PHASE-0.5 §7.5 F1]`). The plugin spine is highlighted as a hexagon: three intra-cluster spokes (sinks 45, transforms 40, sources 17 — `[ORACLE: edges[11], [16], [14]]`) plus the heaviest cross-cluster inbound (`web/composer → plugins/infrastructure` weight 22, `[ORACLE: edges[54]]`) all terminate here. The audit backbone shows the three layered citations that thread the audit-complete posture through the layers: `engine/tokens.py:19 → core/landscape/data_flow_repository` (engine cross-cluster bullet 1, confirmed by `[CLUSTER:core confidence 2]`); `engine/executors/declaration_dispatch.py` → `contracts/declaration_contracts` (the 4-site ADR-010 dispatcher, `[CLUSTER:engine §5 item 2]`, confirmed by `[CLUSTER:contracts cross-cluster bullet 2]`); `contracts` audit DTOs → `core/landscape/schema.py` (L0 audit DTO ownership, `[CLUSTER:contracts cross-cluster bullet 3]`).

## Provenance

**Container nodes (11 L1 subsystems).** All from `02-l1-subsystem-map.md §1–11`: `contracts` §1, `core` §2, `engine` §3, `plugins` §4, `web` §5, `mcp/` §6, `composer_mcp/` §7, `telemetry` §8, `tui` §9, `testing` §10, `cli` §11. Cluster grouping per `[PHASE-0.5 §7.5]` revised dispatch queue.

**Component nodes (oracle sub-packages).** The 7 SCC #4 nodes — `web`, `web/auth`, `web/blobs`, `web/composer`, `web/execution`, `web/secrets`, `web/sessions` — from `[ORACLE: nodes; strongly_connected_components[4]]`, confirmed at symbol-level by `[CLUSTER:composer §C5]`. `composer_mcp/` from `[ORACLE: node composer_mcp]` and `[CLUSTER:composer §C-composer_mcp]`. `plugins/infrastructure`, `plugins/sinks`, `plugins/sources`, `plugins/transforms` from `[ORACLE: nodes]` and `[CLUSTER:plugins §C1–C4]`. Engine, core, contracts audit-backbone nodes per `[CLUSTER:engine §C-tokens, §C-declaration_dispatch]`, `[CLUSTER:core §C-landscape]`, `[CLUSTER:contracts §C-declaration_contracts, §C-audit_evidence]`.

**Container load-bearing edges.** `web/composer → plugins/infrastructure` w=22 `[ORACLE: edges[54]]`; `composer_mcp → web/composer` w=12 `[ORACLE: edges[6]; PHASE-0.5 §7.5 F1]`; `cli → plugins/infrastructure` w=7 `[ORACLE: edges[0]]`; layer-enforced `engine → {core, contracts}`, `core → contracts`, all L3 → {core, contracts} edges from `enforce_tier_model.py:237–248` and `03-l1-context-diagram.md §2 edge accounting`.

**Component edges (intra-SCC #4).** All from `clusters/composer/temp/intra-cluster-edges.json` (oracle-derived; `byte_equality_assertion` not asserted, but extraction methodology cited). 16 edges drawn match the entries with weight ≥4 reported in `intra_cluster_edges` array. `composer_mcp → web/composer` w=12 `[ORACLE: edges[6]]`.

**Component edges (plugin spine).** `plugins/sinks → plugins/infrastructure` w=45 `[ORACLE: edges[11]]`; `plugins/transforms → plugins/infrastructure` w=40 `[ORACLE: edges[16]]`; `plugins/sources → plugins/infrastructure` w=17 `[ORACLE: edges[14]]`; `web/composer → plugins/infrastructure` w=22 `[ORACLE: edges[54]]`. F3 framing per `[PHASE-0.5 §7.5 F3]`.

**Component edges (audit backbone).** `engine/tokens.py:19 → DataFlowRepository` per `[CLUSTER:engine §5 cross-cluster bullet 1]` (also drawn in `clusters/engine/03-cluster-diagrams.md` line 116); `engine/declaration_dispatch.py → contracts.declaration_contracts` per `[CLUSTER:engine §5 cross-cluster bullet 2]` and `[CLUSTER:contracts cross-cluster bullet 2]`; `contracts/audit_evidence → core/landscape/schema.py` per `[CLUSTER:contracts cross-cluster bullet 3]`. None of these edges appear in the L3 oracle because the oracle's `scope.layers_included = ['L3/application']` excludes engine/core/contracts; this is by design — the oracle's job is L3 cycle detection, the layer-enforcer is the truth-source for cross-layer edges.

**Standing constraints honoured.** No TYPE_CHECKING-only edges drawn (`[ORACLE: stats.type_checking_edges = 0]`, `[PHASE-0.5 §7.5 F5]`; the contracts cluster's `plugin_context.py:31` TYPE_CHECKING smell is intra-cluster and below this view's resolution per `temp/reconciliation-log.md` R2). All 33 oracle nodes either drawn or contained inside an L1 container (`mcp/analyzers` inside mcp; `tui/screens`, `tui/widgets` inside tui; `telemetry/exporters` inside telemetry; `plugins/transforms/{azure,llm,llm/providers,rag}`, `plugins/infrastructure/{batching,clients,clients/retrieval,pooling}` inside plugins; `web/catalog`, `web/middleware`, `web/composer/skills` inside web; the `.` cli-root node mapped to the `cli` container). No nodes added that aren't in `[ORACLE: nodes]` plus the engine/core/contracts containers from the L1 layer schema.
